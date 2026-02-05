import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import tempfile
import shutil

from dataset import HarvestDataset
from model import UNet
from convert_to_rgb import RGBConverter


def train_epoch(model, loader, opt, criterion, device):
	model.train()
	epoch_loss = 0.0
	epoch_bce = 0.0
	epoch_dice = 0.0
	for imgs, masks in loader:
		imgs = imgs.to(device)
		masks = masks.to(device)
		opt.zero_grad()
		out = model(imgs)
		# 输出形状: (B,1,H,W)，使用BCEWithLogits损失
		total_loss, bce_loss, dice_loss = criterion(out, masks)
		total_loss.backward()

		# 梯度裁剪，防止梯度爆炸
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

		opt.step()
		epoch_loss += total_loss.item() * imgs.size(0)
		epoch_bce += bce_loss.item() * imgs.size(0)
		epoch_dice += dice_loss.item() * imgs.size(0)
	return epoch_loss / len(loader.dataset), epoch_bce / len(loader.dataset), epoch_dice / len(loader.dataset)


def evaluate_miou(model, loader, device, threshold=0.3):
	"""
	评估二分类语义分割的平均交并比(mIoU)和其他指标
	"""
	model.eval()
	total_iou_bg = 0.0  # 背景IoU
	total_iou_fg = 0.0  # 前景IoU
	total_precision = 0.0  # 精确率
	total_recall = 0.0     # 召回率
	total_f1 = 0.0         # F1分数
	num_samples = 0
	valid_samples = 0  # 有前景的样本数

	with torch.no_grad():
		for imgs, masks in loader:
			imgs = imgs.to(device)
			masks = masks.to(device)

			outputs = model(imgs)
			# 将logits转换为概率，然后转换为二值预测
			preds = (torch.sigmoid(outputs) > threshold).float()

			# 转换为numpy以便更容易计算
			preds = preds.cpu().numpy()
			masks = masks.cpu().numpy()

			for pred, mask in zip(preds, masks):
				pred = pred.squeeze()  # (H, W)
				mask = mask.squeeze()  # (H, W)

				# 计算背景类IoU（类别0）
				pred_bg = (pred == 0)
				mask_bg = (mask == 0)
				intersection_bg = np.logical_and(pred_bg, mask_bg).sum()
				union_bg = np.logical_or(pred_bg, mask_bg).sum()
				iou_bg = intersection_bg / union_bg if union_bg > 0 else 1.0

				# 计算前景类IoU（类别1）
				pred_fg = (pred == 1)
				mask_fg = (mask == 1)
				intersection_fg = np.logical_and(pred_fg, mask_fg).sum()
				union_fg = np.logical_or(pred_fg, mask_fg).sum()

				# 只有当有前景像素时才计算前景IoU
				if mask_fg.sum() > 0:  # 真值有前景
					iou_fg = intersection_fg / union_fg if union_fg > 0 else 0.0
					valid_samples += 1
				else:
					iou_fg = 0.0  # 空mask不计算IoU

				# 计算精确率、召回率、F1（只对有前景的样本）
				if mask_fg.sum() > 0:
					precision = intersection_fg / pred_fg.sum() if pred_fg.sum() > 0 else 0.0
					recall = intersection_fg / mask_fg.sum() if mask_fg.sum() > 0 else 0.0
					f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

					total_precision += precision
					total_recall += recall
					total_f1 += f1

				total_iou_bg += iou_bg
				total_iou_fg += iou_fg
				num_samples += 1

	# 计算平均值
	mean_iou_bg = total_iou_bg / num_samples
	mean_iou_fg = total_iou_fg / num_samples if num_samples > 0 else 0.0

	# 只对有前景的样本计算其他指标
	if valid_samples > 0:
		mean_precision = total_precision / valid_samples
		mean_recall = total_recall / valid_samples
		mean_f1 = total_f1 / valid_samples
	else:
		mean_precision = mean_recall = mean_f1 = 0.0

	miou = (mean_iou_bg + mean_iou_fg) / 2.0

	return miou, mean_iou_bg, mean_iou_fg, mean_precision, mean_recall, mean_f1


def calculate_class_weights(loader, device):
	"""
	计算训练集中前景和背景类别的权重，用于平衡类别不平衡
	返回pos_weight（用于BCE Loss）和dice_weights（用于Dice Loss）
	"""
	total_pixels = 0
	total_fg_pixels = 0
	total_bg_pixels = 0

	with torch.no_grad():
		for _, masks in loader:
			masks = masks.to(device)
			masks_flat = masks.view(-1)
			total_pixels += masks_flat.numel()
			total_fg_pixels += (masks_flat == 1).sum().item()
			total_bg_pixels += (masks_flat == 0).sum().item()

	fg_ratio = total_fg_pixels / total_pixels
	bg_ratio = total_bg_pixels / total_pixels

	print(f"📊 Class distribution analysis:")
	print(f"   Total pixels: {total_pixels}")
	print(f"   Background pixels: {total_bg_pixels} ({bg_ratio:.3f})")
	print(f"   Foreground pixels: {total_fg_pixels} ({fg_ratio:.3f})")

	# 计算pos_weight：背景/前景的比例，限制最大值为5.0避免过激平衡
	pos_weight = min(bg_ratio / fg_ratio, 5.0) if fg_ratio > 0 else 1.0

	# 计算Dice权重：用于加权Dice Loss
	dice_weights = torch.tensor([bg_ratio, fg_ratio], dtype=torch.float32).to(device)

	print(f"⚖️  Class weights:")
	print(f"   BCE pos_weight: {pos_weight:.3f}")
	print(f"   Dice weights: BG={bg_ratio:.3f}, FG={fg_ratio:.3f}")

	return pos_weight, dice_weights


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", type=int, default=100)
	parser.add_argument("--batch-size", type=int, default=8)
	parser.add_argument("--lr", type=float, default=5e-4)
	parser.add_argument("--img-size", type=int, default=256)
	parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
	parser.add_argument("--base-channels", type=int, default=32, help="Base channels for UNet")
	parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
	parser.add_argument("--balance-classes", action="store_true", help="Enable class balancing for imbalanced datasets")
	parser.add_argument("--class-balance-weight", type=float, default=1.0, help="Weight multiplier for class balancing (higher values = stronger balancing)")
	parser.add_argument("--threshold", type=float, default=0.4, help="Prediction threshold for binary classification (default: 0.4 for balanced performance)")
	parser.add_argument("--filter-empty-masks", action="store_true", help="Filter out samples with no foreground pixels during training")
	# 默认路径相对于项目根目录（此文件的上一级目录）解析
	default_root = Path(__file__).resolve().parents[1]
	parser.add_argument("--images-dir", default="D:/AIRS/train/收割/images")
	parser.add_argument("--masks-dir", default="D:/AIRS/train/收割/labels")
	parser.add_argument("--save-dir", default="checkpoints")
	return parser.parse_args()


def main():
	args = parse_args()
	os.makedirs(args.save_dir, exist_ok=True)

	device = torch.device(args.device)

	# 初始化临时目录变量
	temp_dir = None

	try:
		# 检查输入目录中的文件类型
		input_dir = Path(args.images_dir)
		tif_files = list(input_dir.glob("*.tif"))
		png_files = list(input_dir.glob("*.png"))
		jpg_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))

		print(f"📁 训练数据目录文件统计:")
		print(f"   TIF文件: {len(tif_files)}")
		print(f"   PNG文件: {len(png_files)}")
		print(f"   JPG文件: {len(jpg_files)}")

		# 确定要使用的图像目录
		images_dir = args.images_dir

		if len(tif_files) > 0:
			print(f"\n🔄 检测到 {len(tif_files)} 个TIF文件，正在转换为PNG格式用于训练...")

			# 创建临时目录
			temp_dir = tempfile.mkdtemp(prefix="train_rgb_")
			temp_path = Path(temp_dir)
			print(f"📂 临时目录: {temp_path}")

			# 使用convert_to_rgb.py的逻辑进行转换
			converter = RGBConverter(str(input_dir), str(temp_path))
			converter.batch_convert(output_format='PNG')

			# 检查转换结果
			converted_pngs = list(temp_path.glob("*.png"))
			print(f"✅ 转换完成，共生成 {len(converted_pngs)} 个PNG文件用于训练")

			if len(converted_pngs) == 0:
				raise ValueError("TIF转PNG失败，没有生成任何PNG文件")

			# 使用转换后的目录
			images_dir = str(temp_path)
		else:
			print("✅ 训练目录中没有TIF文件，直接使用现有PNG/JPG文件")

		dataset = HarvestDataset(images_dir=images_dir, masks_dir=args.masks_dir, img_size=args.img_size, augment=True)

		# 有用的检查
		if len(dataset) == 0:
			print("No images found for training.")
			print("images_dir:", images_dir)
			print("masks_dir:", args.masks_dir)
			raise SystemExit("Dataset is empty — check paths or filenames.")

		# 可选：过滤掉空mask的样本
		if args.filter_empty_masks:
			print("🔍 Filtering out samples with no foreground pixels...")
			valid_indices = []
			for i in range(len(dataset)):
				try:
					_, mask = dataset[i]
					if mask.sum() > 0:  # 有前景像素
						valid_indices.append(i)
				except Exception as e:
					print(f"⚠️  Error checking sample {i}: {e}")
					continue

			from torch.utils.data import Subset
			dataset = Subset(dataset, valid_indices)
			print(f"✅ Filtered dataset: {len(valid_indices)}/{len(dataset)} samples with foreground")

		# 分割数据集：90%训练，10%验证
		train_size = int(0.9 * len(dataset))
		val_size = len(dataset) - train_size
		train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

		print(f"Dataset split: {train_size} train samples, {val_size} validation samples")

		train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
		val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

		model = UNet(n_channels=3, n_classes=1, base_c=args.base_channels).to(device)

	# 使用AdamW优化器和学习率调度，增加权重衰减防止过拟合
	# 由于数据不平衡，使用更小的初始学习率
		opt = optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-4, betas=(0.9, 0.999))

		# 使用带预热的余弦退火学习率调度
		warmup_epochs = max(1, args.epochs // 10)  # 预热轮数为总轮数的1/10
		def lr_lambda(epoch):
			if epoch < warmup_epochs:
				# 预热阶段：线性增加
				return (epoch + 1) / warmup_epochs
			else:
				# 余弦退火阶段
				progress = (epoch - warmup_epochs) / (args.epochs - warmup_epochs)
				return 0.5 * (1 + math.cos(math.pi * progress))

		scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

		# 计算类别权重用于平衡不平衡数据集
		pos_weight, dice_weights = calculate_class_weights(train_loader, device)

		# 使用稳定的组合损失函数：加权BCE + Dice Loss
		class StableDiceBCELoss(nn.Module):
			def __init__(self, pos_weight=1.0, bce_weight=0.5, dice_weight=0.5):
				super(StableDiceBCELoss, self).__init__()
				self.pos_weight = pos_weight
				self.bce_weight = bce_weight
				self.dice_weight = dice_weight

			def forward(self, inputs, targets, smooth=1e-6):
				# 加权BCE Loss - 使用pos_weight平衡类别不平衡
				pos_weight_tensor = torch.tensor(self.pos_weight, dtype=inputs.dtype, device=inputs.device)
				bce = nn.functional.binary_cross_entropy_with_logits(
					inputs, targets,
					pos_weight=pos_weight_tensor
				)

				# Dice Loss
				inputs_sigmoid = torch.sigmoid(inputs)
				inputs_flat = inputs_sigmoid.view(-1)
				targets_flat = targets.view(-1)

				intersection = (inputs_flat * targets_flat).sum()
				dice_coeff = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
				dice_loss = 1.0 - dice_coeff

				# 组合损失：平衡BCE和Dice权重
				total_loss = self.bce_weight * bce + self.dice_weight * dice_loss

				return total_loss, bce, dice_loss

		# 使用正确的Focal Loss + Dice Loss组合
		class FocalDiceLoss(nn.Module):
			def __init__(self, focal_alpha=0.75, focal_gamma=2.0, dice_weight=0.3, class_weights=None):
				super(FocalDiceLoss, self).__init__()
				self.focal_alpha = focal_alpha
				self.focal_gamma = focal_gamma
				self.dice_weight = dice_weight
				self.class_weights = class_weights  # [bg_weight, fg_weight]

			def forward(self, inputs, targets, smooth=1e-6):
				# Focal Loss部分 (标准实现)
				# 将logits转换为概率
				inputs_sigmoid = torch.sigmoid(inputs)

				# 计算BCE loss (不使用pos_weight，避免与Focal Loss冲突)
				bce_loss = nn.functional.binary_cross_entropy(
					inputs_sigmoid, targets, reduction='none'
				)

				# 计算pt (预测正确的概率)
				pt = torch.where(targets == 1, inputs_sigmoid, 1 - inputs_sigmoid)

				# Focal Loss: -α*(1-pt)^γ * log(pt)
				focal_loss = -self.focal_alpha * (1 - pt) ** self.focal_gamma * torch.log(pt + 1e-8)

				# 如果有类别权重，应用权重
				if self.class_weights is not None:
					weights = torch.where(targets == 1,
					self.class_weights[1],  # 前景权重
					self.class_weights[0])  # 背景权重
					focal_loss = focal_loss * weights

				focal_loss = focal_loss.mean()

				# Dice Loss部分
				inputs_flat = inputs_sigmoid.view(-1)
				targets_flat = targets.view(-1)

				intersection = (inputs_flat * targets_flat).sum()
				dice_coeff = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
				dice_loss = 1.0 - dice_coeff

				# 组合损失
				total_loss = focal_loss + self.dice_weight * dice_loss

				return total_loss, focal_loss, dice_loss

		# 使用类别权重而不是pos_weight
		class_weights = [1.0, pos_weight]  # 背景权重1.0，前景权重根据不平衡程度
		criterion = FocalDiceLoss(focal_alpha=0.25, focal_gamma=2.0, dice_weight=0.3, class_weights=class_weights)

		best_miou = 0.0
		best_loss = float('inf')
		patience_counter = 0
		best_epoch = 0

		for epoch in range(1, args.epochs + 1):
			# 训练
			train_loss, train_focal, train_dice = train_epoch(model, train_loader, opt, criterion, device)

			# 验证
			miou, iou_bg, iou_fg, precision, recall, f1 = evaluate_miou(model, val_loader, device, threshold=args.threshold)
			print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} (Focal: {train_focal:.4f}, Dice: {train_dice:.4f})")
			print(f"  Val mIoU: {miou:.4f} (BG: {iou_bg:.4f}, FG: {iou_fg:.4f}) - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

			# 保存每个epoch
			save_path = Path(args.save_dir) / f"unet_epoch{epoch}.pth"
			torch.save({"epoch": epoch, "model_state": model.state_dict(), "opt_state": opt.state_dict()}, save_path)

			# 更新学习率
			scheduler.step()

			# 检查是否有改善（基于F1分数而不是IoU，因为IoU对小目标不公平）
			current_score = f1  # 使用F1分数作为主要指标
			improved = current_score > best_miou  # 复用best_miou变量名
			if improved:
				best_miou = current_score
				best_epoch = epoch
				patience_counter = 0
				torch.save(model.state_dict(), Path(args.save_dir) / "best_unet_S.pth")
				print(f"🎉 New best model saved with F1: {best_miou:.4f}")
			else:
				patience_counter += 1
				print(f"⏳ No improvement for {patience_counter} epochs")

			# 打印当前学习率
			current_lr = opt.param_groups[0]['lr']
			print(f"📈 Current learning rate: {current_lr:.6f}")

			# 早停检查
			if patience_counter >= args.patience:
				print(f"⏹️  Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
				print(f"📊 Best mIoU: {best_miou:.4f} at epoch {best_epoch}")
				break

		# 如果没有触发早停，训练完成
		if patience_counter < args.patience:
			print(f"🏁 Training completed all {args.epochs} epochs")

		print(f"🏆 Final results:")
		print(f"   Best validation mIoU: {best_miou:.4f} (epoch {best_epoch})")
		print(f"   Final learning rate: {opt.param_groups[0]['lr']:.6f}")
		print(f"   Model saved to: {args.save_dir}/best_unet_S.pth")

	except Exception as e:
		print(f"❌ 训练过程中出错: {e}")
		import traceback
		traceback.print_exc()
		return 1
	finally:
		# 清理临时目录（无论是否出错都要执行）
		if temp_dir is not None:
			try:
				shutil.rmtree(temp_dir)
				print(f"🧹 已清理临时目录: {temp_dir}")
			except Exception as e:
				print(f"⚠️  清理临时目录失败: {e}")

	return 0


if __name__ == "__main__":
	main()


