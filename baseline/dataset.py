import os
from glob import glob
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import rasterio
from tqdm import tqdm


class HarvestDataset(Dataset):
	def __init__(self, images_dir="D:/AIRS/train/收割/images", masks_dir="D:/AIRS/train/收割/labels", img_size=256, augment=False, split='train', val_ratio=0.2, seed=42):
		self.images_dir = images_dir
		self.masks_dir = masks_dir
		all_img_paths = sorted(glob(os.path.join(images_dir, "*.png")))

		# 分割数据集
		np.random.seed(seed)
		n_val = int(len(all_img_paths) * val_ratio)
		indices = np.random.permutation(len(all_img_paths))

		if split == 'train':
			selected_indices = indices[n_val:]
		else:  # 'val'
			selected_indices = indices[:n_val]

		self.img_paths = [all_img_paths[i] for i in selected_indices]
		self.img_size = img_size
		self.augment = augment

		# 数据增强变换（仅用于训练）
		if augment and split == 'train':
			self.augment_transform = T.Compose([
				T.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
				T.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
				T.RandomRotation(degrees=30),    # 随机旋转±30度
				T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),  # 颜色增强
			])
		else:
			self.augment_transform = None

		# 图像和掩码的基本变换
		self.img_transform = T.Compose([
			T.Resize((img_size, img_size)),
			T.ToTensor(),
			# 使用ImageNet统计作为合理的默认归一化
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		# 掩码变换：最近邻插值调整大小，然后转换为张量(0/1)
		self.mask_transform = T.Compose([
			T.Resize((img_size, img_size), interpolation=Image.NEAREST),
			T.ToTensor(),
		])

	def __len__(self):
		return len(self.img_paths)

	def _mask_path_for_image(self, img_path):
		# PNG图像文件名可能与原始TIF不同，需要转换回原始名称
		base = os.path.basename(img_path)
		# 移除_rgb后缀（如果有的话）并将扩展名改回.tif
		if base.endswith('.png'):
			# 如果文件名包含_rgb后缀，去掉它
			if '_rgb' in base:
				base = base.replace('_rgb', '')
			# 将扩展名从.png改为.tif
			base = base.replace('.png', '.tif')

		return os.path.join(self.masks_dir, base)

	def _read_rgb_from_png(self, path):
		"""
		读取PNG格式的RGB图像
		"""
		try:
			# 直接使用PIL读取PNG图像
			img = Image.open(path).convert("RGB")
			return img
		except Exception as e:
			print(f"❌ 读取PNG文件失败 {path}: {e}")
			# 返回一个小的黑色图像作为fallback
			return Image.new('RGB', (256, 256), (0, 0, 0))

	def _read_mask_from_tif(self, path):
		with rasterio.open(path) as src:
			mask = src.read(1)
		mapped = np.zeros_like(mask, dtype=np.uint8)
		mapped[mask == 1] = 1
		mapped[mask == 255] = 1
		mapped255 = (mapped * 255).astype(np.uint8)
		return Image.fromarray(mapped255, mode="L")

	def __getitem__(self, idx):
		img_path = self.img_paths[idx]
		mask_path = self._mask_path_for_image(img_path)

		# 现在输入是PNG格式的RGB图像，直接使用PIL读取
		ext = os.path.splitext(img_path)[1].lower()
		if ext == ".png":
			image = self._read_rgb_from_png(img_path)
		else:
			# fallback：尝试直接读取
			image = Image.open(img_path).convert("RGB")

		if not os.path.exists(mask_path):
			raise FileNotFoundError(f"Mask not found for image {img_path} -> expected {mask_path}")
		# 掩码也可能是tif格式
		mask_ext = os.path.splitext(mask_path)[1].lower()
		if mask_ext in [".tif", ".tiff"]:
			mask = self._read_mask_from_tif(mask_path)
		else:
			mask = Image.open(mask_path).convert("L")

		# 应用数据增强（如果启用且为训练模式）
		if self.augment_transform is not None:
			# 将PIL图像转换为tensor进行增强，然后转回PIL
			image_tensor = T.ToTensor()(image)
			mask_tensor = T.ToTensor()(mask)

			# 应用相同的随机变换到图像和掩码
			seed = torch.randint(0, 2**32, (1,)).item()
			torch.manual_seed(seed)
			image_tensor = self.augment_transform(image_tensor)
			torch.manual_seed(seed)
			mask_tensor = self.augment_transform(mask_tensor)

			# 转回PIL图像
			image = T.ToPILImage()(image_tensor)
			# 对于掩码，需要特殊处理
			mask = T.ToPILImage()(mask_tensor.squeeze(0))

		image = self.img_transform(image)
		mask = self.mask_transform(mask)  # float tensor in [0,1]
		mask = (mask > 0.5).float()

		return image, mask


def analyze_pixel_distribution(dataset):
	"""
	分析数据集中的像素分布，检查类别不平衡情况

	参数:
		dataset: HarvestDataset实例

	返回值:
		dict: 包含像素统计信息的字典
	"""
	print(f"开始分析数据集，共 {len(dataset)} 个样本...")

	total_pixels = 0
	foreground_pixels = 0  # 已收割区域 (类别1)
	background_pixels = 0   # 未收割区域 (类别0)

	# 用于存储每张图像的统计信息
	image_stats = []

	for idx in tqdm(range(len(dataset)), desc="分析图像"):
		try:
			_, mask = dataset[idx]  # 只获取mask，不需要图像

			# mask是tensor格式，形状为(1, H, W)，值范围[0,1]
			# 转换为numpy并二值化
			mask_np = (mask.squeeze().numpy() > 0.5).astype(np.uint8)

			# 统计像素
			img_pixels = mask_np.size
			img_foreground = np.sum(mask_np == 1)
			img_background = np.sum(mask_np == 0)

			total_pixels += img_pixels
			foreground_pixels += img_foreground
			background_pixels += img_background

			# 记录每张图像的统计
			image_stats.append({
				'index': idx,
				'total_pixels': img_pixels,
				'foreground_pixels': img_foreground,
				'background_pixels': img_background,
				'foreground_ratio': img_foreground / img_pixels,
				'background_ratio': img_background / img_pixels
			})

		except Exception as e:
			print(f"处理第 {idx} 个样本时出错: {e}")
			continue

	# 计算整体统计
	foreground_ratio = foreground_pixels / total_pixels if total_pixels > 0 else 0
	background_ratio = background_pixels / total_pixels if total_pixels > 0 else 0
	balance_ratio = foreground_pixels / background_pixels if background_pixels > 0 else float('inf')

	# 计算不平衡程度
	if foreground_ratio < 0.1:
		balance_level = "严重不平衡"
		balance_score = 1  # 1-5分制，1分最严重
	elif foreground_ratio < 0.2:
		balance_level = "中度不平衡"
		balance_score = 2
	elif foreground_ratio < 0.3:
		balance_level = "轻度不平衡"
		balance_score = 3
	elif foreground_ratio < 0.4:
		balance_level = "基本平衡"
		balance_score = 4
	else:
		balance_level = "平衡"
		balance_score = 5

	stats = {
		'total_samples': len(dataset),
		'total_pixels': total_pixels,
		'foreground_pixels': foreground_pixels,
		'background_pixels': background_pixels,
		'foreground_ratio': foreground_ratio,
		'background_ratio': background_ratio,
		'balance_ratio': balance_ratio,
		'balance_level': balance_level,
		'balance_score': balance_score,
		'image_stats': image_stats
	}

	return stats


def print_pixel_analysis(stats):
	"""
	打印像素分布分析结果

	参数:
		stats: analyze_pixel_distribution返回的统计字典
	"""
	print("\n" + "="*60)
	print("📊 数据集像素分布分析报告")
	print("="*60)

	print("\n📈 整体统计:")
	print(f"  总样本数: {stats['total_samples']}")
	print(f"  总像素数: {stats['total_pixels']:,}")
	print(f"  前景像素比例: {stats['foreground_ratio']:.4f} ({stats['foreground_ratio']*100:.2f}%)")
	print(f"  背景像素比例: {stats['background_ratio']:.4f} ({stats['background_ratio']*100:.2f}%)")
	print(f"  前景/背景比例: {stats['balance_ratio']:.4f}")

	print("\n⚖️  平衡程度评估:")
	print(f"  评估等级: {stats['balance_level']} (评分: {stats['balance_score']}/5)")
	print(f"  建议: {get_balance_suggestion(stats)}")

	print("\n🔍 详细信息:")	# 找出前景比例最高的和最低的几张图像
	image_stats = stats['image_stats']
	if image_stats:
		# 按前景比例排序
		sorted_by_fg = sorted(image_stats, key=lambda x: x['foreground_ratio'], reverse=True)

		print("  前景比例最高的5张图像:")
		for i, img_stat in enumerate(sorted_by_fg[:5]):
			print(f"    图像 {img_stat['index']}: 前景比例 {img_stat['foreground_ratio']:.4f}")
		print("  前景比例最低的5张图像:")
		for i, img_stat in enumerate(sorted_by_fg[-5:]):
			print(f"    图像 {img_stat['index']}: 前景比例 {img_stat['foreground_ratio']:.4f}")
	print("="*60 + "\n")


def get_balance_suggestion(stats):
	"""
	根据统计结果给出平衡性建议

	参数:
		stats: 统计字典

	返回值:
		str: 建议文本
	"""
	fg_ratio = stats['foreground_ratio']

	if fg_ratio < 0.05:
		return "极度不平衡！前景像素占比过低，建议：1) 数据增强(过采样前景区域) 2) 使用加权损失函数 3) 考虑收集更多已收割区域数据"
	elif fg_ratio < 0.1:
		return "严重不平衡！前景像素占比很低，建议：1) 使用Focal Loss或Dice Loss 2) 类别权重调整 3) 数据平衡采样"
	elif fg_ratio < 0.2:
		return "中等不平衡，前景像素偏少，建议：1) 轻微权重调整 2) 数据增强 3) 考虑Dice系数损失"
	elif fg_ratio < 0.4:
		return "基本平衡，但仍有改善空间，建议：1) 继续使用当前设置 2) 可考虑Dice Loss辅助"
	else:
		return "数据集相对平衡，可以正常训练"


# 使用示例
if __name__ == "__main__":
	# 创建数据集实例
	dataset = HarvestDataset()

	# 进行像素分布分析
	stats = analyze_pixel_distribution(dataset)

	# 打印分析报告
	print_pixel_analysis(stats)