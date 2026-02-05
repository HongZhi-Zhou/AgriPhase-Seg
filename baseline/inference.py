#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量推理脚本
使用训练好的UNet模型对测试数据集进行预测并生成掩码
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import subprocess
import sys
import re

from segmentation.dataset import HarvestDataset
from segmentation.model import UNet
from segmentation.convert_to_rgb import RGBConverter
import tempfile
import shutil


class ModelPredictor:
    def __init__(self, model_path, device='auto', base_channels=None):
        """
        初始化预测器

        Args:
            model_path: 模型权重文件路径
            device: 计算设备 ('auto', 'cpu', 'cuda')
        """
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"使用设备: {self.device}")

        # 加载模型 - 使用与训练时相同的参数
        print(f"使用base_c={base_channels}初始化模型")
        self.model = UNet(n_channels=3, n_classes=1, base_c=base_channels)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        print(f"✅ 模型加载成功: {model_path}")

    def predict_single_image(self, image_tensor):
        """
        预测单张图像

        Args:
            image_tensor: 输入图像张量 (C, H, W)

        Returns:
            numpy.ndarray: 预测掩码 (H, W)
        """
        with torch.no_grad():
            # 添加批次维度并移动到设备
            input_tensor = image_tensor.unsqueeze(0).to(self.device)

            # 前向传播
            output = self.model(input_tensor)

            # 应用sigmoid并转换为二值掩码
            prob = torch.sigmoid(output)
            pred_mask = (prob > 0.4).float()  # 使用与训练评估一致的阈值

            # 移除批次维度并转换为numpy
            pred_mask = pred_mask.squeeze(0).squeeze(0).cpu().numpy()

            return pred_mask

    def predict_dataset(self, dataset, output_dir, batch_size=8):
        """
        批量预测整个数据集

        Args:
            dataset: 数据集对象
            output_dir: 输出目录
            batch_size: 批处理大小
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"开始批量预测 {len(dataset)} 个样本...")

        success_count = 0

        for idx in tqdm(range(len(dataset)), desc="预测进度"):
            try:
                # 获取图像和真实掩码
                image, true_mask = dataset[idx]

                # 预测
                pred_mask = self.predict_single_image(image)

                # 生成输出文件名（基于原始图像名）
                img_path = dataset.img_paths[idx]
                img_stem = Path(img_path).stem

                # 如果有_rgb后缀，去掉它
                if '_rgb' in img_stem:
                    img_stem = img_stem.replace('_rgb', '')

                output_filename = f"{img_stem}.png"
                output_path = output_dir / output_filename

                # 将预测掩码转换为PIL图像并保存
                # 调换含义：pred_mask: 0=未收获, 1=已收获
                # 用户要求：0对应掩码值1(已收获), 255对应掩码值2(未收获)
                # 所以需要取反：1-已收获 → 0, 0-未收获 → 1，然后乘以255
                inverted_mask = 1 - pred_mask  # 0→1, 1→0
                pred_mask_uint8 = (inverted_mask * 255).astype(np.uint8)
                pred_image = Image.fromarray(pred_mask_uint8, mode='L')
                pred_image.save(output_path)

                success_count += 1

            except Exception as e:
                print(f"❌ 处理样本 {idx} 时出错: {e}")
                continue

        print("" + "="*60)
        print("📊 批量预测完成统计")
        print("="*60)
        print(f"总样本数: {len(dataset)}")
        print(f"✅ 预测成功: {success_count}")
        print(f"❌ 预测失败: {len(dataset) - success_count}")

        print(f"💾 输出目录: {output_dir}")

        return success_count


def run_evaluation(pred_dir, true_dir):
    """
    运行评估脚本并提取mIoU值

    Args:
        pred_dir: 预测结果目录
        true_dir: 真实标签目录

    Returns:
        float: mIoU值，如果失败则返回None
    """
    try:
        # 构建评估命令
        temp_output = "temp_evaluation_results.txt"
        cmd = [
            sys.executable, "segmentation/evaluate.py",
            "--pred-dir", pred_dir,
            "--true-dir", true_dir,
            "--output-file", temp_output,
            "--plot-file", "nul" if os.name == 'nt' else "/dev/null"  # Windows使用nul，Linux使用/dev/null
        ]

        # 运行评估脚本
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="D:/PythonProject")

        if result.returncode == 0:
            # 从输出中提取mIoU值
            output = result.stdout
            # 查找mIoU行
            miou_match = re.search(r'平均IoU:\s+([\d.]+)', output)
            if miou_match:
                miou_value = float(miou_match.group(1))
                return miou_value
            else:
                print("⚠️ 无法从评估输出中提取mIoU值")
                print("评估输出:", output[:500])  # 显示前500个字符用于调试
                return None
        else:
            print(f"❌ 评估脚本执行失败: {result.stderr}")
            return None

    except Exception as e:
        print(f"❌ 运行评估时出错: {e}")
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用训练好的模型进行批量推理')
    parser.add_argument('--model-path', type=str, default='checkpoints/best_unet_S.pth',
                       help='模型权重文件路径')
    parser.add_argument('--images-dir', type=str,
                       default='D:/AIRS/test/清雪扣棚/images',
                       help='测试图像目录')
    parser.add_argument('--masks-dir', type=str,
                       default='D:/AIRS/test_label/task3',
                       help='真实掩码目录')
    parser.add_argument('--output-dir', type=str, default='D:/AIRS/prediction/task3',
                       help='预测结果输出目录')
    parser.add_argument('--device', type=str, default='auto',
                       help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='批处理大小')
    parser.add_argument('--base-channels', type=int, default=32,
                       help='模型的基础通道数（默认32，与训练时保持一致）')

    args = parser.parse_args()

    print("🔮 UNet批量推理工具")
    print("="*50)
    print(f"📁 模型路径: {args.model_path}")
    print(f"🖼️  图像目录: {args.images_dir}")
    print(f"🎯 掩码目录: {args.masks_dir}")
    print(f"💾 输出目录: {args.output_dir}")
    print(f"🖥️  计算设备: {args.device}")
    print()

    try:
        # 检查文件是否存在
        if not Path(args.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {args.model_path}")

        if not Path(args.images_dir).exists():
            raise FileNotFoundError(f"图像目录不存在: {args.images_dir}")

        # 检查输入目录中的文件类型
        input_dir = Path(args.images_dir)
        tif_files = list(input_dir.glob("*.tif"))
        png_files = list(input_dir.glob("*.png"))
        jpg_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))

        print(f"📁 输入目录文件统计:")
        print(f"   TIF文件: {len(tif_files)}")
        print(f"   PNG文件: {len(png_files)}")
        print(f"   JPG文件: {len(jpg_files)}")

        # 确定要使用的图像目录
        images_dir = args.images_dir
        temp_dir = None

        if len(tif_files) > 0:
            print(f"\n🔄 检测到 {len(tif_files)} 个TIF文件，正在转换为PNG格式...")

            # 创建临时目录
            temp_dir = tempfile.mkdtemp(prefix="inference_rgb_")
            temp_path = Path(temp_dir)
            print(f"📂 临时目录: {temp_path}")

            # 使用convert_to_rgb.py的逻辑进行转换
            converter = RGBConverter(str(input_dir), str(temp_path))
            converter.batch_convert(output_format='PNG')

            # 检查转换结果
            converted_pngs = list(temp_path.glob("*.png"))
            print(f"✅ 转换完成，共生成 {len(converted_pngs)} 个PNG文件")

            if len(converted_pngs) == 0:
                raise ValueError("TIF转PNG失败，没有生成任何PNG文件")

            # 使用转换后的目录
            images_dir = str(temp_path)
        else:
            print("✅ 输入目录中没有TIF文件，直接使用现有PNG/JPG文件")

        # 创建预测器
        predictor = ModelPredictor(args.model_path, args.device, args.base_channels)

        # 创建数据集（推理时使用全部数据，设置val_ratio=0）
        dataset = HarvestDataset(
            images_dir=images_dir,
            masks_dir=args.masks_dir,
            split='train',
            val_ratio=0.0  # 设置为0以使用全部数据进行推理
        )

        if len(dataset) == 0:
            raise ValueError("数据集为空，请检查图像和掩码路径")

        print(f"📊 数据集大小: {len(dataset)}")

        # 执行批量预测
        success_count = predictor.predict_dataset(dataset, args.output_dir, args.batch_size)

        # 清理临时目录
        if temp_dir is not None:
            try:
                shutil.rmtree(temp_dir)
                print(f"🧹 已清理临时目录: {temp_dir}")
            except Exception as e:
                print(f"⚠️  清理临时目录失败: {e}")

        if success_count > 0:
            print("🎉 批量推理完成！")
            print(f"预测结果保存在: {args.output_dir}")

            # 自动运行评估并输出mIoU
            print("\n🔍 正在自动评估预测结果...")
            miou_value = run_evaluation(args.output_dir, args.masks_dir)

            if miou_value is not None:
                print(".4f")
                print("\n💡 如需详细评估报告，请运行:")
                print(f"python segmentation/evaluate.py --pred-dir {args.output_dir} --true-dir {args.masks_dir}")
            else:
                print("⚠️ 评估失败，请手动运行评估脚本检查结果")
        else:
            print("❌ 没有成功生成任何预测结果")
    except Exception as e:
        print(f"❌ 推理过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
