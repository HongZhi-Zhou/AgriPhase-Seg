#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量转换遥感图像为RGB图像脚本
将数据集中的TIF格式遥感图像转换为RGB图像并保存
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import rasterio
from tqdm import tqdm
import argparse


class RGBConverter:
    def __init__(self, input_dir, output_dir):
        """
        初始化RGB转换器

        Args:
            input_dir: 输入图像文件夹路径
            output_dir: 输出RGB图像文件夹路径
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 查找所有TIF文件（只处理.tif结尾的文件）
        self.tif_files = list(self.input_dir.glob("*.tif"))

        if len(self.tif_files) == 0:
            raise ValueError(f"在 {input_dir} 中没有找到TIF文件")

        print(f"找到 {len(self.tif_files)} 个TIF文件待转换")
        print(f"输出目录: {self.output_dir}")

    def convert_single_image(self, tif_path, quiet=False):
        """
        转换单个TIF图像为RGB（使用调试验证的逻辑）

        Args:
            tif_path: TIF文件路径
            quiet: 是否静默模式（不显示详细信息）

        Returns:
            PIL.Image: RGB图像，失败时返回None
        """
        try:
            with rasterio.open(tif_path) as src:
                count = src.count
                if not quiet:
                    print(f"  处理 {tif_path.name}: {count} 个波段")

                # 检查是否有足够的波段
                required_bands = [4, 3, 2]  # NIR, Red, Green
                available_bands = []

                for band in required_bands:
                    if band <= count:
                        available_bands.append(band)
                    else:
                        # 如果没有足够的波段，使用最后一个可用波段
                        available_bands.append(count)
                        if not quiet:
                            print(f"    警告: 请求的波段{band}不存在，使用波段{count}代替")

                if not quiet:
                    print(f"    使用波段 {available_bands} 作为RGB (NIR->R, Red->G, Green->B)")

                # 读取波段数据
                bands_data = []

                for i, band_num in enumerate(available_bands):
                    band_data = src.read(band_num).astype(np.float32)
                    bands_data.append(band_data)
                    if not quiet:
                        print(f"    波段{band_num}: 范围 [{band_data.min():.2f}, {band_data.max():.2f}], 均值: {band_data.mean():.2f}")

                # 堆叠为RGB数组 (NIR, Red, Green) -> (R, G, B)
                rgb_array = np.stack(bands_data, axis=0)  # (3, H, W)

                if not quiet:
                    print(f"    RGB数组堆叠后 - 形状: {rgb_array.shape}, 范围: [{rgb_array.min()}, {rgb_array.max()}], 均值: {rgb_array.mean():.2f}")

                # 处理数据范围 - 使用更鲁棒的方式
                original_max = rgb_array.max()
                original_min = rgb_array.min()

                if not quiet:
                    print(f"    原始数据范围: [{original_min}, {original_max}]")

                # 根据数据范围确定缩放策略
                if rgb_array.max() > 255:
                    if rgb_array.max() > 65535:
                        # 处理异常大数值 - 使用最大值归一化
                        if not quiet:
                            print(f"    检测到异常大数值 (>65535)，使用最大值归一化")
                        rgb_array = rgb_array / (rgb_array.max() / 255.0)
                    elif rgb_array.max() > 1000:
                        # 标准16-bit数据
                        if not quiet:
                            print(f"    检测到16-bit数据，使用/256缩放")
                        rgb_array = rgb_array / 256.0
                    else:
                        # 可能是已经预处理过的数据
                        if not quiet:
                            print(f"    数据范围适中 ({rgb_array.max()})，使用最大值归一化")
                        rgb_array = rgb_array / (rgb_array.max() / 255.0)

                    if not quiet:
                        print(f"    缩放后范围: [{rgb_array.min():.2f}, {rgb_array.max():.2f}]")
                else:
                    if not quiet:
                        print(f"    数据已在8-bit范围内，无需缩放 (最大值: {original_max})")

                # 检查并修复异常值
                nan_count = np.isnan(rgb_array).sum()
                inf_count = np.isinf(rgb_array).sum()

                if nan_count > 0:
                    if not quiet:
                        print(f"    ⚠️  检测到 {nan_count} 个NaN值，正在修复为0")
                    rgb_array = np.nan_to_num(rgb_array, nan=0.0)

                if inf_count > 0:
                    if not quiet:
                        print(f"    ⚠️  检测到 {inf_count} 个无穷大值，正在修复")
                    rgb_array = np.nan_to_num(rgb_array, posinf=255.0, neginf=0.0)

                # 再次检查数据范围（修复后）
                if not quiet:
                    print(f"    异常值修复后范围: [{rgb_array.min():.2f}, {rgb_array.max():.2f}]")

                # 确保在0-255范围内
                rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

                # 最终统计
                non_zero_ratio = (rgb_array > 0).mean()
                if not quiet:
                    print(f"    最终结果 - 范围: [{rgb_array.min()}, {rgb_array.max()}], 非零像素: {non_zero_ratio:.2%}")

                # 处理全黑或几乎全黑的图像
                if rgb_array.max() == 0:
                    print(f"    ❌ 检测到全黑图像！尝试使用百分位数拉伸修复...")

                    # 重新读取原始数据进行百分位数拉伸
                    with rasterio.open(tif_path) as src:
                        bands_data_fixed = []
                        for band_num in available_bands:
                            band_data = src.read(band_num).astype(np.float32)

                            # 计算有效像素的百分位数
                            valid_pixels = band_data[band_data > 0]
                            if len(valid_pixels) > 0:
                                p2, p98 = np.percentile(valid_pixels, [2, 98])
                                if p98 > p2:  # 确保有足够的动态范围
                                    stretched = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                                    stretched = (stretched * 255).astype(np.uint8)
                                    print(f"      波段{band_num} 百分位数拉伸: [{p2:.2f}, {p98:.2f}] -> [0, 255]")
                                else:
                                    # 如果动态范围不足，使用最大值归一化
                                    if band_data.max() > 0:
                                        stretched = (band_data / band_data.max() * 255).astype(np.uint8)
                                        print(f"      波段{band_num} 最大值归一化: [0, {band_data.max():.2f}] -> [0, 255]")
                                    else:
                                        stretched = np.zeros_like(band_data, dtype=np.uint8)
                                        print(f"      波段{band_num} 全为0值，保持为0")
                            else:
                                # 如果没有有效像素
                                stretched = np.zeros_like(band_data, dtype=np.uint8)
                                print(f"      波段{band_num} 无有效像素，设为0")

                            bands_data_fixed.append(stretched)

                        rgb_array = np.stack(bands_data_fixed, axis=0)

                    if not quiet:
                        print(f"    百分位数拉伸后范围: [{rgb_array.min()}, {rgb_array.max()}]")

                elif non_zero_ratio < 0.01:  # 少于1%的像素非零
                    print(f"    ⚠️  警告: 图像几乎全黑 (仅 {non_zero_ratio:.2%} 像素非零)")
                    print(f"        这可能表明数据处理有问题或原始数据就是全黑的")
                else:
                    if not quiet:
                        print("    ✅ 图像处理成功")

                # 转换为PIL图像格式 (H, W, C)
                rgb_img = np.transpose(rgb_array, (1, 2, 0))
                return Image.fromarray(rgb_img)

        except Exception as e:
            print(f"❌ 转换失败 {tif_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def batch_convert(self, output_format='PNG'):
        """
        批量转换所有图像

        Args:
            output_format: 输出格式 ('PNG' 或 'JPEG')
        """
        success_count = 0
        fail_count = 0

        print(f"\n开始批量转换 {len(self.tif_files)} 个文件...")
        print(f"输出格式: {output_format}")

        for tif_path in tqdm(self.tif_files, desc="转换进度"):
            # 生成输出文件名
            stem = tif_path.stem  # 不含扩展名的文件名
            output_filename = f"{stem}.{output_format.lower()}"
            output_path = self.output_dir / output_filename

            # 转换图像（启用详细调试信息）
            rgb_img = self.convert_single_image(tif_path, quiet=False)

            if rgb_img is not None:
                # 检查是否为全黑图像
                rgb_array = np.array(rgb_img)
                if rgb_array.max() == 0:
                    print(f"\n❌ 检测到全黑图像: {tif_path.name}")
                    print("重新运行详细调试模式...")
                    # 重新运行并显示详细信息
                    rgb_img = self.convert_single_image(tif_path, quiet=False)
                    # 既然是全黑的，算作失败
                    fail_count += 1
                    continue

                try:
                    # 保存图像
                    if output_format.upper() == 'PNG':
                        rgb_img.save(output_path, 'PNG')
                    elif output_format.upper() == 'JPEG':
                        # JPEG不支持透明度，转换为RGB模式
                        if rgb_img.mode != 'RGB':
                            rgb_img = rgb_img.convert('RGB')
                        rgb_img.save(output_path, 'JPEG', quality=95)
                    else:
                        raise ValueError(f"不支持的输出格式: {output_format}")

                    success_count += 1

                except Exception as e:
                    print(f"❌ 保存失败 {output_path}: {e}")
                    fail_count += 1
            else:
                fail_count += 1

        # 输出统计结果
        print("" + "="*60)
        print("📊 批量转换完成统计")
        print("="*60)
        print(f"总文件数: {len(self.tif_files)}")
        print(f"✅ 转换成功: {success_count}")
        print(f"❌ 转换失败: {fail_count}")
        print(".1f")
        print(f"💾 输出目录: {self.output_dir}")

        if success_count > 0:
            print("🎉 转换完成！RGB图像已保存到指定目录")
        else:
            print("❌ 没有成功转换任何文件，请检查输入文件和错误信息")
    def show_sample_info(self):
        """
        显示样本图像的信息
        """
        if len(self.tif_files) == 0:
            return

        sample_file = self.tif_files[0]
        print(f"\n🔍 样本文件信息 ({sample_file.name}):")

        try:
            with rasterio.open(sample_file) as src:
                print(f"   图像尺寸: {src.width} x {src.height}")
                print(f"   波段数量: {src.count}")
                print(f"   数据类型: {src.dtypes[0]}")
                print(f"   坐标系统: {src.crs}")

                # 显示各波段的基本统计
                print("   各波段统计:")
                for i in range(1, min(5, src.count + 1)):  # 显示前5个波段
                    band_data = src.read(i)
                    print(f"     波段{i}: 范围[{band_data.min()}, {band_data.max()}], 均值:{band_data.mean():.1f}")

        except Exception as e:
            print(f"   ❌ 读取样本文件失败: {e}")


def test_single_image():
    """测试单个图像的转换过程"""
    # 测试第一个图像
    input_dir = "D:/PythonProject/dataset/harvest_result/images"
    test_image = "D:/PythonProject/dataset/harvest_result/tifs/tile_00041.tif"

    print("🧪 测试单个图像转换")
    print("="*50)
    print(f"测试图像: {test_image}")

    try:
        converter = RGBConverter(input_dir, input_dir)  # 使用相同目录作为输出以便测试

        # 测试转换（启用详细输出）
        rgb_img = converter.convert_single_image(Path(test_image), quiet=False)

        if rgb_img is not None:
            print("✅ 测试成功！")
            return True
        else:
            print("❌ 测试失败！")
            return False

    except Exception as e:
        print(f"❌ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    # 直接设置参数（无需命令行参数）
    input_dir = "D:/PythonProject/dataset/harvest_result/images"  # 输入图像目录
    output_dir = "D:/PythonProject/segmentation/save_img"  # 输出RGB图像目录
    output_format = "PNG"  # 输出格式: 'PNG' 或 'JPEG'

    print("🌈 遥感图像RGB转换工具")
    print("="*50)
    print(f"📁 输入目录: {input_dir}")
    print(f"💾 输出目录: {output_dir}")
    print(f"🎨 输出格式: {output_format}")

    try:
        # 首先测试单个图像
        print("\n🔍 首先测试单个图像...")
        test_success = test_single_image()

        if not test_success:
            print("❌ 单个图像测试失败，终止批量转换")
            print("请检查图像文件和数据处理逻辑")
            return 1

        print("\n✅ 单个图像测试成功，开始批量转换...")

        # 创建转换器
        converter = RGBConverter(input_dir, output_dir)

        # 显示样本信息
        converter.show_sample_info()

        # 执行批量转换（输出格式设置为PNG以获得最佳质量）
        converter.batch_convert(output_format)

    except Exception as e:
        print(f"❌ 转换过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
