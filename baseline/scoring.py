#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本
计算预测掩码与真实标签的mIoU等指标
支持多任务评估：收割、整地、清雪扣棚
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 使用更兼容的字体
plt.rcParams['axes.unicode_minus'] = False


def read_image(path):
    """统一读取 TIFF/PNG/JPG 等格式为单通道灰度图 (H, W)，像素值保持原样"""
    try:
        img = Image.open(path)
        # 转为 NumPy 数组，保留原始数据类型（如 uint8, uint16）
        img_array = np.array(img)
        # 如果是彩色图，转为灰度（但应尽量避免；仅当必要时）
        if img_array.ndim == 3:
            if img_array.shape[2] == 1:
                img_array = img_array.squeeze(-1)
            else:
                # 使用 PIL 自带的灰度转换（更安全）
                img = img.convert('L')
                img_array = np.array(img)
        return img_array
    except Exception as e:
        raise ValueError(f"无法读取图像: {path}（可能是损坏或不支持的格式）") from e


def calculate_metrics_single_task(pred_dir, gt_dir, class_labels=[1, 2], pred_ext='.png', gt_ext=('.tif', '.tiff')):
    """
    计算单个任务的预测结果与真实标签的指标（修正版）
    修复了真实标签与预测标签混淆的核心错误
    """
    num_classes = len(class_labels)
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    # 获取预测文件
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.lower().endswith(pred_ext)])
    # 获取标签文件
    gt_files = sorted([f for f in os.listdir(gt_dir) if any(f.lower().endswith(ext) for ext in gt_ext)])

    if len(pred_files) != len(gt_files):
        raise ValueError(f"预测文件数({len(pred_files)})与真实文件数({len(gt_files)})不匹配！")

    label2idx = {label: idx for idx, label in enumerate(class_labels)}

    for pred_file, gt_file in zip(pred_files, gt_files):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        pred_img = read_image(pred_path)
        gt_img = read_image(gt_path)

        # 确保是二维数组（H, W）
        if pred_img.ndim == 3:
            pred_img = np.mean(pred_img, axis=2).astype(np.uint8)
        if gt_img.ndim == 3:
            gt_img = np.mean(gt_img, axis=2).astype(np.uint8)

        if pred_img.shape != gt_img.shape:
            raise ValueError(f"图像尺寸不匹配：{pred_file}({pred_img.shape}) vs {gt_file}({gt_img.shape})")

        # ✅ 修正1：正确处理预测标签（基于pred_img）
        if pred_img.dtype == np.uint8:
            # 0 -> 类别1 (1), 255 -> 类别2 (2)
            pred_labels = np.where(pred_img == 0, 1, np.where(pred_img == 255, 2, 0)).astype(np.uint8)
        else:
            # 概率图：>0.5 -> 类别2 (2), <=0.5 -> 类别1 (1)
            pred_labels = np.where(pred_img > 0.5, 2, 1).astype(np.uint8)

        # ✅ 修正2：正确处理真实标签（基于gt_img，仅保留class_labels中的类别）
        gt_labels = gt_img.copy()

        # ✅ 修正3：创建有效掩码（只保留真实标签和预测标签都在class_labels中的像素）
        valid_mask = np.isin(gt_labels, class_labels) & np.isin(pred_labels, class_labels)
        if not np.any(valid_mask):
            continue  # 跳过无有效像素的图像

        # 提取有效像素
        gt_valid = gt_labels[valid_mask]
        pred_valid = pred_labels[valid_mask]

        # 映射为索引 (1->0, 2->1)
        gt_idx = np.vectorize(label2idx.get)(gt_valid)
        pred_idx = np.vectorize(label2idx.get)(pred_valid)

        # ✅ 修正4：构建混淆矩阵（行=真实标签, 列=预测标签）
        for g, p in zip(gt_idx, pred_idx):
            conf_matrix[g, p] += 1

    # 计算指标
    class_iou = {}
    for idx, label in enumerate(class_labels):
        intersection = conf_matrix[idx, idx]
        union = conf_matrix[idx, :].sum() + conf_matrix[:, idx].sum() - intersection
        iou = intersection / union if union > 0 else 0.0
        if union > 0:  # 仅记录实际出现过的类别
            class_iou[label] = iou

    # 计算mIoU：对实际出现过的类别取平均
    miou = np.mean(list(class_iou.values())) if class_iou else 0.0
    oa = np.trace(conf_matrix) / conf_matrix.sum() if conf_matrix.sum() > 0 else 0.0

    return {
        "class_iou": class_iou,
        "miou": miou,
        "oa": oa,
        "conf_matrix": conf_matrix
    }
class MultiTaskEvaluator:
    def __init__(self, result_base_dir, gt_base_dir):
        """
        初始化多任务评估器

        Args:
            result_base_dir: 结果根目录，包含各任务的子目录（Result/收割, Result/整地, Result/清雪扣棚）
            gt_base_dir: 真实标签根目录，包含各任务的labels子目录
        """
        self.result_base_dir = Path(result_base_dir)
        self.gt_base_dir = Path(gt_base_dir)
        self.tasks = ['task1', 'task2', 'task3']
        self.task_results = {}

        # 检查任务目录是否存在
        for task in self.tasks:
            pred_dir = self.result_base_dir / task
            gt_dir = self.gt_base_dir / task

            pred_exists = pred_dir.exists()
            gt_exists = gt_dir.exists()

            if not pred_exists:
                print(f"[WARNING] 预测目录不存在: {pred_dir}")
            if not gt_exists:
                print(f"[WARNING] 标签目录不存在: {gt_dir}")

            if pred_exists and gt_exists:
                # 检查文件格式：预测结果为PNG，标签为TIF
                pred_count = len(list(pred_dir.glob('*.png')))
                gt_count = len(list(gt_dir.glob('*.tif'))) + len(list(gt_dir.glob('*.tiff')))

                print(f"[OK] 找到任务目录: {task} ({pred_count} 个预测文件, {gt_count} 个标签文件)")

    def evaluate_task(self, task_name):
        """
        评估单个任务

        Args:
            task_name: 任务名称

        Returns:
            dict: 任务评估结果
        """
        pred_dir = self.result_base_dir / task_name
        gt_dir = self.gt_base_dir / task_name

        pred_exists = pred_dir.exists()
        gt_exists = gt_dir.exists()

        if not pred_exists or not gt_exists:
            error_msg = f"目录不存在: "
            if not pred_exists:
                error_msg += f"预测目录 {pred_dir}"
            if not pred_exists and not gt_exists:
                error_msg += ", "
            if not gt_exists:
                error_msg += f"标签目录 {gt_dir}"
            return {"error": error_msg}

        # 检查文件格式：预测结果为PNG，标签为TIF
        pred_files = sorted([f for f in os.listdir(str(pred_dir)) if f.lower().endswith('.png')])
        gt_files = sorted([f for f in os.listdir(str(gt_dir)) if f.lower().endswith(('.tif', '.tiff'))])

        if len(pred_files) == 0:
            return {"error": f"任务 {task_name} 中没有找到预测文件（.png格式）"}

        if len(gt_files) == 0:
            return {"error": f"任务 {task_name} 中没有找到标签文件（.tif/.tiff格式）"}

        print(f"\n🔍 评估任务: {task_name}")
        print(f"   预测文件数量: {len(pred_files)}")
        print(f"   标签文件数量: {len(gt_files)}")

        # 使用单任务评估函数（基于scoring.py格式）
        try:
            metrics = calculate_metrics_single_task(str(pred_dir), str(gt_dir), class_labels=[1, 2], pred_ext='.png', gt_ext=('.tif', '.tiff'))
            return metrics
        except Exception as e:
            return {"error": f"评估失败: {e}"}

    def evaluate_all_tasks(self):
        """
        评估所有任务

        Returns:
            dict: 所有任务的评估结果
        """

        all_results = {}

        for task in self.tasks:
            result = self.evaluate_task(task)
            all_results[task] = result

            if "error" in result:
                print(f"❌ 任务 {task} 评估失败: {result['error']}")
            else:
                print("成功")
        return all_results

    def print_summary(self, results):
        """
        打印评估总结

        Args:
            results: 评估结果字典
        """


        # 各任务结果
        print("\n🎯 各任务性能指标:")
        for task, result in results.items():
            print(f"\n{task}:")
            if "error" in result:
                print(f"  ❌ 错误: {result['error']}")
            else:
                print("成功")
        # 计算平均性能
        valid_tasks = [result for result in results.values() if "error" not in result]
        if valid_tasks:
            avg_miou = np.mean([r['miou'] for r in valid_tasks])
            avg_oa = np.mean([r['oa'] for r in valid_tasks])



        print("="*80)

    def save_results(self, results, output_file='multi_task_evaluation.txt'):
        """
        保存评估结果到文件

        Args:
            results: 评估结果字典
            output_file: 输出文件名
        """
        with open(output_file, 'w') as f:
            for task, result in results.items():
                if "error" not in result:
                    # 任务名称映射
                    task_key_map = {
                        '收割': 'harvest',
                        '整地': 'plowing',
                        '清雪扣棚': 'clearsnow'
                    }
                    task_key = task_key_map.get(task, task.lower())

                    # 只写入mIoU
                    f.write(f"{task_key}_miou: {result['miou']:.6f}\n")
            valid_tasks = [result for result in results.values() if "error" not in result]
            if valid_tasks:
                avg_miou = np.mean([r['miou'] for r in valid_tasks])
                f.write(f"Avg_mIoU: {avg_miou:.6f}\n")
    def save_coda_format(self, results, output_file='/app/output/scores.txt'):
        """
        保存CodaBench格式的结果（只输出三个任务的mIoU）

        Args:
            results: 评估结果字典
            output_file: 输出文件名
        """
        with open(output_file, 'w') as f:
            for task, result in results.items():
                if "error" not in result:
                    # 任务名称映射
                    task_key_map = {
                        '收割': 'harvest',
                        '整地': 'plowing',
                        '清雪扣棚': 'clearsnow'
                    }
                    task_key = task_key_map.get(task, task.lower())

                    # 只写入mIoU
                    f.write(f"{task_key}_miou: {result['miou']:.6f}\n")
            valid_tasks = [result for result in results.values() if "error" not in result]
            if valid_tasks:
                avg_miou = np.mean([r['miou'] for r in valid_tasks])
                f.write(f"Avg_mIoU: {avg_miou:.6f}\n")
        print(f"🏆 CodaBench格式结果已保存到: {output_file}")



def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估预测掩码与真实标签的性能（多任务）')
    parser.add_argument('--pred-dir', type=str, default='D:/AIRS/prediction',
                       help='结果根目录，包含各任务的子目录（Result/收割, Result/整地, Result/清雪扣棚）')
    parser.add_argument('--gt-dir', type=str, default='D:/AIRS/test_label',
                       help='真实标签根目录，包含各任务的子目录')
    parser.add_argument('--output-file', type=str, default='D:\PythonProject\segmentation\scores.txt',
                       help='详细结果输出文件')
    parser.add_argument('--scores-file', type=str, default='D:\PythonProject\segmentation\scores.txt',
                       help='CodaBench格式分数文件')

    args = parser.parse_args()

    print("📊 多任务模型评估工具")
    print("="*50)
    print(f"🔮 结果目录: {args.pred_dir}")
    print(f"🎯 标签目录: {args.gt_dir}")
    print(f"📄 详细结果: {args.output_file}")
    print(f"🏆 分数文件: {args.scores_file}")
    print()

    try:
        # 检查目录
        result_path = Path(args.pred_dir)
        gt_path = Path(args.gt_dir)

        if not result_path.exists():
            raise FileNotFoundError(f"结果目录不存在: {args.pred_dir}")
        if not gt_path.exists():
            raise FileNotFoundError(f"标签目录不存在: {args.gt_dir}")

        # 创建多任务评估器
        evaluator = MultiTaskEvaluator(args.pred_dir, args.gt_dir)

        # 执行所有任务的评估
        results = evaluator.evaluate_all_tasks()

        # 打印总结
        evaluator.print_summary(results)

        # 保存详细结果
        #evaluator.save_results(results, args.output_file)

        # 保存CodaBench格式的结果
        evaluator.save_coda_format(results, args.scores_file)

        print("✅ 多任务评估完成！")

    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
