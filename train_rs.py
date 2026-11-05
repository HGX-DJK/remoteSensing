"""
遥感影像 YOLO 专用微调与训练模块 (Remote Sensing YOLO Training / Fine-tuning)
支持使用遥感数据集 (如 DOTA, NWPU VHR-10 或自定义标注集) 进行迁移学习与微调。
包含针对遥感小目标与多角度特性的数据增强策略配置 (旋转、尺度变换、Mosaic)。
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

# 确保 UTF-8 输出
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="遥感影像地物识别 YOLO 模型训练与微调")
    parser.add_argument("--data", type=str, default="data/rs_dataset/data.yaml", help="数据集配置文件路径 (yaml)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="基础预训练模型权重 (yolo11n.pt, yolo11s.pt, yolo11n-seg.pt 等)")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (Epochs)")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (像素)")
    parser.add_argument("--batch", type=int, default=4, help="批处理大小 (Batch size)")
    parser.add_argument("--device", type=str, default="cpu", help="计算设备 (cpu 或 0/cuda)")
    parser.add_argument("--project", type=str, default="runs/train_rs", help="训练产出目录")
    parser.add_argument("--name", type=str, default="rs_yolo_model", help="试验名称")
    return parser.parse_args()


def train_remote_sensing_model():
    args = parse_args()
    data_cfg = Path(args.data)
    
    if not data_cfg.exists():
        print(f"[!] 数据集配置不存在: {data_cfg}")
        print("[*] 正在尝试先生成样例数据集...")
        from generate_sample_data import prepare_sample_dataset
        prepare_sample_dataset()

    print(f"\n=======================================================")
    print(f"[>] 开始训练/微调遥感地物识别模型")
    print(f"[*] 基础权重: {args.model}")
    print(f"[*] 数据配置: {args.data}")
    print(f"[*] 训练轮数: {args.epochs} | 图像尺寸: {args.imgsz} | 批大小: {args.batch}")
    print(f"[*] 运行设备: {args.device}")
    print(f"=======================================================\n")

    # 加载预训练模型
    model = YOLO(args.model)

    # 针对遥感图像特点的数据增强超参数:
    # 遥感影像具有俯视旋转不变性 (degrees=180 / fliplr / flipud)
    # 以及小目标拼接增强 (mosaic=1.0)
    results = model.train(
        data=str(data_cfg),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # 遥感特定增强策略
        degrees=180.0,   # 遥感任意角度旋转
        fliplr=0.5,      # 水平翻转
        flipud=0.5,      # 垂直翻转
        scale=0.5,       # 多尺度变换 (适应遥感不同地面分辨率 GSD)
        mosaic=1.0,      # Mosaic 拼接增强，极大提升遥感小目标检测率
        mixup=0.1,       # 混合正则化
        save=True,
        verbose=True
    )

    best_weight = Path(args.project) / args.name / "weights" / "best.pt"
    print("\n-------------------------------------------------------")
    print(f"[OK] 训练完成！最佳模型权重已保存至: {best_weight}")
    print("-------------------------------------------------------")
    return results


if __name__ == "__main__":
    train_remote_sensing_model()
