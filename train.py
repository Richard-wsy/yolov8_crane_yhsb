import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


# 获取文件路径并设置根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加 ROOT 到 PATH

def train(opt):
    # 加载预训练模型
    model = YOLO(opt.weights)

    # 训练模型
    model.train(
        data=opt.data,  # 数据集配置文件路径
        epochs=opt.epochs,  # 训练的总周期数
        batch=opt.batch,  # 每个批次的样本数
        imgsz=opt.imgsz,  # 输入图像大小
        device=opt.device,  # 训练设备（如 '0', '0,1,2,3' 或 'cpu'）
        workers=opt.workers,  # 数据加载器的工作线程数
        optimizer=opt.optimizer,  # 优化器选择
        lr0=opt.lr0,  # 初始学习率
        lrf=opt.lrf,  # 最终学习率
        weight_decay=opt.weight_decay,  # 权重衰减
        momentum=opt.momentum,  # 优化器动量
        save_period=opt.save_period,  # 每隔多少周期保存一次模型
        single_cls=opt.single_cls,  # 是否将多类数据作为单类处理
        freeze=opt.freeze,  # 冻结的层
        cos_lr=opt.cos_lr,  # 是否使用余弦退火学习率调度
        label_smoothing=opt.label_smoothing,  # 标签平滑参数
        patience=opt.patience,  # 早停策略的耐心值
    )

    # 评估模型
    model.val()

    print("Training and evaluation complete. Models and results are saved in the 'runs' directory.")

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8n.pt', help='初始权重文件路径')
    parser.add_argument('--data', type=str, default='yolo_dataset.yaml', help='数据集配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练的总周期数')
    parser.add_argument('--batch', type=int, default=64, help='每个批次的样本数')
    parser.add_argument('--imgsz', type=int, default=640, help='训练和验证的图像大小（以像素为单位）')
    parser.add_argument('--device', default='0', help='训练设备，例如 0 或 0,1,2,3 或 cpu')
    parser.add_argument('--workers', type=int, default=12, help='数据加载器的最大工作线程数')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='选择优化器')
    parser.add_argument('--lr0', type=float, default=0.1, help='初始学习率')
    parser.add_argument('--lrf', type=float, default=0.1, help='最终学习率')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='优化器的权重衰减')
    parser.add_argument('--momentum', type=float, default=0.937, help='优化器动量')
    parser.add_argument('--save_period', type=int, default=-1, help='每隔多少周期保存一次模型（如果 < 1 则禁用）')
    parser.add_argument('--single_cls', action='store_true', help='将多类数据作为单类处理')
    parser.add_argument('--freeze', type=int, nargs='+', default=[0], help='冻结层：backbone=10, first3=0 1 2')
    parser.add_argument('--cos_lr', action='store_true', help='使用余弦退火学习率调度')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑参数')
    parser.add_argument('--patience', type=int, default=150, help='早停策略的耐心值（无改进的周期数）')
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    train(opt)
