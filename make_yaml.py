import yaml

# 分类说明
categories = [
    {"id": 0, "name": "Bridge Girder Erection Machine"},
    {"id": 1, "name": "Crawler Crane"},
    {"id": 2, "name": "Gantry Crane"},
    {"id": 3, "name": "Guardrail"},
    {"id": 4, "name": "Tower Crane"},
    {"id": 5, "name": "Truck Crane"},
    {"id": 6, "name": "car"},
    {"id": 7, "name": "person"},
    {"id": 8, "name": "truck"}
]

info = {
    "year": 2024,
    "version": "1.0",
    "contributor": "Label Studio"
}

# 创建YOLO数据集的yaml文件内容
yolo_dataset = {
    'train': 'F:/yolov8_crane_yhsb/dataset/images/train',  # 训练集路径
    'val': 'F:/yolov8_crane_yhsb/dataset/images/val',  # 验证集路径
    'nc': len(categories),  # 类别数量
    'names': [category['name'] for category in categories]  # 类别名称列表
}

# 保存为yaml文件
with open('yolo_dataset.yaml', 'w') as file:
    yaml.dump(yolo_dataset, file, default_flow_style=False)

print("yolo_dataset.yaml 文件已创建。")
