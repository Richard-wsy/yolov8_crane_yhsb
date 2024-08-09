import os
import random
import shutil

# 设置路径
dataset_dir = 'dataset'
images_dir = os.path.join(dataset_dir, 'images')
labels_dir = os.path.join(dataset_dir, 'labels')
train_images_dir = os.path.join(images_dir, 'train')
val_images_dir = os.path.join(images_dir, 'val')
train_labels_dir = os.path.join(labels_dir, 'train')
val_labels_dir = os.path.join(labels_dir, 'val')

# 创建train和val文件夹
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# 获取所有图片文件列表
all_images = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]
total_images = len(all_images)

# 计算val文件夹中图片数量
val_count = int(0.1 * total_images)

# 随机选择val文件夹中的图片
val_images = random.sample(all_images, val_count)
train_images = list(set(all_images) - set(val_images))


def move_files(image_files, src_image_dir, dest_image_dir, src_label_dir, dest_label_dir):
    for image_file in image_files:
        # 移动图片文件
        src_image_path = os.path.join(src_image_dir, image_file)
        dest_image_path = os.path.join(dest_image_dir, image_file)
        shutil.move(src_image_path, dest_image_path)

        # 移动对应的标签文件
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_label_path = os.path.join(src_label_dir, label_file)
        dest_label_path = os.path.join(dest_label_dir, label_file)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dest_label_path)


# 移动文件到train文件夹
move_files(train_images, images_dir, train_images_dir, labels_dir, train_labels_dir)

# 移动文件到val文件夹
move_files(val_images, images_dir, val_images_dir, labels_dir, val_labels_dir)

print("文件夹划分完成！")
