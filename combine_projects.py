import os
import shutil

# 项目路径列表
project_paths = ['data/隐患识别-起重机f1', 'data/隐患识别-起重机f2', 'data/隐患识别-起重机f3', 'data/隐患识别-起重机f4', 'data/隐患识别-起重机f5', 'data/隐患识别-起重机f6']

# 目标合并文件夹路径
merged_images_path = 'dataset/images'
merged_labels_path = 'dataset/labels'
os.makedirs(merged_images_path, exist_ok=True)
os.makedirs(merged_labels_path, exist_ok=True)

# 初始化计数器
counter = 1

# 遍历每个项目
for project_path in project_paths:
    images_path = os.path.join(project_path, 'images')
    labels_path = os.path.join(project_path, 'labels')

    # 遍历每个图像文件
    for image_file in os.listdir(images_path):
        image_file_path = os.path.join(images_path, image_file)
        label_file_path = os.path.join(labels_path, os.path.splitext(image_file)[0] + '.txt')

        # 检查标签文件是否存在以及是否为空
        if os.path.exists(label_file_path):
            with open(label_file_path, 'r') as file:
                label_content = file.read().strip()
                if not label_content:
                    # 如果标签文件为空，跳过该文件
                    continue
        else:
            # 如果标签文件不存在，跳过该文件
            continue

        # 重命名图像文件和标签文件
        new_image_file_name = f"{counter}.jpg"
        new_label_file_name = f"{counter}.txt"

        new_image_file_path = os.path.join(merged_images_path, new_image_file_name)
        new_label_file_path = os.path.join(merged_labels_path, new_label_file_name)

        # 复制文件到目标文件夹并重命名
        shutil.copy(image_file_path, new_image_file_path)
        shutil.copy(label_file_path, new_label_file_path)

        counter += 1

# 合并 classes.txt 和 notes.json
shutil.copy(os.path.join(project_paths[0], 'classes.txt'), 'dataset/classes.txt')
shutil.copy(os.path.join(project_paths[0], 'notes.json'), 'dataset/notes.json')

print("合并完成！")
