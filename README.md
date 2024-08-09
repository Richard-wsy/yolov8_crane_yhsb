# yolov-data-train

## 功能说明
yolov-data-train 项目旨在使用YOLOv8进行模型训练，以实现高效的对象检测。该项目包括以下主要功能：

* 数据预处理：label-studio平台数据标注项目下载后的数据整合。
* 模型训练和评估：使用YOLOv8架构进行模型训练。

## 启用步骤
* 环境准备
   * 在终端中创建虚拟环境：
     ```bash
     conda create -n yolov-data-train python=3.9 -y
     conda activate yolov-data-train
     ```
   * 如果在Conda环境中无法使用git命令，可以按照以下命令安装git：
     ```bash
     conda install git
     ```
   * 克隆该项目：
     ```bash
     git clone https://github.com/Richard-wsy/yolov8_crane_yhsb.git
     ```
   * 查看自己设备cuda版本：
     ```bash
     nvidia-smi
     nvcc -V
     ```
   * 到pytorch官网根据自己设备型号选择torch版本，例如：在终端中运行`nvidia-smi`,得到`CUDA Version: 12.5`,可以在pytorch官网中依次选择：`Your OS：你的操作系统如果是Windows选择Windows，如果是Linux选择Linux`、`Package：选择Pip`、`Language：编程语言选择Python`、`Compute Platform：cuda选择比刚刚在终端运行的cuda小的版本(例如：终端运行nvidia-smi得到12.5时就可以在pytorch官网中选择cuda 12.1、如果得到11.8，在官网中只能选择11.8，不能选择12.1)`最后复制`Run this Command栏的命令`如：
     ```
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     ```
     官网链接：[Pytorch](https://pytorch.org/)
   * 安装依赖包：
     ```bash
     cd yolov-data-train #进入yolov-data-train文件目录
     pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 #将刚刚在pytorch官网复制好的命令在终端中运行
     pip install -r requirements.txt
     ```
   * 数据准备：
     * 当前文件夹下新建名称为`data`文件夹
       ```bash
       mkdir data
       ```
     * 在公司建立的label-studio标注平台库，导出标好的数据文件，导出格式为yolo格式，将导出的文件解压到刚刚新建的`data`文件中，label-studio的网址如下：
       [label-studio]()
     * 将导入到`data`文件中的不同项目标签数据文件进行整合，`combine_projects.py`文件中可以修改`project_paths = ['data/projects1', 'data/projects2', 'data/projects3']`在终端中运行：
       ```bash
       python combine_projects.py
       ```
     * 运行脚本后当前目录下会出现`dataset`文件，将`dataset`文件夹下`images`文件和`labels`文件里的数据进行分割制作训练集`train`和验证集`val`，在终端中运行如下命令：
       ```bash
       python profils_seg.py
       ```
     * 制作数据集配置文件，得到`.yaml`文件，终端运行如下命令：
       ```bash
       python make_yaml.py
       ```
   * 模型准备：
     在[yolov官网](https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes)中下载预训练模型文件到当前文件夹下，各个预训练模型文件下载地址如下：
     * [yolov8n.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)
     * [yolov8s.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt)
     * [yolov8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt)
     * [yolov8l.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt)
     * [yolov8x.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt)
  * 训练数据：
    * 训练参数配置，在`train.py`脚本中可以修改如下参数：
      ```
      parser.add_argument('--weights', type=str, default='yolov8n.pt', help='初始权重文件路径')
      parser.add_argument('--data', type=str, default='yolo_dataset.yaml', help='数据集配置文件路径')
      parser.add_argument('--epochs', type=int, default=300, help='训练的总周期数')
      parser.add_argument('--batch', type=int, default=64, help='每个批次的样本数')
      parser.add_argument('--imgsz', type=int, default=640, help='训练和验证的图像大小（以像素为单位）')
      parser.add_argument('--device', default='0', help='训练设备，例如 0 或 0,1,2,3 或 cpu')
      parser.add_argument('--workers', type=int, default=12, help='数据加载器的最大工作线程数')
      parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='AdamW', help='选择优化器')
      parser.add_argument('--lr0', type=float, default=0.1, help='初始学习率')
      parser.add_argument('--lrf', type=float, default=0.001, help='最终学习率')
      parser.add_argument('--weight_decay', type=float, default=0.0001, help='优化器的权重衰减')
      parser.add_argument('--momentum', type=float, default=0.937, help='优化器动量')
      parser.add_argument('--save_period', type=int, default=-1, help='每隔多少周期保存一次模型（如果 < 1 则禁用）')
      parser.add_argument('--single_cls', action='store_true', help='将多类数据作为单类处理')
      parser.add_argument('--freeze', type=int, nargs='+', default=[0], help='冻结层：backbone=10, first3=0 1 2')
      parser.add_argument('--cos_lr', action='store_true', help='使用余弦退火学习率调度')
      parser.add_argument('--label_smoothing', type=float, default=0.0, help='标签平滑参数')
      parser.add_argument('--patience', type=int, default=300, help='早停策略的耐心值（无改进的周期数）')
      ```
    * 开始训练，在终端中运行如下命令：
      ```bash
      python train.py
      ```
    * 训练结果,训练完成后目录下会出现`runs`文件，里面有训练过程记录的数据和训练好的模型权重`weights`文件，文件保存了`best.pt`和`last.pt`两个模型。
## 电脑配置问题：
* 建议在内存为32g及以上的配置运行，如果运行后一段时间卡住，尝试将`batch`参数调小。
* 可以尝试调整修改不同参数，达到好的训练效果。
