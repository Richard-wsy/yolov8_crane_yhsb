import torch

# 查看是否有可用的CUDA设备
print(torch.cuda.is_available())

# 查看可用的CUDA设备数量
print(torch.cuda.device_count())

# 查看当前CUDA设备的名称
print(torch.cuda.get_device_name(0))
