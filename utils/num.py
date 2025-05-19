import os
from collections import defaultdict

# 获取当前目录下所有文件
directory = "Pytorch-UNet-master/data/imgs"
files = os.listdir(directory)
# 分组文件
file_groups = defaultdict(list)
for file in files:
    prefix = file.split('-')[0]  # 获取前缀
    file_groups[prefix].append(file)

# 找到每个组中最小的数量
min_group_size = min(len(group) for group in file_groups.values())

# 对每个分组进行处理

for prefix, group in file_groups.items():
    if len(group) > min_group_size:
        print(group)
        # 删除超过最小数量的第一张和最后一张
        os.remove(os.path.join(directory, group[-1]))  # 删除最后一张
#         os.remove(os.path.join(directory, group[0]))  # 删除第一张

# print(file_groups)
# for prefix, group in file_groups.items():
#     for i, file in enumerate(group):
#         new_name = f"{prefix}-slice{i:02d}.png"
#         os.rename(os.path.join(directory, file), os.path.join(directory, new_name))

