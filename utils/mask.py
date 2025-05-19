import nibabel as nib

# 加载NIfTI文件
mask_img = nib.load('E:/MRSpingSeg/MRSpineSeg_Challenge_SMU/test1/MR/Case96.nii.gz')

# 获取掩膜数据
mask_data = mask_img.get_fdata()

# 获取掩膜数据的维度数
num_dims = mask_data.ndim

num_channels = mask_data.shape[-1]

# 输出通道数
print("掩膜维度数：",num_dims)
print("掩膜的通道数为:", num_channels)

import nibabel as nib
import numpy as np

# 加载 NIfTI 文件
nifti_file = 'your_nifti_file.nii'
nifti_data = nib.load(nifti_file)
nifti_array = nifti_data.get_fdata()

# 获取切片数
num_slices = nifti_array.shape[-1]

# 将每个切片转换为单通道图像，并堆叠成多通道图像
multi_channel_img = np.zeros((nifti_array.shape[0], nifti_array.shape[1], num_slices))
for i in range(num_slices):
    single_channel_img = nifti_array[..., i]  # 提取第 i 个切片
    multi_channel_img[..., i] = single_channel_img

# multi_channel_img 现在是一个多通道的图像，每个通道代表了原始 NIfTI 文件中的一个切片

