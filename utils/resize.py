import cv2
import os

def resize_images(input_folder, output_folder, target_size):
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):  # 假设您的数据集是PNG格式
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # 读取图像
            img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

            # 调整大小
            resized_img = cv2.resize(img, target_size)

            # 保存调整大小后的图像
            cv2.imwrite(output_path, resized_img)

# 示例用法
input_folder = 'E:/MRSpingSeg/MRSpineSeg_Challenge_SMU/train/slice/img'
output_folder = 'E:/MRSpingSeg/MRSpineSeg_Challenge_SMU/train/reslice/img'
target_size = (256, 256)  # 设置目标大小

resize_images(input_folder, output_folder, target_size)
