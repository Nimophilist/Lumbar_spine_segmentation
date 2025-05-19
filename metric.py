import torch
import pickle
import numpy as np
import argparse
import logging
import torch.nn.functional as F

from unet import UNet
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def calculate_dsc(mask_pred, mask_true):
    intersection = torch.sum((mask_pred == mask_true) & (mask_pred != 0), dim=(1, 2))
    dice = (2. * intersection.float() + 1e-8) / (torch.sum(mask_pred != 0, dim=(1, 2)).float() + torch.sum(mask_true != 0, dim=(1, 2)).float() + 1e-8)
    return dice


# 计算Intersection over Union (IOU)
def calculate_iou(mask_pred, mask_true):
    intersection = torch.sum((mask_pred == mask_true) & (mask_pred != 0), dim=(1, 2))
    union = torch.sum((mask_pred != 0) | (mask_true != 0), dim=(1, 2))
    iou = intersection.float() / (union.float() + 1e-8)  # Add a small epsilon to avoid division by zero
    return iou


def calculate_accuracy(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.view(-1)
    mask_true_flat = mask_true.view(-1)

    # Calculate accuracy
    accuracy = accuracy_score(mask_true_flat.cpu(), mask_pred_flat.cpu())

    return accuracy

def calculate_recall(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.view(-1)
    mask_true_flat = mask_true.view(-1)

    # Calculate recall
    recall = recall_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

    return recall

def calculate_f1_score(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.view(-1)
    mask_true_flat = mask_true.view(-1)

    # Calculate F1 score
    f1 = f1_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

    return f1

def calculate_precision(mask_pred, mask_true):
    # Flatten the masks
    mask_pred_flat = mask_pred.view(-1)
    mask_true_flat = mask_true.view(-1)

    # Calculate precision
    precision = precision_score(mask_true_flat.cpu(), mask_pred_flat.cpu(), average='macro', zero_division=1)

    return precision

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    net.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    logging.info('Model loaded!')

    with open('/root/data_loaders.pkl', 'rb') as f:
        data = pickle.load(f)

    # 获取验证集数据
    test_data = data['test']
    # dices = []
    # ious = []
    # precisions = []
    # recalls = []
    # accuracys = []
    # f1s = []
    # net.eval()
    # for i, batch in enumerate(test_data):
    #     test_images = batch['image'].to(device)
    #     true_test = batch['mask'].to(device)
    #     with torch.no_grad():
    #         output = net(test_images)
    #         pred_test = output.argmax(dim=1)
    #         dice = calculate_dsc(pred_test,true_test)
    #         dices.append(dice)
    #         iou = calculate_iou(pred_test, true_test)
    #         ious.append(iou)
    #         precision = calculate_precision(pred_test,true_test)
    #         precisions.append(precision)
    #         recall = calculate_recall(pred_test, true_test)
    #         recalls.append(recall)
    #         accuracy = calculate_accuracy(pred_test,true_test)
    #         accuracys.append(accuracy)
    #         f1 = calculate_f1_score(pred_test,true_test)
    #         f1s.append(f1)
    # dice = np.average([d.cpu().numpy() for d in dices])
    # iou = np.average([iou.cpu().numpy() for iou in ious])
    # precision = np.average([precision.cpu().numpy() for precision in precisions])
    # recall = np.average([recall.cpu().numpy() for recall in recalls])
    # accuracy = np.average([accuracy.cpu().numpy() for accuracy in accuracys])
    # f1 = np.average([f1.cpu().numpy() for f1 in f1s])
    
    # print("Dice Score:", dice)
    # print("Iou:", iou)
    # print("Accuracy:", accuracy)
    # print("Recall:", recall)
    # print("F1 Score:", f1)
    # print("Precision:", precision)
    # 在循环外初始化指标总和
    dices = []
    ious = []
    precisions = []
    recalls = []
    accuracys = []
    f1s = []
    # total_dice = 0.0
    # total_iou = 0.0
    # total_precision_sum = 0
    # total_recall_sum = 0
    # total_accuracy_sum = 0
    # total_f1_sum = 0
    net.eval()
    for i, batch in enumerate(test_data):
        test_images = batch['image'].to(device)
        true_tests = batch['mask'].to(device)
        for j, (test_image, true_test) in enumerate(zip(test_images, true_tests)):
            test_image = test_image.unsqueeze(0)
            true_test = true_test.unsqueeze(0)
            with torch.no_grad():
                output = net(test_image)
                pred_test = output.argmax(dim=1)
                # 计算当前批次的指标
                dice = calculate_dsc(pred_test, true_test)
                dices.append(dice)
                iou = calculate_iou(pred_test, true_test)
                ious.append(iou)
                precision = calculate_precision(pred_test, true_test)  # 添加一维以匹配真实标签的形状
                precisions.append(precision)
                recall = calculate_recall(pred_test, true_test)  # 添加一维以匹配真实标签的形状
                recalls.append(recall)
                accuracy = calculate_accuracy(pred_test, true_test)  # 添加一维以匹配真实标签的形状
                accuracys.append(accuracy)
                f1 = calculate_f1_score(pred_test, true_test)  # 添加一维以匹配真实标签的形状
                f1s.append(f1)
                
                
# 将张量移动到 CPU 上，并转换为 NumPy 数组
dices_cpu = [d.cpu().numpy() for d in dices]
ious_cpu = [i.cpu().numpy() for i in ious]


# 计算平均值
average_dice = np.mean(dices_cpu)
average_iou = np.mean(ious_cpu)
average_precision = np.mean(precisions)
average_recall = np.mean(recalls)
average_accuracy = np.mean(accuracys)
average_f1 = np.mean(f1s)

print("Average Dice Score:", average_dice)
print("Average IOU:", average_iou)
print("Average Precision:", average_precision)
print("Average Recall:", average_recall)
print("Average Accuracy:", average_accuracy)
print("Average F1 Score:", average_f1)