# import torch
# import pickle
# import numpy as np
# import argparse
# import logging
# import torch.nn.functional as F

# from unet import UNet
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
# from utils.dice_score import multiclass_dice_coeff, dice_coeff


# def calculate_dsc(mask_pred, mask_true, num_classes):
#     dices = []
#     for class_id in range(1, num_classes):  # iterate over each class (excluding background)
#         pred_mask_class = (mask_pred == class_id).float()
#         true_mask_class = (mask_true == class_id).float()
#         intersection = torch.sum(pred_mask_class * true_mask_class)
#         union = torch.sum(pred_mask_class) + torch.sum(true_mask_class)
#         dice = (2. * intersection + 1e-8) / (union + 1e-8)
#         dices.append(dice)
#     return dices


# def calculate_iou(mask_pred, mask_true, num_classes):
#     ious = []
#     for class_id in range(1, num_classes):  # iterate over each class (excluding background)
#         pred_mask_class = (mask_pred == class_id).float()
#         true_mask_class = (mask_true == class_id).float()
#         intersection = torch.sum(pred_mask_class * true_mask_class)
#         union = torch.sum(pred_mask_class) + torch.sum(true_mask_class) - intersection
#         iou = (intersection + 1e-8) / (union + 1e-8)
#         ious.append(iou)
#     return ious


# def calculate_accuracy(mask_pred, mask_true):
#     # Flatten the masks
#     mask_pred_flat = mask_pred.view(-1)
#     mask_true_flat = mask_true.view(-1)

#     # Calculate accuracy
#     accuracy = accuracy_score(mask_true_flat.cpu(), mask_pred_flat.cpu())

#     return accuracy


# def calculate_recall(mask_pred, mask_true, num_classes):
#     recalls = []
#     for class_id in range(1, num_classes):  # iterate over each class (excluding background)
#         pred_mask_class = (mask_pred == class_id).float()
#         true_mask_class = (mask_true == class_id).float()
#         recall = recall_score(true_mask_class.cpu().numpy().flatten(),
#                               pred_mask_class.cpu().numpy().flatten(), zero_division=1)
#         recalls.append(recall)
#     return recalls


# def calculate_f1_score(mask_pred, mask_true, num_classes):
#     f1s = []
#     for class_id in range(1, num_classes):  # iterate over each class (excluding background)
#         pred_mask_class = (mask_pred == class_id).float()
#         true_mask_class = (mask_true == class_id).float()
#         f1 = f1_score(true_mask_class.cpu().numpy().flatten(),
#                       pred_mask_class.cpu().numpy().flatten(), zero_division=1)
#         f1s.append(f1)
#     return f1s


# def calculate_precision(mask_pred, mask_true, num_classes):
#     precisions = []
#     for class_id in range(1, num_classes):  # iterate over each class (excluding background)
#         pred_mask_class = (mask_pred == class_id).float()
#         true_mask_class = (mask_true == class_id).float()
#         precision = precision_score(true_mask_class.cpu().numpy().flatten(),
#                                     pred_mask_class.cpu().numpy().flatten(), zero_division=1)
#         precisions.append(precision)
#     return precisions


# def get_args():
#     parser = argparse.ArgumentParser(description='Predict masks from input images')
#     parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
#                         help='Specify the file in which the model is stored')
#     parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
#                         help='Minimum probability value to consider a mask pixel white')
#     parser.add_argument('--scale', '-s', type=float, default=0.5,
#                         help='Scale factor for the input images')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

#     return parser.parse_args()


# if __name__ == '__main__':
#     args = get_args()
#     net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Loading model {args.model}')
#     logging.info(f'Using device {device}')

#     net.to(device=device)
#     state_dict = torch.load(args.model, map_location=device)
#     mask_values = state_dict.pop('mask_values', [0, 1])
#     net.load_state_dict(state_dict)
#     logging.info('Model loaded!')

#     with open('/root/autodl-tmp/Pytorch-UNet-master/data_loaders.pkl', 'rb') as f:
#         data = pickle.load(f)

#     # 获取验证集数据
#     test_data = data['test']

#     # 初始化存储每个类别指标的列表
#     dices = [[] for _ in range(args.classes)]
#     ious = [[] for _ in range(args.classes)]
#     precisions = [[] for _ in range(args.classes)]
#     recalls = [[] for _ in range(args.classes)]
#     accuracys = []
#     f1s = []

#     net.eval()
#     with torch.no_grad():
#         for i, batch in enumerate(test_data):
#             test_images = batch['image'].to(device)
#             true_tests = batch['mask'].to(device)
#             output = net(test_images)
#             pred_tests = output.argmax(dim=1)

#             for j, (pred_test, true_test) in enumerate(zip(pred_tests, true_tests)):
#                 for class_id in range(1, args.classes):
#                     dice = calculate_dsc(pred_test, true_test, args.classes)[class_id]
#                     dices[class_id].append(dice.item())
#                     iou = calculate_iou(pred_test, true_test, args.classes)[class_id]
#                     ious[class_id].append(iou.item())
#                     precision = calculate_precision(pred_test, true_test, args.classes)[class_id]
#                     precisions[class_id].append(precision)
#                     recall = calculate_recall(pred_test, true_test, args.classes)[class_id]
#                     recalls[class_id].append(recall)

#                 accuracy = calculate_accuracy(pred_test, true_test)
#                 accuracys.append(accuracy)
#                 f1 = calculate_f1_score(pred_test, true_test, args.classes)[class_id]
#                 f1s.append(f1)

#     # 计算每个类别的平均值
#     average_dices = [np.mean(d) for d in dices]
#     average_ious = [np.mean(i) for i in ious]
#     average_precisions = [np.mean(p) for p in precisions]
#     average_recalls = [np.mean(r) for r in recalls]
#     average_accuracys = np.mean(accuracys)
#     average_f1s = np.mean(f1s)

#     # 打印每个类别的平均值
#     for class_id in range(args.classes):
#         print(f"Class {class_id}:")
#         print(f"  Average Dice Score: {average_dices[class_id]}")
#         print(f"  Average IOU: {average_ious[class_id]}")
#         print(f"  Average Precision: {average_precisions[class_id]}")
#         print(f"  Average Recall: {average_recalls[class_id]}")

#     print(f"Overall Metrics:")
#     print(f"  Average Accuracy: {average_accuracys}")
#     print(f"  Average F1 Score: {average_f1s}")
import torch
import pickle
import numpy as np
import argparse
import logging
import torch.nn.functional as F

from unet import UNet
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def calculate_dsc(mask_pred, mask_true, num_classes):
    dices = []
    for class_id in range(num_classes):  # iterate over each class including background
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()
        intersection = torch.sum(pred_mask_class * true_mask_class)
        union = torch.sum(pred_mask_class) + torch.sum(true_mask_class)
        dice = (2. * intersection + 1e-8) / (union + 1e-8)
        dices.append(dice)
    return dices


def calculate_iou(mask_pred, mask_true, num_classes):
    ious = []
    for class_id in range(num_classes):  # iterate over each class including background
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()
        intersection = torch.sum(pred_mask_class * true_mask_class)
        union = torch.sum(pred_mask_class) + torch.sum(true_mask_class) - intersection
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)
    return ious


def calculate_accuracy(mask_pred, mask_true, num_classes):
    accuracys = []
    for class_id in range(num_classes):
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()

        # Calculate accuracy
        accuracy = accuracy_score(true_mask_class.cpu().numpy().flatten(),
                              pred_mask_class.cpu().numpy().flatten())
        accuracys.append(accuracy)
    return accuracys


def calculate_recall(mask_pred, mask_true, num_classes):
    recalls = []
    for class_id in range(num_classes):  # iterate over each class including background
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()
        recall = recall_score(true_mask_class.cpu().numpy().flatten(),
                              pred_mask_class.cpu().numpy().flatten(), zero_division=1)
        recalls.append(recall)
    return recalls


def calculate_f1_score(mask_pred, mask_true, num_classes):
    f1s = []
    for class_id in range(num_classes):  # iterate over each class including background
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()
        f1 = f1_score(true_mask_class.cpu().numpy().flatten(),
                      pred_mask_class.cpu().numpy().flatten(), zero_division=1)
        f1s.append(f1)
    return f1s


def calculate_precision(mask_pred, mask_true, num_classes):
    precisions = []
    for class_id in range(num_classes):  # iterate over each class including background
        pred_mask_class = (mask_pred == class_id).float()
        true_mask_class = (mask_true == class_id).float()
        precision = precision_score(true_mask_class.cpu().numpy().flatten(),
                                    pred_mask_class.cpu().numpy().flatten(), zero_division=1)
        precisions.append(precision)
    return precisions


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

    with open('/root/autodl-tmp/data_loaders.pkl', 'rb') as f:
        data = pickle.load(f)

    # 获取验证集数据
    test_data = data['test']

    # 初始化存储每个类别指标的列表
    dices = [[] for _ in range(args.classes)]
    ious = [[] for _ in range(args.classes)]
    precisions = [[] for _ in range(args.classes)]
    recalls = [[] for _ in range(args.classes)]
    accuracys = [[] for _ in range(args.classes)]
    f1s = [[] for _ in range(args.classes)]

    net.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_data):
            test_images = batch['image'].to(device)
            true_tests = batch['mask'].to(device)
            output = net(test_images)
            pred_tests = output.argmax(dim=1)

            for j, (pred_test, true_test) in enumerate(zip(pred_tests, true_tests)):
                dice_list = calculate_dsc(pred_test, true_test, args.classes)
                iou_list = calculate_iou(pred_test, true_test, args.classes)
                precision_list = calculate_precision(pred_test, true_test, args.classes)
                recall_list = calculate_recall(pred_test, true_test, args.classes)
                f1_list = calculate_f1_score(pred_test, true_test, args.classes)
                accuracy_list = calculate_accuracy(pred_test, true_test, args.classes)

                for class_id in range(args.classes):
                    dices[class_id].append(dice_list[class_id].item())
                    ious[class_id].append(iou_list[class_id].item())
                    precisions[class_id].append(precision_list[class_id])
                    recalls[class_id].append(recall_list[class_id])
                    f1s[class_id].append(f1_list[class_id])
                    accuracys[class_id].append(accuracy_list[class_id])

    # 计算每个类别的平均值
    average_dices = [np.mean(d) for d in dices]
    average_ious = [np.mean(i) for i in ious]
    average_precisions = [np.mean(p) for p in precisions]
    average_recalls = [np.mean(r) for r in recalls]
    average_accuracys = [np.mean(a) for a in accuracys]
    average_f1s = [np.mean(f1) for f1 in f1s]

    # 打印每个类别的平均值
    for class_id in range(args.classes):
        print(f"Class {class_id}:")
        print(f"  Average Dice Score: {average_dices[class_id]}")
        print(f"  Average IOU: {average_ious[class_id]}")
        print(f"  Average Precision: {average_precisions[class_id]}")
        print(f"  Average Recall: {average_recalls[class_id]}")
        print(f"  Average F1 Score: {average_f1s[class_id]}")
        print(f"  Average Accuracy: {average_accuracys[class_id]}")

    # print(f"Overall Metrics:")
    # print(f"  Average Accuracy: {average_accuracys}")
