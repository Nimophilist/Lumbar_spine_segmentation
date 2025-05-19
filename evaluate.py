import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    criterion = nn.CrossEntropyLoss() if net.n_classes > 1 else nn.BCEWithLogitsLoss()
    # dice_score = 0
    val_loss = 0                     # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):  
        with tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False) as pbar:
            for iteration, batch in enumerate(dataloader):
                image, mask_true = batch['image'], batch['mask']

                # move images and labels to correct device and type
                image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                mask_true = mask_true.to(device=device, dtype=torch.long)
                # single_true_vector = torch.split(mask_true, 1, dim=1)
                # single_true_vector = [torch.squeeze(mask, dim=1) for mask in single_true_vector]

                # predict the mask
                mask_pred = net(image)
                # single_pred_vector = torch.split(mask_pred, 1, dim=1)
                # single_pred_vector = [torch.squeeze(mask, dim=1) for mask in single_pred_vector]
                # num_slice = len(single_pred_vector)
                # for pred, true in zip(single_pred_vector, single_true_vector):
                if net.n_classes == 1:
                            # 计算二分类损失和Dice损失
                    loss = criterion(mask_pred.squeeze(1), mask_true.float())
                    loss += dice_loss(F.sigmoid(mask_pred.squeeze(1)), mask_true.float(), multiclass=False)
                else:
                    # 计算多分类损失和Dice损失
                    # temp_pred = mask_pred.permute(0, 2, 3, 1).contiguous().view(-1, net.n_classes)
                    # temp_true = mask_true.view(-1)
                    loss = criterion(mask_pred, mask_true)
                    loss += dice_loss(
                            F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    # temp_pred = mask_pred.transpose(2, 3).transpose(3, 4).contiguous().view(-1, mask_pred.size(2))
                    # temp_true = mask_true.view(-1)
                    # loss = criterion(temp_pred, temp_true)
                    # loss += dice_loss(
                    #             F.softmax(mask_pred, dim=2).float(),
                    #             F.one_hot(mask_true, net.n_classes).permute(0, 1, 4, 2, 3).float(),
                    #             multiclass=True
                    #         )
                val_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': val_loss/(iteration + 1)})
    
    net.train()
    return val_loss