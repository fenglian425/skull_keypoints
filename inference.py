from config import *
from evaluate import accuracy, get_max_preds
from model import get_model
import torch
from dataset import get_loaders
import time
from utils import AverageMeter
from loss import *
import numpy as np
from utils import visualize_keypoints
import cv2
import os


def evaluate(cfg: Config, weight=None, show=True):
    val_dir = os.path.join(cfg.WEIGHT_DIR, cfg.NAME, 'val')
    os.makedirs(val_dir, exist_ok=True)

    val_batch_time = AverageMeter()
    val_data_time = AverageMeter()
    val_losses = AverageMeter()
    val_acc = AverageMeter()

    model = get_model(cfg)
    model.cuda()

    model.load_state_dict(torch.load(weight))
    train_loader, val_loader = get_loaders(cfg)

    val_end = time.time()
    model.eval()

    if cfg.LOSS_TYPE == 'MSE':
        criterion = JointsMSELoss(use_target_weight=False)
    elif cfg.LOSS_TYPE == 'OHKM':
        criterion = JointsOHKMMSELoss(use_target_weight=True)

    num = 0
    print(f'image size {cfg.IMAGE_SIZE}, heatmap size {cfg.HEATMAP_SIZE}')
    for i, (input, target, target_weight) in enumerate(val_loader):
        val_data_time.update(time.time() - val_end)

        with torch.no_grad():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target, target_weight)

        preds, pred_val = get_max_preds(output.detach().cpu().numpy())
        gts, gt_val = get_max_preds(target.detach().cpu().numpy())
        if show:
            for idx in range(input.shape[0]):
                num += 1

                pred = preds[idx]
                gt = gts[idx]

                diff = gt - pred
                # print(diff.shape)
                l1 = abs(diff[:, 0]) + abs(diff[:, 1])
                l2 = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
                # print(num)
                # print('l1', np.argmax(l1),np.max(l1),np.mean(l1),np.round(l1,2).tolist())
                print('l2', np.max(l2),np.mean(l2),np.argmax(l2),np.argmax(l2),np.argmax(l2),np.round(l2,2).tolist())
                # print(f'l1 max:{}')

                img = input[idx].detach().cpu().numpy()
                img = (img * 255).transpose(1, 2, 0).astype(np.uint8)

                # pred = pred[idx] * 4
                # gt = gt[idx] * 4

                scale = cfg.SHOW_SCALE
                img_with_pred = visualize_keypoints(img, pred*4, scale=scale)
                img_with_gt = visualize_keypoints(img, gt*4, scale=scale)

                # cv2.imwrite(os.path.join(val_dir, f'{num}.jpg'), np.concatenate([img_with_pred, img_with_gt], axis=1))
                # cv2.imshow('pred', img_with_pred)
                # cv2.imshow('gt', img_with_gt)
                # cv2.waitKey()

        val_losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())

        val_acc.update(avg_acc, cnt)

        # measure elapsed time
        val_batch_time.update(time.time() - val_end)
        val_end = time.time()

        # if i % cfg.PRINT_FREQ == 0:
        msg = 'Epoch: [{0}][{1}/{2}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'Val_Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
              'Val_Accuracy {acc_val} ({acc_avg})'.format(
            0, i, len(val_loader), batch_time=val_batch_time,
            speed=input.size(0) / val_batch_time.val,
            data_time=val_data_time, loss=val_losses, acc_val=np.round(val_acc.val, 3),
            acc_avg=np.round(val_acc.avg, 3))
        print(msg)


if __name__ == '__main__':
    cfg = Config()
    cfg = Res512Config()
    cfg = W48Res512Config()
    cfg=W48Res512TopKConfig()
    evaluate(cfg, weight='weights/hrnet_w48_512_topk/65.pth')
    # evaluate(cfg, weight='weights/hrnet_w32_512/85.pth')
