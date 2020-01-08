from model import get_model
from dataset import get_dataset, get_loaders
from config import *
from loss import *
from utils import AverageMeter
from evaluate import accuracy, get_max_preds
import time
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import cv2
import torch
from utils import visualize_keypoints
import os


def train(cfg: Config, validate=False):
    weight_dir = os.path.join(cfg.WEIGHT_DIR, cfg.NAME)
    tmp_dir = os.path.join(weight_dir, 'tmp')

    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # switch to train mode
    model = get_model(cfg)
    # print(model)
    model.cuda()
    train_loader, val_loader = get_loaders(cfg)

    if cfg.LOSS_TYPE == 'MSE':
        criterion = JointsMSELoss(use_target_weight=True)
    elif cfg.LOSS_TYPE == 'OHKM':
        criterion = JointsOHKMMSELoss(use_target_weight=True, topk=cfg.TopK)

    if cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.LR,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
            nesterov=cfg.NESTEROV
        )
    elif cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.LR
        )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.LR_STEP, cfg.LR_FACTOR,
    )

    for epoch in range(cfg.EPOCH):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        lr_scheduler.step()

        end = time.time()
        model.train()
        for i, train_batch in enumerate(train_loader, start=1):
            # measure data loading time
            if 'hro' in cfg.MODEL_TYPE:
                input, target, target_weight, offsets= train_batch
            else:
                input, target, target_weight= train_batch

            data_time.update(time.time() - end)

            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            # compute output
            output = model(input)
            loss = criterion(output, target, target_weight)

            # if isinstance(outputs, list):
            #     loss = criterion(outputs[0], target, target_weight)
            #     for output in outputs[1:]:
            #         loss += criterion(output, target, target_weight)
            # else:
            #     output = outputs
            #     loss = criterion(output, target, target_weight)

            # loss = criterion(output, target, target_weight)

            # compute gradient and do update step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % cfg.PRINT_FREQ == 0 or i == len(train_loader):
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Accuracy {acc_val} ({acc_avg})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    speed=input.size(0) / batch_time.val,
                    data_time=data_time, loss=losses, acc_val=np.round(acc.val, 3),
                    acc_avg=np.round(acc.avg, 3))
                print(msg)
                # logger.info(msg)
                #
                # writer = writer_dict['writer']
                # global_steps = writer_dict['train_global_steps']
                # writer.add_scalar('train_loss', losses.val, global_steps)
                # writer.add_scalar('train_acc', acc.val, global_steps)
                # writer_dict['train_global_steps'] = global_steps + 1
                #
                # prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
                # save_debug_images(config, input, meta, target, pred * 4, output,
                #                   prefix)
        if not validate:
            continue

        val_end = time.time()
        model.eval()

        val_batch_time = AverageMeter()
        val_data_time = AverageMeter()
        val_losses = AverageMeter()
        val_acc = AverageMeter()
        show = True
        for i, (input, target, target_weight) in enumerate(val_loader, start=1):
            # measure data loading time
            val_data_time.update(time.time() - val_end)

            with torch.no_grad():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                target_weight = target_weight.cuda(non_blocking=True)
                # compute output
                output = model(input)
                loss = criterion(output, target, target_weight)

            if show == 0:
                pred, pred_val = get_max_preds(output.detach().cpu().numpy())
                gt, gt_val = get_max_preds(target.detach().cpu().numpy())

                print(input.shape, pred.shape, gt.shape)
                idx = np.random.randint(0, input.shape[0])
                img = input[idx].detach().cpu().numpy()
                img = (img * 255).transpose(1, 2, 0).astype(np.uint8)
                pred = pred[idx] * 4
                gt = gt[idx] * 4

                scale = cfg.SHOW_SCALE
                # print(np.concatenate([pred, gt], axis=-1) * scale)
                # print(gt)
                img_with_pred = visualize_keypoints(img, pred, scale=scale)
                img_with_gt = visualize_keypoints(img, gt, scale=scale)

                cv2.imwrite(os.path.join(tmp_dir, f'{epoch}.jpg'), np.concatenate([img_with_pred, img_with_gt], axis=1))
                show = False

            if epoch % cfg.SAVE_FREQ == 0:
                torch.save(model.state_dict(), os.path.join(weight_dir, f'{epoch}.pth'))

            val_losses.update(loss.item(), input.size(0))

            _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                             target.detach().cpu().numpy())
            val_acc.update(avg_acc, cnt)

            # measure elapsed time
            val_batch_time.update(time.time() - val_end)
            val_end = time.time()

            if i % cfg.PRINT_FREQ == 0 or i == len(val_loader):
                msg = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Speed {speed:.1f} samples/s\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'Val_Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                      'Val_Accuracy {acc_val} ({acc_avg})'.format(
                    epoch, i, len(val_loader), batch_time=val_batch_time,
                    speed=input.size(0) / val_batch_time.val,
                    data_time=val_data_time, loss=val_losses, acc_val=np.round(val_acc.val, 3),
                    acc_avg=np.round(val_acc.avg, 3))
                print(msg)


if __name__ == '__main__':
    cfg = Config()
    cfg = Res512Config()
    cfg = W48Res512Config()
    cfg = W48Res512TopKConfig()
    cfg = W48AugConfig()

    cfg = W48Res512TopKFlipConfig()
    cfg = W48Res512TopKFlipBlurConfig()
    cfg = ExW48()
    train(cfg, validate=True)
