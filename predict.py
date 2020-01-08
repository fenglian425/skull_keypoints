from config import *
from evaluate import accuracy, get_max_preds
from model import get_model
import torch
from dataset import get_loaders
import time
from utils import AverageMeter
from loss import *
import numpy as np
from utils import visualize_keypoints, load_image, preprocess_image, normalize_image
import cv2
import os
import imgaug.augmenters as iaa
import shutil


def predict(file, model, cfg: Config, roi=None, show=True):
    side_length = cfg.IMAGE_SIZE[0]

    image, ori_image, (scale, offset, pts) = preprocess_image(file, side_length=side_length, roi=roi)

    inputs = normalize_image(image)

    with torch.no_grad():
        output = model(inputs)

    preds, pred_val = get_max_preds(output.detach().cpu().numpy())
    pred = preds[0]
    pred *= 4

    show_scale = cfg.SHOW_SCALE
    img_with_pred = visualize_keypoints(image, pred, kps_name=cfg.KEYPOINT_NAMES, scale=show_scale)
    resized_image = cv2.resize(image, (side_length * show_scale, side_length * show_scale))
    img_with_pred = np.concatenate([resized_image, img_with_pred], axis=1)

    ori_pred = pred - offset
    ori_pred = ori_pred * scale
    ori_pred += np.array(pts)

    ori_with_pred = visualize_keypoints(ori_image, ori_pred, kps_name=cfg.KEYPOINT_NAMES, scale=2)

    if show:
        cv2.imshow('pred', img_with_pred)
        cv2.imshow('ori', ori_with_pred)
        # cv2.imshow('gt', img_with_gt)
        cv2.waitKey()
    return img_with_pred, ori_with_pred


if __name__ == '__main__':
    cfg = Config()
    cfg = Res512Config()
    cfg = W48Res512Config()
    cfg = W48Res512TopKConfig()
    cfg = W48Res512TopKFlipBlurConfig()

    TEST_DIR_NAME = 'lateral_data_batch_1'
    # TEST_DIR_NAME = '12_31'

    ROOT_DIR = 'data/test/'
    TEST_DIR = ROOT_DIR + TEST_DIR_NAME
    with open(os.path.join(TEST_DIR, 'rois.json')) as f:
        import json

        rois = json.load(f)
    # print(rois)

    weight = 'weights/hrnet_w48_512_topk/65.pth'
    weight = 'weights/hrnet_w48_512_flip_topk/95.pth'
    # weight = 'weights/hrnet_w48_512_flip_blur_topk/95.pth'

    model = get_model(cfg)
    model.cuda()
    model.load_state_dict(torch.load(weight))

    dst_dir = 'tmp/' + TEST_DIR_NAME
    # dst_dir = 'tmp/no_blur'
    os.makedirs(dst_dir, exist_ok=True)

    detail_dir = os.path.join(dst_dir, 'detail')
    prediction_dir = os.path.join(dst_dir, 'prediction')
    origin_dir = os.path.join(dst_dir, 'origin')

    os.makedirs(detail_dir, exist_ok=True)
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(origin_dir, exist_ok=True)

    for i, (file, roi) in enumerate(rois.items()):
        # file = file.replace('keypoints/', '')

        image_name = file.split('/')[-1]
        directory = file.split('/')[-2]

        file = os.path.join(TEST_DIR, directory, image_name)
        print(file)

        res_name = f'{directory}_{image_name}'
        print(res_name)
        # image = image[y1:y2, x1:x2]
        # print(roi)
        # img_with_pred, ori_with_pred = predict(file, model, cfg, roi=roi, show=False)
        # cv2.imwrite(os.path.join(prediction_dir, res_name), ori_with_pred)
        # cv2.imwrite(os.path.join(detail_dir, res_name), img_with_pred)
        shutil.copy(file, os.path.join(origin_dir, res_name))

    # evaluate(cfg, weight='weights/hrnet_w32_512/85.pth')
