import cv2
import imgaug.augmenters

from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np
import torch

rng = np.random.RandomState(666)
COLORS = rng.randint(150, 255, (37, 3))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def load_image(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image = np.tile(image[..., np.newaxis], (1, 1, 3))
    return image


def resize(image, side_length):
    height, width = image.shape[:2]

    ratio = width / height

    if width > height:
        scale = width / side_length
        width = side_length
        height = int(side_length / ratio)
    else:
        scale = height / side_length
        width = int(side_length * ratio)
        height = side_length

    # cv2.imshow('ori', image)
    image = cv2.resize(image, (width, height))
    return image, scale


def pad(image, side_length):
    height, width = image.shape[:2]
    dx = (side_length - width) // 2
    dy = (side_length - height) // 2

    image = np.pad(image, ((dy, side_length - height - dy), (dx, side_length - width - dx), (0, 0)))
    return image, (dx, dy)


def resize_and_pad(image, side_length, kps=None):
    image, scale = resize(image, side_length)
    image, offsets = pad(image, side_length)

    # print(scale,offsets, image.shape)
    meta = scale, offsets
    if kps is None:
        return image, meta
    else:
        kps = np.array(kps)
        kps /= scale
        kps += offsets
        return image, kps, meta


def preprocess_image(image_file, side_length, roi=None):
    ori_image = load_image(image_file)

    if roi is not None:
        x1, y1, x2, y2 = list(map(int, roi[:4]))
        image = ori_image[y1:y2, x1:x2]
    else:
        x1, y1 = 0, 0
        image = ori_image

    image, meta = resize_and_pad(image, side_length)

    return image, ori_image, (*meta, (x1, y1))


def normalize_image(image, device='cuda'):
    image = image / 255
    image = np.transpose(image.astype(np.float32), (2, 0, 1))
    image = torch.tensor(image[np.newaxis, ...], device=device)
    return image


def visualize_keypoints(img, kps, scale=2, kps_name=None):
    img = img.copy()
    if isinstance(kps, list):
        kps = [(x, y) for x, y in kps]
    elif isinstance(kps, KeypointsOnImage):
        kps = [(kp.x, kp.y) for kp in kps.keypoints]

    height, width = img.shape[:2]
    img = cv2.resize(img, (width * scale, height * scale))
    kps = [(int(x * scale), int(y * scale)) for x, y in kps]

    font_size = 0.25*scale
    for i, (x, y) in enumerate(kps):
        color = COLORS[i].tolist()

        cv2.circle(img, (x, y), 3, (0, 255, 0))
        if kps_name is None:
            cv2.putText(img, f'{i + 1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255))
        else:
            cv2.putText(img, kps_name[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255))

    return img


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
