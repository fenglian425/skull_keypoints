import cv2
from imgaug.augmentables.kps import KeypointsOnImage
import numpy as np

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


def visualize_keypoints(img, kps, with_num=True, scale=2):
    img = img.copy()
    if isinstance(kps, list):
        kps = [(x, y) for x, y in kps]
    elif isinstance(kps, KeypointsOnImage):
        kps = [(kp.x, kp.y) for kp in kps.keypoints]

    height, width = img.shape[:2]
    img = cv2.resize(img, (width * scale, height * scale))
    kps = [(int(x * scale), int(y * scale)) for x, y in kps]

    for i, (x, y) in enumerate(kps):
        color = COLORS[i].tolist()

        cv2.circle(img, (x, y), 3, (0, 255, 0))
        if with_num:
            cv2.putText(img, f'{i + 1}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    return img


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
