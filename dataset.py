from torch.utils.data import Dataset, DataLoader
from config import *
import numpy as np
import os
from skimage.io import imread
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from utils import visualize_keypoints, resize_and_pad


# 25-34 11 12 0 1
class KeyPointDataset(Dataset):
    def __init__(self, list_ids, is_train, cfg: Config):
        self.list_ids = list_ids
        self.is_train = is_train

        self.cfg = cfg
        self.img_dir = cfg.IMAGE_DIR
        self.label_dir = cfg.LABEL_DIR

        self.heatmap_type = cfg.HEATMAP_TYPE
        self.sigma = cfg.SIGMA
        self.num_keypoints = cfg.NUM_KEYPOINTS

        self.image_size = cfg.IMAGE_SIZE
        self.heatmap_size = cfg.HEATMAP_SIZE

        self.rng = np.random.RandomState(cfg.AUGMENTATION_SEED)

        self.crop_center = iaa.CropToFixedSize(width=self.image_size[0], height=self.image_size[1], position='center')
        self.normalize = iaa.Sequential([
            iaa.PadToFixedSize(width=self.image_size[0], height=self.image_size[1], position='center'),
            iaa.CropToFixedSize(width=self.image_size[0], height=self.image_size[1], position='center')
        ])

        self.joints_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)

        self.augmentation_policies = [
            iaa.Multiply((0.8, 1.2)),  # change brightness, doesn't affect keypoints
            iaa.Affine(
                translate_percent=(-cfg.SHIFT_FACTOR, cfg.SHIFT_FACTOR),
                rotate=(-cfg.ROTATION_FACTOR, +cfg.ROTATION_FACTOR),
                scale=(1 - cfg.SCALE_FACTOR, 1 + cfg.SCALE_FACTOR)
            )  # rotate by exactly 10deg and scale to 50-70%, affects keypoints
        ]

        if self.cfg.FLIPLR:
            self.augmentation_policies.append(iaa.Fliplr(0.5))

        if self.cfg.BLUR:
            self.augmentation_policies.append(iaa.GaussianBlur(sigma=(0., 3.)))

        self.augmenter = iaa.Sometimes(0.4, self.augmentation_policies)

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        img_file = self.list_ids[index]

        # if '204' not in img_file:
        #     return 0,0,0
        label_file = img_file.replace('jpg', 'txt')

        # image = imread(os.path.join(self.img_dir, img_file), as_gray=True)
        image = cv2.imread(os.path.join(self.img_dir, img_file), cv2.IMREAD_GRAYSCALE)
        image = np.tile(image[..., np.newaxis], (1, 1, 3))
        # cv2.imshow('ori', image)

        coords = np.loadtxt(os.path.join(self.label_dir, label_file), dtype=np.float, delimiter=',')
        if self.cfg.KEYPOINTS_TYPE == 'excluded':
            coords = coords[:35]

        height, width = image.shape[:2]
        loose_coef = 0.15
        crop_width = width * loose_coef / 2
        crop_height = height * loose_coef / 2
        xmin, ymin = coords[:, 0].min(), coords[:, 1].min()
        xmax, ymax = coords[:, 0].max(), coords[:, 1].max()
        xmin = max(0, int(xmin - crop_width))
        ymin = max(0, int(ymin - crop_height))
        xmax = min(image.shape[1], int(xmax + crop_width))
        ymax = min(image.shape[0], int(ymax + crop_height))

        image = image[ymin:ymax, xmin: xmax]
        coords -= np.array([xmin, ymin])

        # normalize
        side_length = int(self.image_size[0] * 1.)
        image, coords, _ = resize_and_pad(image, side_length, kps=coords)
        kps = ia.KeypointsOnImage([ia.Keypoint(x=c[0], y=c[1]) for c in coords], shape=image.shape)

        if self.is_train:
            if self.cfg.TEST:
                image_before = visualize_keypoints(image, kps, self.cfg.SHOW_SCALE)
                image, kps = self.augmenter(image=image, keypoints=kps)
                image_after = visualize_keypoints(image, kps, self.cfg.SHOW_SCALE)
                cv2.imshow('before', image_before)
                cv2.imshow('after', image_after)
                cv2.waitKey()
            else:
                image, kps = self.augmenter(image=image, keypoints=kps)

        heatmap, target_weight, offsets = self.generate_target(kps.keypoints)
        # print(target_weight.flatten())

        # if not target_weight.all():
        #     image_with_kp = kps.draw_on_image(image, size=3)
        #     cv2.imshow('before', image_with_kp[..., [2, 1, 0]])
        #     cv2.waitKey()
        #     print(img_file)
        #     for i in range(self.num_keypoints):
        #         kp = kps.keypoints[i]
        #         hm = heatmap[i]
        #
        #         max_idx = np.argmax(hm)
        #         idx = np.unravel_index(max_idx, hm.shape)
        #         print(kp.y / 4, kp.x / 4, idx)
        #         tmp = kp.draw_on_image((heatmap[i] * 255).astype(np.uint8), color=128)
        #         cv2.imshow('tmp', tmp)
        #         cv2.waitKey()
        #         pass

        image = image / 255
        image = np.transpose(image.astype(np.float32), (2, 0, 1))

        if 'hro' in self.cfg.MODEL_TYPE:
            return image, heatmap, target_weight, offsets
        return image, heatmap, target_weight

    def generate_target(self, coords):
        target_weight = np.ones((self.num_keypoints, 1), dtype=np.float32)
        offsets = np.zeros((self.num_keypoints, 2), dtype=np.float32)
        tmp_size = self.sigma * 3

        target = np.zeros((self.num_keypoints,
                           self.heatmap_size[1],
                           self.heatmap_size[0]),
                          dtype=np.float32)

        for joint_id in range(self.num_keypoints):
            feat_stride = np.array(self.image_size) / np.array(self.heatmap_size)

            x, y = coords[joint_id].x / feat_stride[0], coords[joint_id].y / feat_stride[1]
            mu_x = int(x + 0.5)
            mu_y = int(y + 0.5)

            print(x, y, mu_x, mu_y)
            offsets[joint_id] = (x - mu_x, y - mu_y)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                target_weight[joint_id] = 0
                # print('fuck')
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # The gaussian is not normalized, we want the center value to equal 1
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return target, target_weight, offsets


def get_dataset(cfg):
    list_ids = sorted(os.listdir(cfg.IMAGE_DIR))
    rng = np.random.RandomState(cfg.FOLD_SEED)

    fold = cfg.FOLD
    fold_num = cfg.FOLD_NUM

    rng.shuffle(list_ids)
    step = len(list_ids) // fold_num

    val_ids = list_ids[fold * step:(fold + 1) * step]
    train_ids = list_ids[:fold * step] + list_ids[(fold + 1) * step:]

    train_dataset = KeyPointDataset(train_ids, is_train=True, cfg=cfg)
    val_dataset = KeyPointDataset(val_ids, is_train=False, cfg=cfg)
    return train_dataset, val_dataset


def get_loaders(cfg: Config):
    train_dataset, val_dataset = get_dataset(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    cfg = Config()
    cfg = Res512Config()
    cfg = W48AugConfig()
    cfg = W48HM256Config()

    cfg = W48Res512TopKFlipConfig()
    cfg = W48Res512TopKFlipBlurConfig()
    cfg.TEST = True

    train_dataset, val_dataset = get_dataset(cfg)
    # train_dataset.coco.to_json()
    # val_dataset.coco.to_json()

    for image, heatmap, target_weight in train_dataset:
        # print(image.shape)
        # print(heatmap.shape)
        # cv2.imshow('image', image)
        pass
