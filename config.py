ROOT_DIR = 'data/origin/'
IMAGE_DIR = 'data/origin/img'
LABEL_DIR = 'data/origin/label'


class Config():
    NAME = 'w32_256'
    FOLD = 0
    FOLD_NUM = 10
    FOLD_SEED = 111

    # data directory
    ROOT_DIR = 'data/origin/'
    IMAGE_DIR = 'data/origin/img'
    LABEL_DIR = 'data/origin/label'
    KEYPOINT_NAMES = ["G'", "N'", "Prn", "Sn", "Ls",
                      "St", "Li", "Si", "Pog'", "Gn'",
                      "Me'", "N", "Or", "ANS", "A",
                      "UIA", "SPr", "UI", "LI", "Id",
                      "LIA", "B", "Pog", "Gn", "Me",
                      "U6", "L6", "Go", "PNS", "Ptm",
                      "Ar", "Co", "S", "Ba", "P",
                      "ruler0", "ruler1"]

    # weight directory
    WEIGHT_DIR = 'weights'

    # dataset params
    IMAGE_SIZE = (256, 256)
    HEATMAP_SIZE = (64, 64)
    MEAN = 0.
    STD = 1.

    HEATMAP_TYPE = 'gaussian'
    SIGMA = 2
    NUM_KEYPOINTS = 35
    KEYPOINTS_TYPE = 'excluded'
    IGNORE_KEYPOINTS = False

    # data augmentation
    AUGMENTATION_SEED = 222
    SHIFT_FACTOR = 0.1
    SCALE_FACTOR = 0.25
    ROTATION_FACTOR = 0.2
    FLIPLR = False
    BLUR = False

    # model
    MODEL_TYPE = 'w32'
    LOSS_TYPE = 'OHKM'
    TopK = 12

    # train params
    EPOCH = 100
    BATCH_SIZE = 16
    OPTIMIZER = 'adam'
    LR = 0.001
    LR_FACTOR = 0.1
    LR_STEP = (20, 60)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    NESTEROV = False

    # info
    PRINT_FREQ = 10
    SAVE_FREQ = 5
    SHOW_SCALE = 4
    TEST = False

    def get_snapshot(self):
        pass


class Res512Config(Config):
    NAME = 'hrnet_w32_512'

    IMAGE_SIZE = (512, 512)
    HEATMAP_SIZE = (128, 128)

    BATCH_SIZE = 8
    SHOW_SCALE = 2
    pass


class W48Res512Config(Config):
    NAME = 'hrnet_w48_512'

    IMAGE_SIZE = (512, 512)
    HEATMAP_SIZE = (128, 128)

    MODEL_TYPE = 'w48'

    BATCH_SIZE = 4
    SHOW_SCALE = 2
    pass


class W48Res512TopKConfig(Config):
    NAME = 'hrnet_w48_512_topk'

    IMAGE_SIZE = (512, 512)
    HEATMAP_SIZE = (128, 128)

    EPOCH = 100

    MODEL_TYPE = 'w48'

    BATCH_SIZE = 4
    SHOW_SCALE = 2

    TopK = 5

    PRINT_FREQ = 5
    pass


class W48Res512TopKFlipConfig(W48Res512TopKConfig):
    NAME = 'hrnet_w48_512_flip_topk'

    FLIPLR = True


class W48Res512TopKFlipBlurConfig(W48Res512TopKConfig):
    NAME = 'hrnet_w48_512_flip_blur_topk'

    FLIPLR = True
    BLUR = True

class W48AugConfig(W48Res512Config):
    NAME = 'hrnet_w48_512_aug'

    SHIFT_FACTOR = 0.2
    SCALE_FACTOR = 0.1
    ROTATION_FACTOR = 20
    pass


class W48HM256Config(W48Res512Config):
    NAME = 'hrnet_w48_512_256'

    HEATMAP_SIZE = (256, 256)
    # SHIFT_FACTOR = 0.2
    # SCALE_FACTOR = 0.1
    ROTATION_FACTOR = 10
    # pass

class ExW48(W48Res512TopKFlipConfig):
    NAME = 'w64'
    MODEL_TYPE = 'w64'


class HROW48(W48Res512TopKFlipConfig):
    NAME = 'HRO_w48'

    MODEL_TYPE = 'hro_w48'
if __name__ == '__main__':
    cfg = Config()
    for i, name in enumerate(cfg.KEYPOINT_NAMES):
        print(i+1, name)
