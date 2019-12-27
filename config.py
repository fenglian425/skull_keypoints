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

    # model
    MODEL_TYPE = 'hrnet_w32'
    LOSS_TYPE = 'OHKM'

    # train params
    EPOCH = 100
    BATCH_SIZE = 16
    OPTIMIZER = 'adam'
    LR = 0.001
    LR_FACTOR = 0.1
    LR_STEP = (20, 40)
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-5
    NESTEROV = False

    # info
    PRINT_FREQ = 10
    SAVE_FREQ = 5
    SHOW_SCALE = 4

    def get_snapshot(self):
        pass


class Res512Config(Config):
    NAME = 'hrnet_w32_512'

    IMAGE_SIZE = (512, 512)
    HEATMAP_SIZE = (128, 128)

    SHOW_SCALE = 2
    pass
