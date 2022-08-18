from pathlib import Path
import functools
from torch import nn

BATCH_SIZE = 56
LR = 0.0002
BETA1 = 0.5

NUM_WORKERS = 4

# font dataset
DATA_ROOT = "./datasets/font/train/chinese"
STYLE_CHANNEL = 6
IMG_SIZE = 64

TRAIN_PATH = Path("./datasets/font/train")

NORM_LAYER = functools.partial(
            nn.BatchNorm2d, affine=True, track_running_stats=True
        )
