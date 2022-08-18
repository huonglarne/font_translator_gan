from pathlib import Path
import functools
from torch import nn

BATCH_SIZE = 56
LR = 0.0002

LAMBDA_STYLE = 1
LAMBDA_CONTENT = 1
LAMBDA_L1 = 100
BETA1 = 0.5

NUM_WORKERS = 4

# font dataset
DATA_ROOT = "./datasets/font/train/chinese"
STYLE_CHANNEL = 6
IMG_SIZE = 64

TRAIN_PATH = Path("./datasets/font/train")
