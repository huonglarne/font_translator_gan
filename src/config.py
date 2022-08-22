import os
from pathlib import Path
import torch

NUM_WORKERS = os.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 56
LR = 0.0002

N_EPOCH_INIT = 0  # epoch number when training starts. different from 0 if restarting training from a checkpoint
N_EPOCHS = 10  # total number of epochs to train with constant learning rate
N_EPOCHS_DECAY = 10  # number of epochs to linearly decay learning rate to zero after training for N_EPOCHS

LAMBDA_STYLE = 1
LAMBDA_CONTENT = 1
LAMBDA_L1 = 100
BETA1 = 0.5

STYLE_CHANNEL = 6
IMG_SIZE = 64

TRAIN_PATH = Path("./datasets/font/train")
