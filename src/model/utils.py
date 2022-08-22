import torch
from torch.nn import init, DataParallel

from src.config import (
    N_EPOCH_INIT,
    N_EPOCHS,
    N_EPOCHS_DECAY,
)


def init_net(net, device) -> DataParallel:
    net.to(device)
    net = DataParallel(net)
    net.apply(init_func)
    return net


def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch + N_EPOCH_INIT - N_EPOCHS) / float(N_EPOCHS_DECAY + 1)
    return lr_l


def init_func(m, init_gain=0.02):
    classname = m.__class__.__name__
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        init.xavier_normal_(m.weight.data, gain=init_gain)

        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif (
        classname.find("BatchNorm2d") != -1
    ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)
