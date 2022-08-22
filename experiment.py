from src.model.font_trans_gan import FTGAN


model = FTGAN()

import torch
checkpoint = torch.load('checkpoints/test_new_dataset/test_net_G.pth')
model.netG.load_state_dict(checkpoint)

print('stop')