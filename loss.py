import torch
from utils.dealmat import *
from timesformer_pytorch import MUFormer
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from torchsummary import summary
from torch.linalg import lstsq

class SAD(nn.Module):
    def __init__(self, num_bands):
        super(SAD, self).__init__()
        self.num_bands = num_bands

    def forward(self, inp, target):
        input_norm = torch.sqrt(torch.bmm(inp.view(-1, 1, self.num_bands),
                                          inp.view(-1, self.num_bands, 1)))
        target_norm = torch.sqrt(torch.bmm(target.view(-1, 1, self.num_bands),
                                           target.view(-1, self.num_bands, 1)))

        summation = torch.bmm(inp.view(-1, 1, self.num_bands), target.view(-1, self.num_bands, 1))
        angle = torch.acos(summation / (input_norm * target_norm))

        return angle
L = 224
loss2 = SAD(L)

def sad_loss(out, HSI):
    loss_sad_1 = loss2(out[0, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 0, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_1 = loss_sad_1[~torch.isnan(loss_sad_1)]
    loss_sad_1 = torch.sum(loss_sad_1)
    
    loss_sad_2 = loss2(out[1, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 1, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_2 = loss_sad_2[~torch.isnan(loss_sad_2)]
    loss_sad_2 = torch.sum(loss_sad_2)
    
    loss_sad_3 = loss2(out[2, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 2, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_3 = loss_sad_3[~torch.isnan(loss_sad_3)]
    loss_sad_3 = torch.sum(loss_sad_3)
    
    loss_sad_4 = loss2(out[3, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 3, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_4 = loss_sad_4[~torch.isnan(loss_sad_4)]
    loss_sad_4 = torch.sum(loss_sad_4)
    
    loss_sad_5 = loss2(out[4, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 4, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_5 = loss_sad_5[~torch.isnan(loss_sad_5)]
    loss_sad_5 = torch.sum(loss_sad_5)
    
    loss_sad_6 = loss2(out[5, :, :, :].contiguous().view(1, L, -1).transpose(1, 2),
                        HSI[0, 5, :, :, :].contiguous().view(1, L, -1).transpose(1, 2))
    loss_sad_6 = loss_sad_6[~torch.isnan(loss_sad_6)]
    loss_sad_6 = torch.sum(loss_sad_6)
    
    return loss_sad_1, loss_sad_2, loss_sad_3, loss_sad_4, loss_sad_5, loss_sad_6

