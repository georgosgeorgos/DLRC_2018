import numpy as np
import torch as th
import torch.nn as nn

class LossReconstruction(nn.Module):
    def __init__(self):
        super(LossReconstruction, self).__init__()
        self.mse = nn.MSELoss()
        self.L_huber = nn.SmoothL1Loss()

    def forward(self, input, tgt):
        mse = self.mse(input, tgt)
        return mse