import torch.nn as nn


class LossEntropy(nn.Module):
    def __init__(self):
        super(LossEntropy, self).__init__()
        self.entropy = nn.CrossEntropyLoss()

    def forward(self, input, tgt):
        res = self.entropy(input, tgt)
        return res
