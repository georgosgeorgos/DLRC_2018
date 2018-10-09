import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class lossNegSampling(nn.Module):
    def __init__(self):
        super(lossNegSampling, self).__init__()
        self.m = nn.Sigmoid()
        self.loss = nn.BCELoss() #nn.CrossEntropyLoss()

    def forward(self, pred, lbl):
        #pred = pred.permute(0,2,1)
        #lbl  = lbl.permute(0,2,1)
        lbl   = lbl.cpu().data.numpy()
        #pred_ = th.zeros((pred.size()[0], pred.size()[1]))
        pred  = self.m(pred)
        pred  = pred.cpu().data.numpy()

        pred_ = pred[lbl==1]
        lbl   = lbl.argmax(axis=2)
        pred_ = pred_.reshape(lbl.shape)

        pred_ = Variable(th.from_numpy(pred_), requires_grad=True)
        lbl   = th.from_numpy(lbl).float()
        
        loss = self.loss(pred_, lbl)
        return loss