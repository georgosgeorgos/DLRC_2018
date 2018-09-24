import torch
from torch.autograd import Variable

def V(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=True)

def std(log_var):
    res = torch.exp(0.5 * log_var)
    return res

def reshape(z, l=9, s=2):
    res = torch.zeros((z.size()[0], l, s))
    index = 0
    for j in range(s):
        res[:, :, j] = z[:,(index*l):((index+1)*l)]
        index +=1
    return res

def expand(x, s=2):
    n, m = x.size()
    x_expanded = torch.zeros(n, m, 2)
    for j in range(s):
        x_expanded[:, :, j] = x
    return x_expanded

