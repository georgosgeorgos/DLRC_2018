import torch
from torch.autograd import Variable

def V(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile, requires_grad=True)

def std(log_var):
	res = torch.exp(0.5 * log_var)
	return res

def reshape(z, l=9, s=3):
	res = z.view(-1, l, s)
	return res