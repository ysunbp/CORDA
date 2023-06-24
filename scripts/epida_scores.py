import torch
import numpy as np


EPS = 1e-10
def MI(z,zt):
    C = z.size()[1]
    # actually they are not independent
    P = (z.unsqueeze(2) * zt.unsqueeze(1)).sum(dim=0)
    P = ((P + P.t())/2) / P.sum()
    P[(P<EPS).data] = EPS
    Pi = P.sum(dim=1).view(C,1).expand(C,C)
    Pj = P.sum(dim=0).view(1,C).expand(C,C)
    # revise by 1.0
    return 1.0-(P * (-torch.log(Pi)-torch.log(Pj)+torch.log(P))).sum()

def CEM(z,zt):
    return MI(z,zt)-H(z)

def H(P):
    P[(P<EPS).data] = EPS
    return -(P*torch.log(P)).sum()

def REM(z,zt):
    zt[(zt<EPS).data] = EPS
    return -torch.sum(z*torch.log(zt))

def gradmutualgain(output,one_hot,softmaxed,softmaxed_y,loss_fn=None):
    up = REM(softmaxed.unsqueeze(0),one_hot.unsqueeze(0))
    # make all the less than zero > 0
    down = 1.0+CEM(softmaxed.unsqueeze(0),softmaxed_y.unsqueeze(0))
    return up,down