from vggishdfl import VGGishDFL
import torch
import math

if __name__ == '__main__':

    loss_fn = VGGishDFL(hook_relus=True)

    T = 2
    freq = 200
    t = int(T * 16000)

    trg = torch.randn(8, t)
    est = torch.cos(2*math.pi*torch.linspace(0, freq*T, t)).unsqueeze(0)*.5 + trg

    loss = loss_fn(est, trg)
    print(loss)

    est = est - trg
    loss = loss_fn(est, trg, reduce=False)
    print(loss)