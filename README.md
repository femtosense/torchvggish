# VGGish
Based on a `torch`-compatible port of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>[1]</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings will be identical.

## Deep Feature Loss
We wrap the VGGish model so that it can be used as a deep feature loss, for use as a content-loss, e.g. when training speech enhancement networks. Deep Feature Losses often outperform simple MSE-based loss functions.

## Usage

```python
import torch
import vggishdfl

device = 'cuda:0'

dfl = vggishdfl.VGGishDFL()
dfl = dfl.to(device)

tgt = torch.randn(8, 16000, device=device)
est = torch.randn(8, 16000, device=device)

loss = dfl(est, tgt)
```

<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

