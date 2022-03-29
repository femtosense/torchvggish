# VGGish
Forked from a [`torch`-compatible port](https://github.com/harritaylor/torchvggish) of [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset)<sup>[1]</sup>, 
a feature embedding frontend for audio classification models. The weights are ported directly from the tensorflow model, so embeddings will be identical.

## Deep Feature Loss
We wrap the VGGish model so that it can be used as a deep feature loss, for use as a content-loss, e.g. when training speech enhancement networks. Deep Feature Losses often outperform simple MSE-based loss functions.

## Installation
After cloning the repo,
```
cd vggishdfl
pip install .
```
or
```
pip install -e .
```

## Usage

```python
import torch
import vggishdfl

device = 'cuda:0'

dfl = vggishdfl.VGGishDFL()
dfl = dfl.to(device)

tgt = torch.randn(8, 16000, device=device)
est = torch.randn(8, 16000, device=device)

# estimate is first; target is second
loss = dfl(est, tgt)
```

## Per-Layer vs. Output Loss
The most important option when configuring the VGGishDFL is the `hook_relu` boolean.
If `hook_relu` is `True`, the deep feature loss will be computed based on intermediate activations. Each ReLU activation in the VGGish network will be compared between the target and estimate waveforms to compute a per-ReLU MSE. The per-ReLU MSEs are each normalized by the variance of the target activation, and then a total loss is computed by averaging the normalized per-ReLU MSE. 

For a given relu activation, with estimate activation `x_est` and target activation `x_tgt`, the normalized per-ReLU MSE is given as:
```
mse_layer = mean((x_tgt - x_est)**2) / mean(x_tgt**2)
```

If `hook_relu` is `False`, the loss is computed as an MSE from the final output activations of the VGGish network.

<hr>
[1]  S. Hershey et al., ‘CNN Architectures for Large-Scale Audio Classification’,\
    in International Conference on Acoustics, Speech and Signal Processing (ICASSP),2017\
    Available: https://arxiv.org/abs/1609.09430, https://ai.google/research/pubs/pub45611
    

