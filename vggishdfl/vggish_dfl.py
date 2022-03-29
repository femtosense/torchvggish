"""
Deep Feature Loss based on vggish network
"""
import torch
from torch import nn, Tensor, hub
from typing import *
from .vggish import VGG, make_layers, Postprocessor
from . import vggish_params
from .vggish_input_torch import VGGishPreprocessor

MODEL_URLS = {
    'vggish': 'https://github.com/harritaylor/torchvggish/'
              'releases/download/v0.1/vggish-10086976.pth',
    'pca': 'https://github.com/harritaylor/torchvggish/'
           'releases/download/v0.1/vggish_pca_params-970ea276.pth'
}

def relu_save_hook(module, input, output):
    if isinstance(module, nn.ReLU):
        module.saved_act = output

class VGGish(nn.Module):
    def __init__(self, urls=None, pretrained=True, postprocess=True,
        progress=True, hook_relus=True):
        super().__init__()

        self.vgg = VGG(make_layers())

        if urls is None:
            urls = MODEL_URLS

        if pretrained:
            state_dict = hub.load_state_dict_from_url(urls['vggish'], progress=progress)
            self.vgg.load_state_dict(state_dict)

        self.postprocess = postprocess
        if self.postprocess:
            self.pproc = Postprocessor()
            if pretrained:
                state_dict = hub.load_state_dict_from_url(urls['pca'], progress=progress)
                # TODO: Convert the state_dict to torch
                state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_EIGEN_VECTORS_NAME], dtype=torch.float
                )
                state_dict[vggish_params.PCA_MEANS_NAME] = torch.as_tensor(
                    state_dict[vggish_params.PCA_MEANS_NAME].reshape(-1, 1), dtype=torch.float
                )

                self.pproc.load_state_dict(state_dict)
        
        self.preprocessor = VGGishPreprocessor()

        if hook_relus:
            self.hook_relus()
            self.hooked = True
        else:
            self.hooked = False

    def hook_relus(self):
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                module.register_forward_hook(relu_save_hook)

    def collect_relu_activations(self) -> List[Tensor]:
        assert self.hooked
        acts = []
        for module in self.modules():
            if isinstance(module, nn.ReLU):
                if hasattr(module, 'saved_act'):
                    acts.append(module.saved_act)
                    module.saved_act = None
        return acts

    def forward(self, data) -> Tuple[Tensor, List[Tensor]]:
        logmels = self.preprocessor(data)
        feats = self.vgg(logmels)
        if self.postprocess:
            feats = self.pproc(feats)
        
        if self.hooked:
            acts = self.collect_relu_activations()
        else:
            acts = []

        if self.postprocess:
            acts.append(feats)

        return feats, acts

def normed_mse(pred: Tensor, targ: Tensor, eps=1e-9):
    num = (pred - targ).pow(2).sum()
    den = targ.pow(2).sum()
    return num / (den + eps)

class VGGishDFL(nn.Module):
    """
    Deep Feature Loss, based on trained VGGish classifier.

    Arguments:
        urls (dict, optional): torch.hub urls to download weights for the trained classifier,
            defaults to harritaylor's torch.hub parameters.
        pretrained (bool, optional): If True, loads pretrained weights from torch.hub.
         Default True.
        postprocess (bool, optional): If True, post-processes to calculate vggish embeddings. Default True.
        progress (bool, optional): If True, displays a progress bar when downloading weights.
            Default True.
        hook_relus (bool, optional): If True, Deep Feature Loss is computed as the mean of
            normalized MSEs for each ReLU activation in the network. Default True.
        no_param_grad (bool, optional): If True, parameters are frozen. Default True.
    """
    def __init__(self, urls=None, pretrained=True, postprocess=True,
        progress=True, hook_relus=True, no_param_grad=True):
        super().__init__()

        self.hooked = hook_relus

        self.vggish = VGGish(urls=urls, pretrained=pretrained, postprocess=postprocess,
            progress=progress, hook_relus=hook_relus)

        if no_param_grad:
            for w in self.parameters():
                w.requires_grad = False


    def forward(self, enh, tgt):

        tgt_feat, tgt_act = self.vggish(tgt)
        enh_feat, enh_act = self.vggish(enh)

        if not self.hooked:
            loss = normed_mse(enh_feat, tgt_feat, axis=[1, 2])

        else:
            loss = 0
            for e, t in zip(enh_act, tgt_act):
                loss += normed_mse(e, t)
            loss = loss / len(tgt_act)

        return loss


