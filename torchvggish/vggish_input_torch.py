"""
Port of vggish_input.py from numpy to native torch, so
gradients can be passed through the preprocessing operation
"""

from torch.nn import functional as F
import math
from torch import Tensor, nn
import mel_features_torch
import vggish_params
from typing import *

class VGGishPreprocessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logmel = mel_features_torch.LogMelSpectrogram(
            audio_sample_rate=vggish_params.SAMPLE_RATE,
            log_offset=vggish_params.LOG_OFFSET,
            window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
            hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
            num_mel_bins=vggish_params.NUM_MEL_BINS,
            lower_edge_hertz=vggish_params.MEL_MIN_HZ,
            upper_edge_hertz=vggish_params.MEL_MAX_HZ
        )

    def forward(self, data):
        logmels = self.logmel(data)

        # Need to break logmels into nonoverlapping 0.96s segments
        B, T, C = logmels.shape

        n = math.ceil(T/vggish_params.NUM_FRAMES)
        pad_amount = n * vggish_params.NUM_FRAMES - T
        logmels = F.pad(logmels, pad=(0, 0, 0, pad_amount))
        # Expand along the batch dimension
        logmels = logmels.reshape(B*n, 1, vggish_params.NUM_FRAMES, C)

        return logmels

        

