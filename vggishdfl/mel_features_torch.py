"""
Port of mel_features.py from numpy to native torch, so
gradients can be passed through the preprocessing operation
"""
import numpy as np
import math
import torch
from torch import Tensor, nn
from typing import *

def periodic_hann(window_length, device=None):
    """Calculate a "periodic" Hann window.

    The classic Hann window is defined as a raised cosine that starts and
    ends on zero, and where every value appears twice, except the middle
    point for an odd-length window.  Matlab calls this a "symmetric" window
    and np.hanning() returns it.  However, for Fourier analysis, this
    actually represents just over one cycle of a period N-1 cosine, and
    thus is not compactly expressed on a length-N Fourier basis.  Instead,
    it's better to use a raised cosine that ends just before the final
    zero value - i.e. a complete cycle of a period-N cosine.  Matlab
    calls this a "periodic" window. This routine calculates it.

    Args:
        window_length: The number of points in the returned window.

    Returns:
        A 1D np.array containing the periodic hann window.
    """
    return 0.5 - (0.5 * torch.cos(2 * math.pi / window_length *
                                torch.arange(window_length, device=device, dtype=float))).float()

def stft_magnitude(signal: Tensor, fft_length: int,
                    hop_length=None,
                    window_length=None,
                    device=None):
    """Calculate the short-time Fourier transform magnitude -- batched.

    Args:
        signal: 2D tensor of the input time-domain signal, of shape (B, T).
        fft_length: Size of the FFT to apply.
        hop_length: Advance (in samples) between each frame passed to FFT.
        window_length: Length of each block of samples to pass to FFT.

    Returns:
        3D np.array where each row contains the magnitudes of the fft_length/2+1
        unique values of the FFT for the corresponding frame of input samples.
    """
    if window_length is None:
        window_length = fft_length

    window = periodic_hann(window_length, device)
    stft = torch.stft(signal, n_fft=fft_length, 
        win_length=window_length, hop_length=hop_length, return_complex=True,
        window=window)
    return stft.absolute()

# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0

def hertz_to_mel(frequencies_hertz):
    """Convert frequencies to mel scale using HTK formula.

    Args:
        frequencies_hertz: Scalar or np.array of frequencies in hertz.

    Returns:
        Object of same size as frequencies_hertz containing corresponding values
        on the mel scale.
    """
    return _MEL_HIGH_FREQUENCY_Q * np.log(
        1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))

def spectrogram_to_mel_matrix(num_mel_bins=20,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0) -> Tensor:
    """Return a matrix that can post-multiply spectrogram rows to make mel.

    Returns a np.array matrix A that can be used to post-multiply a matrix S of
    spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
    "mel spectrogram" M of frames x num_mel_bins.  M = S A.

    The classic HTK algorithm exploits the complementarity of adjacent mel bands
    to multiply each FFT bin by only one mel weight, then add it, with positive
    and negative signs, to the two adjacent mel bands to which that bin
    contributes.  Here, by expressing this operation as a matrix multiply, we go
    from num_fft multiplies per frame (plus around 2*num_fft adds) to around
    num_fft^2 multiplies and adds.  However, because these are all presumably
    accomplished in a single call to np.dot(), it's not clear which approach is
    faster in Python.  The matrix multiplication has the attraction of being more
    general and flexible, and much easier to read.

    Args:
        num_mel_bins: How many bands in the resulting mel spectrum.  This is
        the number of columns in the output matrix.
        num_spectrogram_bins: How many bins there are in the source spectrogram
        data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
        only contains the nonredundant FFT bins.
        audio_sample_rate: Samples per second of the audio at the input to the
        spectrogram. We need this to figure out the actual frequencies for
        each spectrogram bin, which dictates how they are mapped into mel.
        lower_edge_hertz: Lower bound on the frequencies to be included in the mel
        spectrum.  This corresponds to the lower edge of the lowest triangular
        band.
        upper_edge_hertz: The desired top edge of the highest frequency band.

    Returns:
        An np.array with shape (num_spectrogram_bins, num_mel_bins).

    Raises:
        ValueError: if frequency edges are incorrectly ordered or out of range.
    """
    nyquist_hertz = audio_sample_rate / 2.
    if lower_edge_hertz < 0.0:
        raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
    if lower_edge_hertz >= upper_edge_hertz:
        raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                        (lower_edge_hertz, upper_edge_hertz))
    if upper_edge_hertz > nyquist_hertz:
        raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                        (upper_edge_hertz, nyquist_hertz))
    spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
    spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
    # The i'th mel band (starting from i=1) has center frequency
    # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
    # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
    # the band_edges_mel arrays.
    band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                                hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
    # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
    # of spectrogram values.
    mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
    for i in range(num_mel_bins):
        lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
        # Calculate lower and upper slopes for every spectrogram bin.
        # Line segments are linear in the *mel* domain, not hertz.
        lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                    (center_mel - lower_edge_mel))
        upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                    (upper_edge_mel - center_mel))
        # .. then intersect them with each other and zero.
        mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                            upper_slope))
    # HTK excludes the spectrogram DC bin; make sure it always gets a zero
    # coefficient.
    mel_weights_matrix[0, :] = 0.0

    mel_weights_matrix = torch.tensor(mel_weights_matrix).float()
    return mel_weights_matrix

class LogMelSpectrogram(nn.Module):
    def __init__(self, audio_sample_rate=8000,
                        log_offset=0.0,
                        window_length_secs=0.025,
                        hop_length_secs=0.010, **kwargs):
        super().__init__()

        self.window_length_samples = int(round(audio_sample_rate * window_length_secs))
        self.hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
        self.fft_length = 2 ** int(np.ceil(np.log(self.window_length_samples) / np.log(2.0)))

        self.audio_sample_rate = audio_sample_rate
        self.log_offset = log_offset
        self.window_length_secs = window_length_secs
        self.hop_length_secs = hop_length_secs

        mel_matrix = spectrogram_to_mel_matrix(
            num_spectrogram_bins=int(self.fft_length // 2 + 1),
            audio_sample_rate=audio_sample_rate, **kwargs
        )
        self.mel_matrix = nn.Parameter(mel_matrix, requires_grad=False)

    def forward(self, data: Tensor) -> Tensor:
        spec = stft_magnitude(data, fft_length=self.fft_length,
            window_length=self.window_length_samples,
            hop_length=self.hop_length_samples,
            device=data.device)
        spec = spec.transpose(-1, -2)
        mel = torch.matmul(spec, self.mel_matrix)
        return torch.log(mel + self.log_offset)

if __name__ == '__main__':
    audio = torch.randn(8, 16000)
    lms = LogMelSpectrogram(16000, log_offset=0.01)
    spec = lms(audio)
    print(spec.shape)