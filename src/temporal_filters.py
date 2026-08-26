"""
Learnable temporal filter layers for EEG deep-learning pipelines.

Implements 1D/2D sinc and 1D/2D wavelet band-pass filters as differentiable
PyTorch layers. Instead of hand-tuning a fixed filter bank, the low-frequency
and scale (bandwidth) of each filter are learned end-to-end alongside the
rest of the network — intended as drop-in preprocessing layers for
EEGNet / SpatialNet / FBCSP-style architectures.

Cleaned up from the original research notebook: removed an unused
TensorFlow/Keras import block left over from an earlier experiment,
added docstrings, and made every subclass consistently use its locally
reshaped `_scale` tensor (a couple of the Hilbert/complex variants were
reading `self._scale` — the un-reshaped 1D buffer — directly; PyTorch's
broadcasting rules happen to make this numerically equivalent, but it was
inconsistent with the other layers and confusing to read).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalPad(nn.Module):
    """
    'Same'-style padding for 1D/2D temporal convolutions, with optional
    extra padding for a Hilbert transform applied after filtering (so the
    transform doesn't produce edge artefacts).

    NOTE: this class was referenced by the original notebook but its body
    lived outside the fetched cell — verify it matches the version used to
    produce your reported results before relying on this file as a full
    replacement.
    """

    def __init__(self, padding, dim, kernel_size, padding_mode="zeros", hilbert=False):
        super().__init__()
        self.padding = padding
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.hilbert = hilbert
        self.padding_hilbert = kernel_size // 2 if hilbert else 0

    def forward(self, x):
        pad_amount = self.kernel_size // 2
        total_pad = pad_amount + self.padding_hilbert
        if self.dim == "1d":
            return F.pad(x, (total_pad, total_pad), mode=self.padding_mode.replace("zeros", "constant"))
        return F.pad(x, (total_pad, total_pad), mode=self.padding_mode.replace("zeros", "constant"))


class HilbertLayer(nn.Module):
    """Analytic-signal (Hilbert transform) layer used by the *Hilbert filter variants."""

    def forward(self, x):
        n = x.shape[-1]
        xf = torch.fft.fft(x, dim=-1)
        h = torch.zeros(n, device=x.device, dtype=x.dtype)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1 : n // 2] = 2
        else:
            h[0] = 1
            h[1 : (n + 1) // 2] = 2
        return torch.fft.ifft(xf * h, dim=-1)


class TemporalFilter(nn.Module):
    """
    Base class holding the learnable frequency parameters shared by every
    filter variant below.

    Each filter is parameterized by a low frequency (`fmin`) and, for the
    sinc variants, a bandwidth, or for the wavelet variants, a centre
    frequency + bandwidth (scale). Any of `fmin`, `bandwidth`, or `freq`
    can be fixed (pass a value) or left learnable (pass `None`, the
    default), in which case it's registered as an `nn.Parameter` and
    randomly initialised from `seed`.
    """

    def __init__(
        self,
        n_channels,
        kernel_size,
        srate,
        fmin=None,
        freq=10,
        bandwidth=30,
        margin_bandwidth=25,
        fmin_variety=12,
        margin_fmin=4,
        seed=None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.srate = srate
        self.fmin = fmin
        self.fmin_variety = fmin_variety
        self.margin_fmin = margin_fmin
        self.margin_bandwidth = margin_bandwidth
        self.bandwidth = bandwidth
        self.freq = freq

        if self.kernel_size % 2 == 0:
            self.register_buffer(
                "_scale", torch.arange(-self.kernel_size // 2, self.kernel_size // 2 + 1) / self.srate
            )
        else:
            self.register_buffer(
                "_scale", torch.arange(-self.kernel_size // 2 + 1, self.kernel_size // 2 + 1) / self.srate
            )

        if seed is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())

        if self.bandwidth is None:
            self.coef_bandwidth = nn.Parameter(self._create_parameters_bandwidth(self.n_channels, seed))
        else:
            bandwidth = self._as_channel_tensor(bandwidth)
            self.register_buffer("_bandwidth", bandwidth)

        if self.fmin is None:
            self.coef_fmin = nn.Parameter(self._create_parameters_fmin(self.n_channels, seed))
        else:
            fmin = self._as_channel_tensor(fmin)
            self.register_buffer("_fmin", fmin)

        if self.freq is not None:
            freq = self._as_channel_tensor(freq)
            self.register_buffer("_freq", freq)

    def _as_channel_tensor(self, value):
        """Broadcast a scalar/1-element value to one entry per channel."""
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32).reshape((1,))
        assert value.shape[0] in (1, self.n_channels)
        if value.shape[0] != self.n_channels:
            value = value.repeat(self.n_channels)
        return value

    def _create_parameters_bandwidth(self, n_coef, seed):
        generator = torch.Generator()
        generator.manual_seed(seed + 1)
        return torch.rand(size=(n_coef,), generator=generator) * self.margin_bandwidth

    def _create_parameters_fmin(self, n_coef, seed):
        generator = torch.Generator()
        generator.manual_seed(seed + 1)
        return torch.rand(size=(n_coef,), generator=generator) * self.fmin_variety + self.margin_fmin

    def _create_frequencies(self):
        bandwidth = self.coef_bandwidth if self.bandwidth is None else self._bandwidth
        fmin = self.coef_fmin if self.fmin is None else self._fmin
        freq = self._freq if self.freq is not None else fmin + bandwidth / 2
        freq_low = fmin
        freq_high = fmin + bandwidth
        return bandwidth, freq_low, freq_high, freq


class SincLayer1d(TemporalFilter):
    """Learnable band-pass sinc filter over a (channels, time) signal."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="1d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)
        self.register_buffer("_hamming_window", torch.hamming_window(kernel_size).reshape((1, 1, -1)))

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1, 1, -1))
        freq_low, freq_high = freq_low.reshape((-1, 1, 1)), freq_high.reshape((-1, 1, 1))
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        return self._hamming_window * 2 * (filt_high - filt_low) / self.srate

    def forward(self, x):
        x = self.pad(x)
        _, freq_low, freq_high, _ = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-2]
        return F.conv1d(x, filt, groups=self.in_channels, padding="valid")


class SincLayer2d(TemporalFilter):
    """2D counterpart of `SincLayer1d` (e.g. for spectrogram-like inputs)."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="2d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)
        self.register_buffer("_hamming_window", torch.hamming_window(kernel_size).reshape((1, 1, 1, -1)))

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1, 1, 1, -1))
        freq_low, freq_high = freq_low.reshape((-1, 1, 1, 1)), freq_high.reshape((-1, 1, 1, 1))
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        return self._hamming_window * 2 * (filt_high - filt_low) / self.srate

    def forward(self, x):
        x = self.pad(x)
        _, freq_low, freq_high, _ = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-3]
        return F.conv2d(x, filt, groups=self.in_channels, padding="valid")


class SincHilbertLayer1d(TemporalFilter):
    """`SincLayer1d` followed by an analytic-signal (envelope) transform."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="1d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=True)
        self.register_buffer("_hamming_window", torch.hamming_window(kernel_size).reshape((1, 1, -1)))
        self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1, 1, -1))
        freq_low, freq_high = freq_low.reshape((-1, 1, 1)), freq_high.reshape((-1, 1, 1))
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        return self._hamming_window * 2 * (filt_high - filt_low) / self.srate

    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        _, freq_low, freq_high, _ = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-2]
        x = F.conv1d(x, filt, groups=self.in_channels, padding="valid")
        if not return_filtered:
            x = torch.abs(self.hilbert(x))
        return x[..., self.pad.padding_hilbert:-self.pad.padding_hilbert]


class SincHilbertLayer2d(TemporalFilter):
    """2D counterpart of `SincHilbertLayer1d`."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="2d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=True)
        self.register_buffer("_hamming_window", torch.hamming_window(kernel_size).reshape((1, 1, -1)))
        self.hilbert = HilbertLayer()

    def _create_filters(self, freq_low, freq_high):
        _scale = self._scale.reshape((1, 1, 1, -1))
        freq_low, freq_high = freq_low.reshape((-1, 1, 1, 1)), freq_high.reshape((-1, 1, 1, 1))
        filt_low = freq_low * torch.special.sinc(2 * freq_low * _scale)
        filt_high = freq_high * torch.special.sinc(2 * freq_high * _scale)
        return self._hamming_window * 2 * (filt_high - filt_low) / self.srate

    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        _, freq_low, freq_high, _ = self._create_frequencies()
        filt = self._create_filters(freq_low, freq_high)
        assert self.in_channels == x.shape[-3]
        x = F.conv2d(x, filt, groups=self.in_channels, padding="valid")
        if not return_filtered:
            x = torch.abs(self.hilbert(x))
        return x[..., self.pad.padding_hilbert:-self.pad.padding_hilbert]


class WaveletLayer1d(TemporalFilter):
    """Learnable real Morlet-style wavelet filter (1D)."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="1d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)

    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1, 1, -1))
        freq, bandwidth = freq.reshape((-1, 1, 1)), bandwidth.reshape((-1, 1, 1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi) ** 2
        filt = (2 * math.pi * sigma2) ** (-1 / 2) / (self.srate / 2)
        filt = filt * torch.cos(2 * math.pi * freq * _scale)
        filt = filt * torch.exp(-(_scale**2) / (2 * sigma2))
        return filt

    def forward(self, x):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-2]
        return F.conv1d(x, filt, groups=self.in_channels, padding="valid")


class WaveletLayer2d(TemporalFilter):
    """2D counterpart of `WaveletLayer1d`."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="2d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)

    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1, 1, 1, -1))
        freq, bandwidth = freq.reshape((-1, 1, 1, 1)), bandwidth.reshape((-1, 1, 1, 1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi) ** 2
        filt = (2 * math.pi * sigma2) ** (-1 / 2) / (self.srate / 2)
        filt = filt * torch.cos(2 * math.pi * freq * _scale)
        filt = filt * torch.exp(-(_scale**2) / (2 * sigma2))
        return filt

    def forward(self, x):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-3]
        return F.conv2d(x, filt, groups=self.in_channels, padding="valid")


class ComplexWaveletLayer1d(TemporalFilter):
    """Complex (analytic) Morlet wavelet filter (1D) — gives an instantaneous-envelope output directly."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="1d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)

    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1, 1, -1))
        freq, bandwidth = freq.reshape((-1, 1, 1)), bandwidth.reshape((-1, 1, 1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi) ** 2
        filt = (2 * math.pi * sigma2) ** (-1 / 2) / (self.srate / 2)
        filt = filt * (torch.exp(1j * 2 * math.pi * freq * _scale) - torch.exp(-0.5 * (2 * math.pi * freq) ** 2))
        filt = filt * torch.exp(-(_scale**2) / (2 * sigma2))
        return filt

    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-2]
        if return_filtered:
            return F.conv1d(x, filt.real, groups=self.in_channels, padding="valid")
        x = x.to(torch.complex64)
        x = F.conv1d(x, filt, groups=self.in_channels, padding="valid")
        return torch.abs(x)


class ComplexWaveletLayer2d(TemporalFilter):
    """2D counterpart of `ComplexWaveletLayer1d`."""

    def __init__(self, in_channels, out_channels, kernel_size, srate, fmin_init, fmax_init,
                 freq=None, bandwidth=None, padding_mode="zeros", seed=None):
        super().__init__(out_channels, kernel_size, srate, fmin_init, fmax_init, freq, bandwidth, seed=seed)
        self.in_channels = in_channels
        self.pad = TemporalPad(padding="same", dim="2d", kernel_size=kernel_size, padding_mode=padding_mode, hilbert=False)

    def _create_filters(self, freq, bandwidth):
        _scale = self._scale.reshape((1, 1, 1, -1))
        freq, bandwidth = freq.reshape((-1, 1, 1, 1)), bandwidth.reshape((-1, 1, 1, 1))
        sigma2 = (2 * math.log(2)) / (bandwidth * math.pi) ** 2
        filt = (2 * math.pi * sigma2) ** (-1 / 2) / (self.srate / 2)
        filt = filt * (torch.exp(1j * 2 * math.pi * freq * _scale) - torch.exp(-0.5 * (2 * math.pi * freq) ** 2))
        filt = filt * torch.exp(-(_scale**2) / (2 * sigma2))
        return filt

    def forward(self, x, return_filtered=False):
        x = self.pad(x)
        freq, bandwidth, _, _ = self._create_frequencies()
        filt = self._create_filters(freq, bandwidth)
        assert self.in_channels == x.shape[-3]
        if return_filtered:
            return F.conv2d(x, filt.real, groups=self.in_channels, padding="valid")
        x = x.to(torch.complex64)
        x = F.conv2d(x, filt, groups=self.in_channels, padding="valid")
        return torch.abs(x)
