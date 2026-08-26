# temporal-filters

Learnable, parameterized temporal filters (1D/2D wavelet, 1D/2D sinc) designed as drop-in preprocessing layers for EEG deep-learning architectures such as EEGNet, SpatialNet, and FBCSP.

## Overview

Fixed band-pass filter banks are a common first stage in EEG deep-learning pipelines, but their cutoffs are hand-tuned rather than learned. `TemporalFilter` implements wavelet- and sinc-based filters as differentiable layers with a small, interpretable parameter set — a low-frequency parameter for all filter types, plus a scale parameter for the wavelet and sinc variants — so the filter bank can be optimized end-to-end with the rest of the network.

## What's inside

- A universal `TemporalFilter` layer implementation covering all four filter types (wavelet 1D/2D, sinc 1D/2D).
- Example integrations comparing the filters inside EEGNet-, SpatialNet-, and FBCSP-style architectures, to validate that the layer behaves as expected inside a real training loop.

## Repository contents

| Path | Purpose |
|---|---|
| `src/temporal_filters.py` | The filter layers themselves (sinc, wavelet, Hilbert, complex-wavelet — 1D and 2D) |
| `src/reference_models.py` | Reference EEG architectures (EEGNet family, DeepConvNet, ShallowConvNet) used to sanity-check the filters inside real models |
| `EEGML_WithinSession/` | Within-session EEG model experiments using the filters *(carry this folder over from the old repo as-is)* |

## Setup

```bash
pip install -r requirements.txt   # torch + numpy for src/temporal_filters.py
pip install tensorflow            # only needed for src/reference_models.py
```

## Tech stack

Python, PyTorch, NumPy (TensorFlow/Keras additionally for the reference models)
