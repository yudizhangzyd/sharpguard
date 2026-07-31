"""Upstream-exact LIBERO frame preprocessing, with no torch dependency.

This module is deliberately importable with numpy and Pillow alone. Two callers
need it and they cannot share an environment:

  * sharpguard.libero_sim, inside the GPU rollout;
  * experiments/resize_kernel_check.py, which runs in a throwaway venv holding
    tensorflow-cpu so it can check this code against upstream's actual
    implementation. That check has to validate the function that ships, not a
    transcription of it, which is why the resize lives here and not inline in
    libero_sim.

Why any of this exists. The decoder gate returned Task SR ~0 four times. The
reference diff (bolt htrg4uchwi) found that upstream's
experiments/robot/libero/libero_utils.resize_image does three things our
rollout did not:

    img = tf.image.encode_jpeg(img)                       # RLDS stored JPEGs
    img = tf.io.decode_image(img, ..., dtype=tf.uint8)
    img = tf.image.resize(img, size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

and its docstring says why: "To make input images in distribution with respect
to the inputs seen at training time, we follow the same resizing scheme used in
the Octo dataloader, which OpenVLA uses for training."

The obvious shortcut -- Pillow's LANCZOS -- was measured against TF rather than
assumed, and it is not equivalent: bolt q5z79humta reported up to 23/255 LSB of
disagreement on high-frequency content, against a threshold of 4 fixed before
the measurement. So the resize is reimplemented here directly from the
definition of a scaled Lanczos-3 kernel, which is what tf.image.resize's
antialias path computes, and validated numerically instead of trusted.
"""
from __future__ import annotations

import io

import numpy as np

MODES = ("none", "np_lanczos", "pil_lanczos", "tf_upstream")


# ---------------------------------------------------------------------------
# resize
# ---------------------------------------------------------------------------

def _lanczos3(t: np.ndarray) -> np.ndarray:
    """Lanczos-3 kernel: sinc(t) * sinc(t/3), zero beyond |t| >= 3.

    np.sinc already carries the pi factors (np.sinc(x) = sin(pi x)/(pi x)) and
    is defined as 1 at 0, so the removable singularity needs no special case.
    """
    t = np.abs(np.asarray(t, dtype=np.float64))
    out = np.zeros_like(t)
    inside = t < 3.0
    ti = t[inside]
    out[inside] = np.sinc(ti) * np.sinc(ti / 3.0)
    return out


def _weights(in_len: int, out_len: int) -> np.ndarray:
    """Dense (out_len, in_len) resampling matrix for one axis.

    This is the antialias=True convention: when downscaling, the kernel's
    support is stretched by the scale factor so the filter averages over every
    input pixel that maps into an output pixel. Without that stretch (i.e.
    antialias=False) a 256->224 reduction would alias, and aliasing is exactly
    the kind of high-frequency corruption that puts a frame off-distribution.

    Pixel centres are the half-integer convention -- output pixel i covers
    input coordinate (i + 0.5) * scale - 0.5 -- which is what tf.image.resize
    uses. Weights are normalised over in-range taps only, so edge pixels stay
    unbiased rather than being darkened by an implicit zero border.
    """
    scale = in_len / out_len
    kernel_scale = max(scale, 1.0)          # only stretch when downscaling
    centers = (np.arange(out_len) + 0.5) * scale - 0.5
    j = np.arange(in_len)
    # (out_len, in_len) matrix of kernel taps.
    w = _lanczos3((j[None, :] - centers[:, None]) / kernel_scale)
    norms = w.sum(axis=1, keepdims=True)
    # A row can only sum to zero if the kernel misses every input pixel, which
    # cannot happen for a 3-tap-radius kernel over a non-empty axis; guard
    # anyway so a future size choice fails loudly instead of emitting NaNs.
    if np.any(norms == 0):
        raise ValueError(f"degenerate resize weights for {in_len} -> {out_len}")
    return w / norms


def lanczos3_resize(img: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize HxWx3 uint8 to size x size with an antialiased Lanczos-3 kernel.

    Matches tf.image.resize(..., method="lanczos3", antialias=True) followed by
    round/clip/uint8, which is what upstream applies. Note that the rounding
    happens once at the end, in float64, so the two separable passes do not
    accumulate uint8 quantisation error.
    """
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
        raise ValueError(f"expected HxWxC uint8, got shape {arr.shape}")
    x = arr.astype(np.float64)
    wy = _weights(x.shape[0], size)
    wx = _weights(x.shape[1], size)
    x = np.einsum("oi,ijc->ojc", wy, x)     # rows
    x = np.einsum("oj,ijc->ioc", wx, x)     # columns
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# JPEG round-trip
# ---------------------------------------------------------------------------

def jpeg_roundtrip(img: np.ndarray, quality: int = 95,
                   subsampling: int = 0) -> np.ndarray:
    """Encode to JPEG and decode straight back, as upstream does.

    Upstream's comment is "Encode as JPEG, as done in RLDS dataset builder":
    the training frames were lossily compressed, so a pixel-exact simulator
    render is itself slightly off-distribution. tf.image.encode_jpeg defaults
    to quality=95 with chroma_downsampling=True; Pillow's `subsampling`
    argument is the same control, and 0 means 4:4:4 (no downsampling) while 2
    means 4:2:0. The default here is whichever value
    experiments/resize_kernel_check.py measured closest to TF -- it is a
    measured parameter, not a guess, and that script re-checks it.
    """
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).convert("RGB").save(
        buf, format="JPEG", quality=quality, subsampling=subsampling)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# the mode dispatch used by the rollout
# ---------------------------------------------------------------------------

def preprocess(img: np.ndarray, mode: str, size: int = 224) -> np.ndarray:
    """Prepare an already-180-degree-flipped render for the model.

    Modes, and why each exists:
      "none"        -- pass the raw render through. What the four failed gates
                       ran; kept as the factorial's anchor cell.
      "np_lanczos"  -- upstream's three steps, implemented here. This is the
                       mode the gate should run.
      "pil_lanczos" -- the same steps with Pillow's LANCZOS. Retained only so
                       the factorial can show what it costs: bolt q5z79humta
                       measured it up to 23/255 LSB from TF, so it must not be
                       described as reproducing upstream.
      "tf_upstream" -- calls tensorflow directly. Exact by construction, but
                       unusable in the eval environment, where tensorflow
                       clobbers the numpy<2 pin (bolt/setup-openvla.sh:41).
    """
    if mode not in MODES:
        raise ValueError(f"unknown image preproc {mode!r}; expected {MODES}")
    arr = np.asarray(img, dtype=np.uint8)
    if mode == "none":
        return arr
    if mode == "np_lanczos":
        return lanczos3_resize(jpeg_roundtrip(arr), size)
    if mode == "pil_lanczos":
        from PIL import Image
        rt = jpeg_roundtrip(arr)
        return np.asarray(Image.fromarray(rt).resize((size, size),
                                                     Image.LANCZOS),
                          dtype=np.uint8)
    try:
        import tensorflow as tf
    except ImportError as e:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "image preproc 'tf_upstream' needs tensorflow, which this "
            "environment deliberately does not have. Use 'np_lanczos', which "
            "experiments/resize_kernel_check.py validates against it."
        ) from e
    t = tf.image.encode_jpeg(arr)
    t = tf.io.decode_image(t, expand_animations=False, dtype=tf.uint8)
    t = tf.image.resize(t, (size, size), method="lanczos3", antialias=True)
    return tf.cast(tf.clip_by_value(tf.round(t), 0, 255), tf.uint8).numpy()
