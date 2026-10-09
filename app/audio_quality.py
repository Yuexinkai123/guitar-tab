"""Reject inaudible separation residue before neural pitch inference."""
from pathlib import Path

import numpy as np
import soundfile as sf


def gate_guitar(stem, stem_rate, mixture, mixture_rate):
    """100 ms energy gate against BOTH an absolute floor and the local mix.

    This is a conservative audibility check, not an instrument classifier.
    No gain normalization is applied before deciding whether a stem exists.
    """
    stem = np.asarray(stem, dtype=np.float32)
    step = max(1, round(stem_rate * .1))
    supported = np.zeros(len(stem), dtype=np.float32)
    for first in range(0, len(stem), step):
        last = min(len(stem), first + step)
        ref_first = round(first / stem_rate * mixture_rate)
        ref_last = max(ref_first + 1, round(last / stem_rate * mixture_rate))
        reference = mixture[..., ref_first:ref_last]
        ref_rms = float(np.sqrt(np.mean(reference ** 2))) if reference.size else 0
        rms = float(np.sqrt(np.mean(stem[first:last] ** 2)))
        # Below -60 dBFS or 36 dB below the mix is not a useful source here.
        if rms >= max(.001, ref_rms * 10 ** (-36 / 20)):
            supported[first:last] = 1
    # Short fades avoid introducing hard-edge clicks into the gated signal.
    from scipy.ndimage import uniform_filter1d
    smooth = uniform_filter1d(supported, size=max(1, round(stem_rate * .008)), mode='constant')
    return stem * smooth, supported


def write_preview(path: Path):
    """Normalize useful extracted audio, with a fixed gain and no noise boost."""
    peak, energy, count = 0., 0., 0
    with sf.SoundFile(path) as source:
        for block in source.blocks(blocksize=65536, dtype='float32'):
            peak = max(peak, float(np.max(np.abs(block), initial=0)))
            energy += float(np.sum(block.astype(np.float64) ** 2)); count += len(block)
        rate = source.samplerate
    rms = float(np.sqrt(energy / max(1, count)))
    gain = min(12., .95 / max(peak, 1e-12), .09 / max(rms, 1e-12)) if rms >= .001 else 1.
    preview = path.with_name('preview.wav')
    with sf.SoundFile(path) as source, sf.SoundFile(preview, 'w', samplerate=rate, channels=1, subtype='PCM_16') as target:
        for block in source.blocks(blocksize=65536, dtype='float32'):
            target.write(block * gain)
    return {'rms_dbfs': round(20 * float(np.log10(max(rms, 1e-12))), 1),
            'preview_gain_db': round(20 * float(np.log10(max(gain, 1e-12))), 1)}
