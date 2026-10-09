"""Spotify Basic Pitch ONNX inference, with the original licensed note decoder."""
import math
from pathlib import Path
import threading

import numpy as np
from scipy.signal import resample_poly

from .vendor.basic_pitch.decoder import output_to_notes_polyphonic, model_frames_to_time

_session = None
_lock = threading.Lock()


def track_polyphonic(audio, rate, report=lambda *args: None):
    import onnxruntime as ort
    global _session
    if not len(audio) or np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)) < .0001:
        return []
    with _lock:
        if _session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            _session = ort.InferenceSession(str(Path(__file__).parent / 'vendor/basic_pitch/nmp.onnx'),
                                           sess_options=options, providers=['CPUExecutionProvider'])
    divisor = math.gcd(rate, 22050)
    y = resample_poly(audio, 22050 // divisor, rate // divisor).astype(np.float32)
    original_length, overlap, window = len(y), 30 * 256, 43844
    padded = np.pad(y, (overlap // 2, 0))
    outputs = {'note': [], 'onset': []}
    positions = range(0, len(padded), window - overlap)
    for index, first in enumerate(positions):
        block = padded[first:first + window]
        block = np.pad(block, (0, window - len(block)))[None, :, None]
        prediction = _session.run(['StatefulPartitionedCall:1', 'StatefulPartitionedCall:2'],
                                  {'serving_default_input_2:0': block})
        for name, value in zip(outputs, prediction):
            outputs[name].append(value[:, 15:-15, :])
        report(65 + round((index + 1) / len(positions) * 25), '识别吉他和弦、分解和弦与重复拨弦')
    count = int(np.floor(original_length * (86 / 22050)))
    activations = {name: np.concatenate(values).reshape(-1, 88)[:count].copy() for name, values in outputs.items()}
    if len(activations['note']) < 2:
        return []
    events = output_to_notes_polyphonic(activations['note'], activations['onset'],
        onset_thresh=0.5, frame_thresh=0.3, min_note_len=7, infer_onsets=True,
        min_freq=80.0, max_freq=850.0, melodia_trick=True)
    times = model_frames_to_time(len(activations['note']) + 1)
    duration = original_length / 22050
    return sorted([{'midi': int(pitch), 'start': round(max(0, float(times[first])), 3),
                    'end': round(min(duration, float(times[last])), 3),
                    'confidence': round(float(confidence), 3)}
                   for first, last, pitch, confidence in events
                   if confidence >= 0.45 and min(duration, times[last]) - max(0, times[first]) >= 0.085],
                  key=lambda note: (note['start'], note['midi']))
