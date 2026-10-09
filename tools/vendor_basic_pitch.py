"""Extract the Apache-2.0 ONNX model and four decoder functions from version 0.4.0."""
import ast
import hashlib
from pathlib import Path
import shutil
import sysconfig

root = Path(__file__).resolve().parent.parent
installed = Path(sysconfig.get_paths()['purelib'])
source = installed / 'basic_pitch'
destination = root / 'app' / 'vendor' / 'basic_pitch'
destination.mkdir(parents=True, exist_ok=True)
text = (source / 'note_creation.py').read_text(encoding='utf-8')
lines = text.splitlines(keepends=True)
names = {'get_infered_onsets', 'constrain_frequency', 'model_frames_to_time', 'output_to_notes_polyphonic'}
header = ''.join(lines[:17]) + '''
# Adapted for Stringnote: extracted the polyphonic decoder only; removed MIDI,
# TensorFlow and sonification dependencies. The inference weights are unchanged.
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import librosa
import scipy.signal
MIDI_OFFSET = 21
MAX_FREQ_IDX = 87
AUDIO_SAMPLE_RATE = 22050
FFT_HOP = 256
ANNOT_N_FRAMES = 172
AUDIO_N_SAMPLES = 43844

'''
functions = [ ''.join(lines[node.lineno - 1:node.end_lineno]) for node in ast.parse(text).body
              if isinstance(node, ast.FunctionDef) and node.name in names ]
(destination / 'decoder.py').write_text(header + '\n\n'.join(functions), encoding='utf-8')
for folder in (destination, destination.parent):
    (folder / '__init__.py').touch()
shutil.copy2(source / 'saved_models/icassp_2022/nmp.onnx', destination / 'nmp.onnx')
shutil.copy2(installed / 'basic_pitch-0.4.0.dist-info/LICENSE', destination / 'LICENSE')
print('ONNX SHA256', hashlib.sha256((destination / 'nmp.onnx').read_bytes()).hexdigest())
