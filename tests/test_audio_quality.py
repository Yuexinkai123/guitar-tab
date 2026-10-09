import numpy as np
import pytest
import soundfile as sf

from app.audio_quality import gate_guitar, write_preview
from app import transcribe as pipeline
from app.polyphonic import track_polyphonic


def test_residual_noise_is_silent_after_energy_gate_even_if_input_mix_is_loud():
    rate = 22050
    time = np.arange(rate * 2) / rate
    mixture = np.tile(.2 * np.sin(2 * np.pi * 220 * time), (2, 1)).astype(np.float32)
    residue = np.random.default_rng(1).normal(0, .00005, len(time)).astype(np.float32)
    gated, support = gate_guitar(residue, rate, mixture, rate)
    assert not np.any(gated) and not np.any(support)
    assert track_polyphonic(residue, rate) == []


def test_gate_retains_audible_quiet_guitar_and_time_gaps():
    rate = 22050
    time = np.arange(rate * 2) / rate
    mixture = np.ones((2, len(time)), dtype=np.float32) * .2
    stem = (.01 * np.sin(2 * np.pi * 164.814 * time)).astype(np.float32)
    stem[rate:] = 0
    gated, support = gate_guitar(stem, rate, mixture, rate)
    assert np.mean(support[:rate]) == 1 and np.mean(support[rate:]) == 0
    assert np.max(np.abs(gated[rate:])) == 0
    assert np.sqrt(np.mean(gated[:rate] ** 2)) > .006


def test_pipeline_refuses_to_make_a_score_from_silent_guitar_residue(tmp_path, monkeypatch):
    rate = 22050
    time = np.arange(rate * 2) / rate
    path = tmp_path / 'mix.wav'
    sf.write(path, .2 * np.sin(2 * np.pi * 220 * time), rate)
    monkeypatch.setattr(pipeline, 'separate_stem', lambda *args: (np.ones(len(time), dtype=np.float32) * .00005, rate))
    with pytest.raises(ValueError, match='没有可靠提取到吉他'):
        pipeline.analyze(path, 0, 30, 'guitar', tmp_path, tmp_path / 'melody.wav', lambda *args: None)


def test_preview_increases_audible_stem_volume_but_does_not_amplify_noise(tmp_path):
    rate = 22050
    time = np.arange(rate) / rate
    path = tmp_path / 'melody.wav'
    original = .01 * np.sin(2 * np.pi * 220 * time)
    sf.write(path, original, rate)
    quality = write_preview(path)
    preview, _ = sf.read(tmp_path / 'preview.wav')
    assert np.sqrt(np.mean(preview ** 2)) > np.sqrt(np.mean(original ** 2)) * 8
    assert np.max(np.abs(preview)) <= .95
    assert quality['preview_gain_db'] > 18
    sf.write(path, original * .001, rate)
    quality = write_preview(path)
    assert quality['preview_gain_db'] == 0
