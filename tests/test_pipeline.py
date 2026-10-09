from io import BytesIO
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import main
from app import transcribe as pipeline
from app.transcribe import TUNING, analyze, arrange_fingering, load_audio, track_notes


def tone_sequence(pitches, seconds=0.5, rate=22050):
    parts = [np.zeros(round(rate * 0.15))]
    for midi in pitches:
        t = np.arange(round(rate * seconds)) / rate
        wave = 0.4 * np.sin(2 * np.pi * 440 * 2 ** ((midi - 69) / 12) * t)
        fade = min(200, len(wave) // 2)
        wave[:fade] *= np.linspace(0, 1, fade)
        wave[-fade:] *= np.linspace(1, 0, fade)
        parts.extend((wave, np.zeros(round(rate * 0.12))))
    return np.concatenate(parts).astype(np.float32)


def test_real_pitch_recognition_preserves_pitch_order_and_timing():
    notes = track_notes(tone_sequence([60, 64, 67]), 22050)
    assert [note['midi'] for note in notes] == [60, 64, 67]
    for index, note in enumerate(notes):
        assert abs(note['start'] - (0.15 + 0.62 * index)) < 0.09
        assert abs(note['end'] - (0.65 + 0.62 * index)) < 0.09
        assert 0 <= note['confidence'] <= 1


def test_no_invented_notes_in_silence():
    assert track_notes(np.zeros(22050), 22050) == []


def test_fingering_is_playable_and_keeps_note_order():
    input_notes = [{'midi': pitch, 'start': index, 'end': index + 0.5, 'confidence': 0.9}
                   for index, pitch in enumerate([60, 64, 67, 69, 67, 64])]
    notes, shift = arrange_fingering(input_notes)
    assert shift == 0 and len(notes) == len(input_notes)
    for before, after in zip(input_notes, notes):
        assert after['midi'] == TUNING[after['string'] - 1] + after['fret']
        assert 0 <= after['fret'] <= 15
        assert after['start'] == before['start']


def test_octave_shift_and_out_of_range_are_explicit():
    notes, shift = arrange_fingering([{'midi': 84, 'start': 0, 'end': 1}])
    assert shift == -12 and notes[0]['original_midi'] == 84 and notes[0]['midi'] == 72
    notes, _ = arrange_fingering([{'midi': 150, 'start': 0, 'end': 1}])
    assert notes == []


def test_load_audio_clips_and_rejects_invalid_ranges(tmp_path):
    path = tmp_path / 'tone.wav'
    sf.write(path, tone_sequence([60, 64, 67]), 22050)
    audio, rate, total = load_audio(path, 0.5, 1)
    assert audio.shape == (2, 22050) and rate == 22050 and total > 1.5
    with pytest.raises(ValueError, match='起始时间'):
        load_audio(path, 20, 1)
    path = tmp_path / 'silence.wav'
    sf.write(path, np.zeros(22050), 22050)
    with pytest.raises(ValueError, match='声音'):
        load_audio(path, 0, 1)


def test_solo_pipeline_writes_real_audio(tmp_path):
    path, output = tmp_path / 'tone.wav', tmp_path / 'melody.wav'
    sf.write(path, tone_sequence([60, 64, 67]), 22050)
    stages = []
    result = analyze(path, 0, 30, 'solo', tmp_path / 'models', output,
                     lambda percent, message: stages.append(percent))
    assert [note['midi'] for note in result['notes']] == [60, 64, 67]
    assert result['clip_start'] == 0 and result['mode'] == 'solo'
    assert sf.info(output).duration == pytest.approx(result['duration'], abs=0.002)
    assert stages == sorted(stages)


def test_full_song_over_90_seconds_keeps_boundaries_silence_and_tail(tmp_path, monkeypatch):
    # Short processing blocks exercise many joins without a slow full-song model run.
    monkeypatch.setattr(pipeline, 'CHUNK_SECONDS', 2)
    rate = 22050
    audio = np.zeros(96 * rate, dtype=np.float32)
    for pitch, start, end in [(60, 0.2, 0.8), (64, 1.7, 2.4), (67, 94.5, 95.5)]:
        first, last = round(start * rate), round(end * rate)
        t = np.arange(last - first) / rate
        audio[first:last] = 0.4 * np.sin(2 * np.pi * 440 * 2 ** ((pitch - 69) / 12) * t)
    path, output = tmp_path / 'long.wav', tmp_path / 'full.wav'
    sf.write(path, audio, rate)
    progress = []
    result = analyze(path, 0, 0, 'solo', tmp_path / 'models', output,
                     lambda percent, message: progress.append(percent))
    assert result['full_song'] is True and result['duration'] == 96
    assert result['chunks'] == 48
    assert [note['midi'] for note in result['notes']] == [60, 64, 67]
    assert result['notes'][1]['start'] < 2 < result['notes'][1]['end']
    assert result['notes'][-1]['start'] > 94
    decoded, output_rate = sf.read(output)
    assert len(decoded) == len(audio) and output_rate == rate
    assert np.max(np.abs(decoded[10 * rate:90 * rate])) == 0
    assert progress == sorted(progress)


def test_full_song_duration_guard_does_not_silently_truncate(tmp_path, monkeypatch):
    path = tmp_path / 'too-long.wav'
    sf.write(path, tone_sequence([60, 64, 67]), 22050)
    monkeypatch.setattr(pipeline, 'MAX_SONG_SECONDS', 1)
    with pytest.raises(ValueError, match='最长'):
        analyze(path, 0, 0, 'solo', tmp_path / 'models', tmp_path / 'out.wav', lambda *args: None)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    main.JOBS.clear()
    with TestClient(main.app) as api:
        yield api


def test_api_input_rejection_and_path_protection(client):
    assert client.get('/').status_code == 200
    assert client.get('/api/jobs/missing').status_code == 404
    assert client.get('/api/jobs/missing/melody').status_code == 404
    assert client.post('/api/transcribe', files={'file': ('x.exe', b'abc')}).status_code == 400
    assert client.post('/api/transcribe', files={'file': ('x.wav', b'')}).status_code == 400
    assert client.post('/api/transcribe', files={'file': ('x.wav', b'abc')}, data={'start': -1}).status_code == 400
    assert client.post('/api/transcribe', files={'file': ('x.wav', b'abc')}, data={'seconds': 100}).status_code == 400
    assert client.post('/api/transcribe', files={'file': ('x.wav', b'abc')}, data={'seconds': 0, 'start': 1}).status_code == 400


def wait_job(client, job_id):
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        job = client.get('/api/jobs/' + job_id).json()
        if job['status'] in ('done', 'error'):
            return job
        time.sleep(0.05)
    pytest.fail('Async transcription did not finish within 60 seconds')


@pytest.mark.parametrize('seconds', [30, 0])
def test_api_upload_to_real_transcription_and_result_audio(client, seconds):
    buffer = BytesIO()
    sf.write(buffer, tone_sequence([60, 64, 67]), 22050, format='WAV')
    response = client.post('/api/transcribe', files={'file': ('../../tone.wav', buffer.getvalue())},
                           data={'mode': 'solo', 'seconds': seconds})
    assert response.status_code == 202
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'done', job
    assert [note['midi'] for note in job['result']['notes']] == [60, 64, 67]
    assert job['result']['full_song'] == (seconds == 0)
    audio = client.get('/api/jobs/' + job['id'] + '/melody')
    assert audio.status_code == 200 and audio.headers['content-type'] == 'audio/wav'
    assert sf.info(BytesIO(audio.content)).duration > 1


def test_worker_failure_is_reported_without_leaking_details(client, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('private path and internal exception')
    monkeypatch.setattr(main, 'analyze', fail)
    response = client.post('/api/transcribe', files={'file': ('x.wav', b'abc')})
    job = wait_job(client, response.json()['id'])
    assert job['status'] == 'error'
    assert 'private path' not in job['message']
