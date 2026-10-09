from io import BytesIO
from xml.etree import ElementTree as ET

import mido
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.fingering import arrange_polyphonic
from app.polyphonic import track_polyphonic
from app.score_export import export_midi, export_musicxml, quantize_score
from app.transcribe import merge_chunk_notes, analyze
from app import transcribe as pipeline
from app import main
from app.main import app


def chords():
    rate = 22050
    audio = np.zeros(5 * rate, dtype=np.float32)
    expected = []
    for start, pitches in [(0.3, [48, 52, 55]), (1.7, [50, 54, 57]), (3.1, [48, 52, 55])]:
        time = np.arange(round(rate * 1.1)) / rate
        for pitch in pitches:
            frequency = 440 * 2 ** ((pitch - 69) / 12)
            wave = sum(np.sin(2 * np.pi * frequency * harmonic * time) / harmonic ** 1.6 for harmonic in range(1, 9))
            first = round(start * rate)
            audio[first:first + len(time)] += (wave * np.exp(-time * 2.8) * 0.12).astype(np.float32)
            expected.append((start, pitch))
    return audio, rate, expected


def test_neural_model_recognizes_three_chords_and_repeated_plucks(tmp_path, monkeypatch):
    audio, rate, expected = chords()
    notes = track_polyphonic(audio, rate)
    assert len(notes) == len(expected)
    for start, pitch in expected:
        assert any(n['midi'] == pitch and abs(n['start'] - start) < 0.06 for n in notes)
    path, output = tmp_path / 'chords.wav', tmp_path / 'out.wav'
    sf.write(path, audio, rate)
    monkeypatch.setattr(pipeline, 'CHUNK_SECONDS', 2)
    result = analyze(path, 0, 0, 'guitar_solo', tmp_path / 'models', output, lambda *args: None)
    assert result['polyphonic'] and result['engine'] == 'basic-pitch-onnx'
    assert len(result['notes']) == 9 and result['octave_shift'] == 0
    assert result['duration'] == 5
    assert result['chunks'] == 3
    assert any(n['start'] < 2 < n['end'] for n in result['notes'])
    assert any(n['start'] < 4 < n['end'] for n in result['notes'])
    assert sf.info(output).frames == len(audio)


def test_silent_input_has_no_polyphonic_notes():
    assert track_polyphonic(np.zeros(22050), 22050) == []


def test_fingering_preserves_pitches_and_avoids_ringing_string_conflicts():
    notes = [{'midi': pitch, 'start': start, 'end': end, 'confidence': .9}
             for pitch, start, end in [(48, 0, 1), (52, 0, 1), (55, 0, 1), (57, .5, 1.5), (48, 1.2, 2)]]
    arranged, shift = arrange_polyphonic(notes)
    assert len(arranged) == len(notes) and shift == 0
    for a in arranged:
        for b in arranged:
            if a is b or a['start'] >= b['end'] or b['start'] >= a['end']: continue
            assert a['string'] != b['string']
            if a['fret'] and b['fret']: assert abs(a['fret'] - b['fret']) <= 4
    impossible = notes + [{'midi': 100, 'start': 0, 'end': 1, 'confidence': .8}]
    assert len(arrange_polyphonic(impossible)[0]) == len(notes)


def test_chunk_join_preserves_repeated_plucks_and_continues_long_notes():
    notes = [dict(midi=48, start=29, end=30, confidence=.8, event_start=29, _cut_right=True),
             dict(midi=48, start=30, end=30.7, confidence=.7, event_start=29.53,
                  _cut_left=True, _context_start=True),
             dict(midi=48, start=30.75, end=31.2, confidence=.8, event_start=30.75),
             dict(midi=52, start=29.8, end=30, confidence=.8, event_start=29.8, _cut_right=True),
             dict(midi=52, start=30, end=31, confidence=.8, event_start=29.81, _cut_left=True)]
    merged = merge_chunk_notes(notes, polyphonic=True)
    assert len(merged) == 3
    assert [(n['start'], n['end']) for n in merged if n['midi'] == 48] == [(29, 30.7), (30.75, 31.2)]
    assert all('event_start' not in note and '_cut_right' not in note for note in merged)
    # A fresh repeated onset exactly at the boundary is not a continuation.
    fresh = [dict(midi=48, start=29, end=30, confidence=.8, event_start=29, _cut_right=True),
             dict(midi=48, start=30, end=31, confidence=.8, event_start=30)]
    assert len(merge_chunk_notes(fresh, polyphonic=True)) == 2


def score():
    return {'title': 'A < B & guitar', 'duration': 5, 'bpm': 120, 'meter': '4/4', 'notes': [
        {'midi': 48, 'string': 6, 'fret': 8, 'start': .5, 'end': 2.5, 'confidence': .8},
        {'midi': 52, 'string': 5, 'fret': 7, 'start': .5, 'end': 1.5, 'confidence': .8},
        {'midi': 48, 'string': 6, 'fret': 8, 'start': 2.5, 'end': 3, 'confidence': .8}]}


def test_midi_preserves_simultaneous_onsets_repeated_notes_and_measured_duration():
    midi = mido.MidiFile(file=BytesIO(export_midi(score())))
    assert midi.length == pytest.approx(5, abs=.002)
    elapsed, attacks = 0, []
    for message in midi:
        elapsed += message.time
        if message.type == 'note_on': attacks.append((message.note, elapsed, message.channel))
    assert attacks == [(48, .5, 5), (52, .5, 4), (48, 2.5, 5)]


@pytest.mark.parametrize('meter,bar_ticks', [('4/4', 1920), ('3/4', 1440), ('6/8', 1440)])
def test_musicxml_balanced_voices_ties_and_escaped_title(meter, bar_ticks):
    data = score(); data['meter'] = meter
    root = ET.fromstring(export_musicxml(data))
    assert root.findtext('work/work-title') == data['title']
    assert root.findtext('.//clef/sign') == 'TAB'
    assert len(root.findall('.//staff-tuning')) == 6
    for bar in root.findall('.//measure'):
        for voice in range(1, 7):
            assert sum(int(n.findtext('duration')) for n in bar.findall('note') if n.findtext('voice') == str(voice)) == bar_ticks
    ties = root.findall('.//tie')
    assert any(t.get('type') == 'start' for t in ties)
    assert sum(t.get('type') == 'start' for t in ties) == sum(t.get('type') == 'stop' for t in ties)


def test_export_api_uses_edited_score_and_rejects_invalid_overlapping_notes():
    with TestClient(app) as client:
        data = score(); data['notes'][0].update(midi=49, fret=9)
        response = client.post('/api/export/musicxml', json=data)
        assert response.status_code == 200
        technical = ET.fromstring(response.content).findall('.//technical')
        assert any(n.findtext('string') == '6' and n.findtext('fret') == '9' for n in technical)
        assert client.post('/api/export/midi', json=data).status_code == 200
        data['notes'][2]['start'] = 2.4
        assert client.post('/api/export/midi', json=data).status_code == 400
        data = score(); data['notes'][0]['confidence'] = 'bad'
        assert client.post('/api/export/midi', json=data).status_code == 400
        assert client.post('/api/export/gp5', json=score()).status_code == 400


def test_quantization_refines_fast_plucks_instead_of_dropping_them():
    data = score()
    data['notes'] = [{'midi': 48, 'string': 6, 'fret': 8, 'start': .5, 'end': .52},
                     {'midi': 48, 'string': 6, 'fret': 8, 'start': .53, 'end': .6}]
    notes, *_ = quantize_score(data)
    assert len(notes) == 2 and notes[0]['tick_end'] <= notes[1]['tick_start']
    data['notes'][0]['end'] = .501
    data['notes'][1].update(start=.502, end=.51)
    with pytest.raises(ValueError, match='间隔过短'):
        quantize_score(data)


def test_completed_job_survives_local_server_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(main, 'DATA', tmp_path)
    monkeypatch.setattr(main, 'JOBS', {})
    job_id = 'a' * 32
    (tmp_path / job_id).mkdir()
    expected = {'id': job_id, 'created': 1234., 'status': 'done', 'result': score()}
    main.JOBS[job_id] = expected
    main.save_job(job_id)
    main.JOBS.clear()
    main.restore_jobs()
    assert main.JOBS[job_id] == expected
