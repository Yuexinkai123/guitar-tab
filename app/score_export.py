"""Editable guitar score exports: measured-time MIDI and quantized TAB MusicXML."""
from io import BytesIO
import math
from xml.etree import ElementTree as ET

import mido

TUNING = (64, 59, 55, 50, 45, 40)
DIVISIONS = 480
VALUES = [(1920, 'whole', False), (1440, 'half', True), (960, 'half', False),
          (720, 'quarter', True), (480, 'quarter', False), (360, 'eighth', True),
          (240, 'eighth', False), (180, '16th', True), (120, '16th', False),
          (90, '32nd', True), (60, '32nd', False), (45, '64th', True),
          (30, '64th', False), (15, '128th', False)]


def validate_score(score):
    notes = score.get('notes', [])
    bpm = score.get('bpm') or 120
    meter = score.get('meter', '4/4')
    duration = score.get('duration', 0)
    if meter not in ('4/4', '3/4', '6/8') or not isinstance(bpm, (int, float)) or not math.isfinite(bpm) or not 30 <= bpm <= 300:
        raise ValueError('请选择有效的排谱速度（30～300 BPM）和拍号。')
    if not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= 1800:
        raise ValueError('谱的长度需要在 0～30 分钟之间。')
    if not isinstance(notes, list) or not 1 <= len(notes) <= 30000:
        raise ValueError('音符数量无效。')
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError('音符格式无效。')
        start, end = note.get('start'), note.get('end')
        string, fret, pitch = note.get('string'), note.get('fret'), note.get('midi')
        confidence = note.get('confidence', 0.75)
        if (not all(isinstance(value, (float, int)) and math.isfinite(value) for value in (start, end))
                or not 0 <= start < end <= duration + 0.02
                or not all(isinstance(value, int) for value in (string, fret, pitch))
                or not 1 <= string <= 6 or not 0 <= fret <= 15 or pitch != TUNING[string - 1] + fret):
            raise ValueError('音符时间或弦、品位无效。请检查手动修改的音符。')
        if not isinstance(confidence, (float, int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('音符置信度无效。')
    for string in range(1, 7):
        line = sorted((note for note in notes if note['string'] == string), key=lambda n: n['start'])
        if any(a['end'] > b['start'] + 0.001 for a, b in zip(line, line[1:])):
            raise ValueError('同一根弦上存在同时发声的音符，请调整指法。')
    return notes, float(bpm), meter, float(duration)


def export_midi(score):
    notes, bpm, meter, duration = validate_score(score)
    midi = mido.MidiFile(ticks_per_beat=DIVISIONS)
    track = mido.MidiTrack(); midi.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
    numerator, denominator = map(int, meter.split('/'))
    track.append(mido.MetaMessage('time_signature', numerator=numerator, denominator=denominator))
    for channel in range(6):
        track.append(mido.Message('program_change', program=24, channel=channel))
    events = []
    for note in notes:
        channel = note['string'] - 1
        velocity = min(110, max(45, round(note.get('confidence', 0.75) * 110)))
        events.extend([(round(note['start'] * bpm / 60 * DIVISIONS), 1,
                        mido.Message('note_on', note=note['midi'], velocity=velocity, channel=channel)),
                       (round(note['end'] * bpm / 60 * DIVISIONS), 0,
                        mido.Message('note_off', note=note['midi'], velocity=0, channel=channel))])
    previous = 0
    for ticks, _, message in sorted(events, key=lambda event: (event[0], event[1])):
        track.append(message.copy(time=max(0, ticks - previous))); previous = ticks
    track.append(mido.MetaMessage('end_of_track', time=max(0, round(duration * bpm / 60 * DIVISIONS) - previous)))
    buffer = BytesIO(); midi.save(file=buffer); return buffer.getvalue()


def quantize_score(score):
    notes, bpm, meter, duration = validate_score(score)
    # Refine the grid for fast repeated plucks rather than dropping attacks
    # that would otherwise land on the same sixteenth-note position.
    grid = 120
    for candidate in (120, 60, 30, 15):
        keys = [(note['string'], round(note['start'] * bpm / 60 * DIVISIONS / candidate)) for note in notes]
        if len(set(keys)) == len(keys):
            grid = candidate
            break
    else:
        raise ValueError('拨弦间隔过短，无法整理成节奏谱。请导出保留原始时间的 MIDI。')
    quantized = []
    for string in range(1, 7):
        line = []
        for note in sorted((note for note in notes if note['string'] == string), key=lambda n: n['start']):
            first = max(0, round(note['start'] * bpm / 60 * DIVISIONS / grid) * grid)
            last = max(first + grid, round(note['end'] * bpm / 60 * DIVISIONS / grid) * grid)
            if line and first < line[-1]['tick_end']:
                if first <= line[-1]['tick_start']:
                    raise ValueError('同一根弦上有重叠音符，请调整指法。')
                line[-1]['tick_end'] = first
            line.append({**note, 'tick_start': first, 'tick_end': last})
        quantized.extend(line)
    numerator, denominator = map(int, meter.split('/'))
    measure_ticks = round(numerator * 4 / denominator * DIVISIONS)
    end = max(math.ceil(duration * bpm / 60 * DIVISIONS / grid) * grid,
              max(note['tick_end'] for note in quantized))
    return sorted(quantized, key=lambda n: (n['tick_start'], n['string'])), bpm, meter, measure_ticks, end


def add(parent, tag, text=None, **attributes):
    element = ET.SubElement(parent, tag, attributes)
    if text is not None: element.text = str(text)
    return element


def spans(length):
    while length > 0:
        ticks, name, dotted = next(value for value in VALUES if value[0] <= length)
        yield ticks, name, dotted
        length -= ticks


def write_span(measure, length, voice, note=None, tied_before=False, tied_after=False):
    parts = list(spans(length))
    for index, (ticks, name, dotted) in enumerate(parts):
        element = add(measure, 'note')
        if note is None:
            add(element, 'rest')
        else:
            pitch = add(element, 'pitch')
            step, alter = [('C', 0), ('C', 1), ('D', 0), ('D', 1), ('E', 0), ('F', 0),
                           ('F', 1), ('G', 0), ('G', 1), ('A', 0), ('A', 1), ('B', 0)][note['midi'] % 12]
            add(pitch, 'step', step)
            if alter: add(pitch, 'alter', alter)
            add(pitch, 'octave', note['midi'] // 12 - 1)
        add(element, 'duration', ticks)
        stop, start = tied_before or index > 0, tied_after or index < len(parts) - 1
        if note is not None:
            if stop: add(element, 'tie', type='stop')
            if start: add(element, 'tie', type='start')
        add(element, 'voice', voice); add(element, 'type', name)
        if dotted: add(element, 'dot')
        if note is not None:
            notation = add(element, 'notations')
            if stop: add(notation, 'tied', type='stop')
            if start: add(notation, 'tied', type='start')
            technical = add(notation, 'technical')
            add(technical, 'string', note['string']); add(technical, 'fret', note['fret'])


def export_musicxml(score):
    notes, bpm, meter, measure_ticks, end = quantize_score(score)
    root = ET.Element('score-partwise', version='4.0')
    work = add(root, 'work'); add(work, 'work-title', str(score.get('title', '吉他六线谱'))[:200])
    identification = add(root, 'identification'); encoding = add(identification, 'encoding')
    add(encoding, 'software', 'Stringnote')
    add(encoding, 'encoding-description', 'Estimated rhythm: adaptive grid (1/16 to 1/128); meter selected by user; verify against recording.')
    parts = add(root, 'part-list'); instrument = add(parts, 'score-part', id='P1')
    add(instrument, 'part-name', 'Guitar'); part = add(root, 'part', id='P1')
    numerator, denominator = meter.split('/')
    by_string = {voice: [note for note in notes if note['string'] == voice] for voice in range(1, 7)}
    for bar in range(math.ceil(end / measure_ticks)):
        measure = add(part, 'measure', number=str(bar + 1))
        lower, upper = bar * measure_ticks, (bar + 1) * measure_ticks
        if bar == 0:
            attributes = add(measure, 'attributes'); add(attributes, 'divisions', DIVISIONS)
            signature = add(attributes, 'time'); add(signature, 'beats', numerator); add(signature, 'beat-type', denominator)
            clef = add(attributes, 'clef'); add(clef, 'sign', 'TAB'); add(clef, 'line', 5)
            details = add(attributes, 'staff-details'); add(details, 'staff-lines', 6)
            for line, (step, octave) in enumerate([('E', 2), ('A', 2), ('D', 3), ('G', 3), ('B', 3), ('E', 4)], 1):
                tuning = add(details, 'staff-tuning', line=str(line)); add(tuning, 'tuning-step', step); add(tuning, 'tuning-octave', octave)
            direction = add(measure, 'direction'); direction_type = add(direction, 'direction-type')
            metronome = add(direction_type, 'metronome'); add(metronome, 'beat-unit', 'quarter'); add(metronome, 'per-minute', round(bpm, 2))
            add(direction, 'sound', tempo=str(bpm))
        for voice in range(1, 7):
            if voice > 1:
                backup = add(measure, 'backup'); add(backup, 'duration', measure_ticks)
            cursor = lower
            for note in by_string[voice]:
                if note['tick_start'] >= upper: break
                if note['tick_end'] <= lower: continue
                begin, finish = max(lower, note['tick_start']), min(upper, note['tick_end'])
                if begin > cursor: write_span(measure, begin - cursor, voice)
                write_span(measure, finish - begin, voice, note,
                           tied_before=note['tick_start'] < lower, tied_after=note['tick_end'] > upper)
                cursor = finish
            if cursor < upper: write_span(measure, upper - cursor, voice)
    ET.indent(root)
    return ET.tostring(root, encoding='utf-8', xml_declaration=True)
