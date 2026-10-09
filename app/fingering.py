"""Beam search voicing with one ringing note per string and bounded hand stretch."""
import librosa

TUNING = (64, 59, 55, 50, 45, 40)


def arrange_polyphonic(notes, max_fret=15, beam_width=64):
    if not notes:
        return [], 0
    # Retain original octave for instrument transcription. No unannounced rearrangement.
    states = [(0.0, (), 0.0, None)]
    ordered = sorted(enumerate(notes), key=lambda pair: (pair[1]['start'], pair[1]['midi']))
    for index, note in ordered:
        next_states = []
        choices = [(string + 1, note['midi'] - base) for string, base in enumerate(TUNING)
                   if 0 <= note['midi'] - base <= max_fret]
        for cost, active, hand, path in states:
            ringing = tuple(entry for entry in active if entry[3] > note['start'])
            used = {entry[0] for entry in ringing}
            fretted = [entry[1] for entry in ringing if entry[1] > 0]
            for string, fret in choices:
                if string in used:
                    continue
                positions = fretted + ([fret] if fret else [])
                if positions and max(positions) - min(positions) > 4:
                    continue
                center = sum(positions) / len(positions) if positions else hand
                penalty = abs(center - hand) * 0.25 + fret * 0.03
                next_states.append((cost + penalty, ringing + ((string, fret, index, note['end']),),
                                    center, (path, index, string, fret)))
            # Report omitted notes instead of inventing an impossible seven-string voicing.
            next_states.append((cost + 8 + 12 * note.get('confidence', 0.5), ringing, hand, path))
        best = {}
        for state in sorted(next_states, key=lambda item: item[0]):
            key = (tuple((entry[0], entry[1], round(entry[3], 2)) for entry in state[1]), round(state[2], 2))
            if key not in best:
                best[key] = state
            if len(best) >= beam_width:
                break
        states = list(best.values())
    path = min(states, key=lambda item: item[0])[3]
    result = []
    while path is not None:
        path, index, string, fret = path
        note = notes[index]
        result.append({**note, 'original_midi': note['midi'], 'name': librosa.midi_to_note(note['midi']),
                       'string': string, 'fret': fret})
    return sorted(result, key=lambda note: (note['start'], note['string'])), 0
