"""Local vocal separation, pitch tracking, and playable guitar fingering."""
from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

from .model_cache import ensure_checkpoint
from .audio_quality import gate_guitar, write_preview

# Index 0 is string 1 (high E), matching the display from top to bottom.
TUNING = (64, 59, 55, 50, 45, 40)
CHUNK_SECONDS = 30
CONTEXT_SECONDS = 0.5
MAX_SONG_SECONDS = 1800
_models = {}
_model_lock = threading.Lock()


def load_audio(path: Path, start: float, seconds: float, *, allow_silence=False):
    with sf.SoundFile(path) as f:
        duration = len(f) / f.samplerate
        if start >= duration:
            raise ValueError("起始时间已经超过歌曲长度，请调小起始时间。")
        f.seek(round(start * f.samplerate))
        audio = f.read(round(seconds * f.samplerate), dtype="float32", always_2d=True)
        rate = f.samplerate
    if len(audio) == 0 or (len(audio) < rate and not allow_silence):
        raise ValueError("分析片段至少需要 1 秒。")
    if not np.all(np.isfinite(audio)):
        raise ValueError("音频包含无效数据，请重新导出后上传。")
    if not allow_silence and np.max(np.abs(audio)) < 1e-5:
        raise ValueError("这个片段没有可听见的声音，请换一个时间段。")
    audio = audio[:, :2]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    return audio.T, rate, duration


def separate_vocals(audio: np.ndarray, rate: int, model_dir: Path, report: Callable):
    return separate_stem(audio, rate, model_dir, report, 'vocals')


def separate_stem(audio: np.ndarray, rate: int, model_dir: Path, report: Callable, stem='guitar'):
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    from demucs.htdemucs import HTDemucs

    model_name = 'htdemucs_6s' if stem == 'guitar' else 'htdemucs'
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    with _model_lock:
        if model_name not in _models:
            report(12, "加载声部分离模型（仅首次需要下载）")
            ensure_checkpoint(model_dir, report, model_name)
            torch.hub.set_dir(str(model_dir))
            # PyTorch 2.6 loads weights safely by default. Allow only the vendor
            # model class required by this fixed, checksum-verified checkpoint.
            with torch.serialization.safe_globals([HTDemucs]):
                _models[model_name] = get_model(model_name)
            _models[model_name].eval()
    model = _models[model_name]
    target_rate = model.samplerate
    divisor = math.gcd(rate, target_rate)
    wave = resample_poly(audio, target_rate // divisor, rate // divisor, axis=-1)
    wave = torch.tensor(wave, dtype=torch.float32)
    reference = wave.mean(0)
    mean, std = reference.mean(), reference.std().clamp(min=1e-6)
    normalized = (wave - mean) / std
    report(25, "正在从混音中分离吉他声部" if stem == 'guitar' else "正在分离人声与伴奏")
    with torch.inference_mode():
        stems = apply_model(model, normalized[None], device="cpu", shifts=0,
                            split=True, overlap=0.25, progress=False, num_workers=0)[0]
    vocals = (stems[model.sources.index(stem)] * std + mean).mean(0).numpy()
    return vocals, target_rate


def track_notes(audio: np.ndarray, rate: int):
    """Return notes with real time boundaries; never invent notes for silence."""
    divisor = math.gcd(rate, 22050)
    y = resample_poly(audio, 22050 // divisor, rate // divisor)
    rate, hop = 22050, 256
    f0, voiced, probability = librosa.pyin(
        y, sr=rate, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("B5"),
        frame_length=2048, hop_length=hop, fill_na=np.nan,
    )
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    energy_gate = max(float(np.max(rms)) * 0.035, 0.0001)
    valid = voiced & (probability >= 0.30) & (rms[:len(f0)] > energy_gate)
    midi = np.zeros(len(f0), dtype=int)
    midi[valid] = np.rint(librosa.hz_to_midi(f0[valid])).astype(int)
    # A five-frame median suppresses brief octave glitches and vibrato flips.
    midi = median_filter(midi, size=5, mode="nearest")
    midi[~valid] = 0
    frame_seconds = hop / rate
    runs = []
    start = 0
    for index in range(1, len(midi) + 1):
        if index < len(midi) and midi[index] == midi[start]:
            continue
        pitch = int(midi[start])
        if pitch and (index - start) * frame_seconds >= 0.085:
            note = {"midi": pitch, "start": round(start * frame_seconds, 3),
                    "end": round(min(index * frame_seconds, len(y) / rate), 3),
                    "confidence": round(float(np.mean(probability[start:index])), 3)}
            if (runs and runs[-1]["midi"] == pitch
                    and note["start"] - runs[-1]["end"] < 0.07):
                runs[-1]["end"] = note["end"]
                runs[-1]["confidence"] = min(runs[-1]["confidence"], note["confidence"])
            else:
                runs.append(note)
        start = index
    return runs


def arrange_fingering(notes: list[dict], max_fret: int = 15):
    """Choose one global octave shift, then minimize hand movement using DP."""
    if not notes:
        return [], 0
    low, high = min(TUNING), max(TUNING) + max_fret
    def octave_cost(shift):
        outside = sum(max(low - n["midi"] - shift, 0) + max(n["midi"] + shift - high, 0)
                      for n in notes)
        return outside * 100 + abs(shift)
    shift = min((-24, -12, 0, 12, 24), key=octave_cost)
    pitches = [n["midi"] + shift for n in notes]
    candidates = [[(s + 1, pitch - base) for s, base in enumerate(TUNING)
                   if 0 <= pitch - base <= max_fret] for pitch in pitches]
    # Rare out-of-range notes retain timing but are omitted and reported.
    retained = [(note, pitch, choices) for note, pitch, choices in zip(notes, pitches, candidates) if choices]
    if not retained:
        return [], shift
    costs, parents = [], []
    for index, (_, _, choices) in enumerate(retained):
        row, back = [], []
        for string, fret in choices:
            base_cost = fret * 0.08 + (0.2 if fret == 0 else 0)
            if index == 0:
                row.append(base_cost)
                back.append(-1)
                continue
            previous = retained[index - 1][2]
            options = [costs[-1][j] + abs(fret - pf) * 0.9 + abs(string - ps) * 0.7
                       + (2 if abs(fret - pf) > 5 else 0)
                       for j, (ps, pf) in enumerate(previous)]
            best = int(np.argmin(options))
            row.append(options[best] + base_cost)
            back.append(best)
        costs.append(row)
        parents.append(back)
    chosen = int(np.argmin(costs[-1]))
    result = []
    for index in range(len(retained) - 1, -1, -1):
        note, pitch, choices = retained[index]
        string, fret = choices[chosen]
        result.append({**note, "original_midi": note["midi"], "midi": pitch,
                       "name": librosa.midi_to_note(pitch), "string": string, "fret": fret})
        chosen = parents[index][chosen]
    return list(reversed(result)), shift


def merge_chunk_notes(notes, polyphonic=False):
    """Join matching boundary fragments, while preserving real pauses."""
    if polyphonic:
        # Merge only fragments belonging to the SAME detected onset across the
        # chunk boundary. Nearby repeated plucks must remain separate events.
        merged = []
        for note in sorted(notes, key=lambda n: (n['start'], n['midi'])):
            match = next((prior for prior in reversed(merged)
                          if prior['midi'] == note['midi']
                          and prior.get('_cut_right') and note.get('_cut_left')
                          and abs(note['start'] - prior['end']) < 0.004
                          and (abs(prior.get('event_start', -10) - note.get('event_start', 10)) < 0.08
                               or note.get('_context_start'))), None)
            if match is not None:
                match['end'] = max(match['end'], note['end'])
                match['confidence'] = min(match['confidence'], note['confidence'])
                match['_cut_right'] = note.get('_cut_right', False)
            else:
                merged.append(dict(note))
        for note in merged:
            for key in ('event_start', '_cut_left', '_cut_right', '_context_start'):
                note.pop(key, None)
        return [note for note in merged if note['end'] - note['start'] >= 0.085]
    result = []
    for note in sorted(notes, key=lambda n: n["start"]):
        if result and result[-1]["midi"] == note["midi"] and note["start"] - result[-1]["end"] < 0.07:
            result[-1]["end"] = max(result[-1]["end"], note["end"])
            result[-1]["confidence"] = min(result[-1]["confidence"], note["confidence"])
        else:
            result.append(dict(note))
    for note in result:
        for key in ('event_start', '_cut_left', '_cut_right', '_context_start'):
            note.pop(key, None)
    return [note for note in result if note["end"] - note["start"] >= 0.085]


def analyze(path: Path, start: float, seconds: float, mode: str, model_dir: Path,
            output: Path, report: Callable):
    report(5, "读取歌曲")
    info = sf.info(path)
    total, rate = info.frames / info.samplerate, info.samplerate
    full_song = seconds == 0
    polyphonic = mode in ('guitar', 'guitar_solo')
    if start < 0 or start >= total:
        raise ValueError("起始时间已经超过歌曲长度，请调小起始时间。")
    if full_song and (start != 0 or total > MAX_SONG_SECONDS):
        raise ValueError("整首模式从歌曲开头分析，目前支持最长 30 分钟的音乐。")
    duration = min(seconds, total - start) if not full_song else total
    if duration < 1:
        raise ValueError("分析片段至少需要 1 秒。")
    frames = min(round(duration * rate), info.frames - round(start * rate))
    block_frames = max(1, round(CHUNK_SECONDS * rate))
    chunk_count = math.ceil(frames / block_frames)
    notes, tempos, peak = [], [], 0.0
    supported_frames = 0
    # Write extracted audio progressively; peak memory does not grow with song length.
    with sf.SoundFile(output, mode="w", samplerate=rate, channels=1, subtype="PCM_16") as target:
        for index, frame_start in enumerate(range(0, frames, block_frames)):
            frame_end = min(frame_start + block_frames, frames)
            read_start = max(0, frame_start - round(CONTEXT_SECONDS * rate))
            read_end = min(frames, frame_end + round(CONTEXT_SECONDS * rate))
            audio, _, _ = load_audio(path, start + read_start / rate,
                                      (read_end - read_start) / rate, allow_silence=True)
            peak = max(peak, float(np.max(np.abs(audio))))

            def chunk_report(percent, message):
                overall = 5 + (index + percent / 100) / chunk_count * 88
                report(round(overall), f"第 {index + 1}/{chunk_count} 段 · {message}")

            if np.max(np.abs(audio)) < 1e-5:
                target.write(np.zeros(frame_end - frame_start, dtype=np.float32))
                chunk_report(100, "静音段已保留")
                continue
            if mode == "song":
                melody, melody_rate = separate_vocals(audio, rate, model_dir, chunk_report)
            elif mode == 'guitar':
                melody, melody_rate = separate_stem(audio, rate, model_dir, chunk_report)
                melody, support = gate_guitar(melody, melody_rate, audio, rate)
                support_first = round((frame_start - read_start) / rate * melody_rate)
                support_last = round((frame_end - read_start) / rate * melody_rate)
                supported_frames += float(np.sum(support[support_first:support_last])) * rate / melody_rate
            else:
                melody, melody_rate = audio.mean(0), rate
            chunk_report(65, "识别吉他多音与时值" if polyphonic else "识别旋律音高与时值")
            if polyphonic:
                from .polyphonic import track_polyphonic
                detected = track_polyphonic(melody, melody_rate, chunk_report)
            else:
                detected = track_notes(melody, melody_rate)
            offset = read_start / rate
            lower, upper = frame_start / rate, frame_end / rate
            for note in detected:
                begin, end = max(lower, note["start"] + offset), min(upper, note["end"] + offset)
                if end > begin:
                    notes.append({**note, "start": round(begin, 3), "end": round(end, 3),
                                  'event_start': note['start'] + offset,
                                  '_cut_left': note['start'] + offset < lower - 0.025,
                                  '_cut_right': note['end'] + offset > upper + 0.001,
                                  '_context_start': note['start'] < 0.12})
            divisor = math.gcd(melody_rate, rate)
            playback = resample_poly(melody, rate // divisor, melody_rate // divisor)
            first = frame_start - read_start
            core = playback[first:first + frame_end - frame_start]
            target.write(np.pad(core, (0, max(0, frame_end - frame_start - len(core)))))
            # A bounded excerpt suffices for a reference tempo.
            if len(tempos) < 3:
                sample = librosa.resample(audio.mean(0), orig_sr=rate, target_sr=22050)
                tempo, _ = librosa.beat.beat_track(y=sample, sr=22050)
                bpm = float(np.asarray(tempo).reshape(-1)[0])
                if np.isfinite(bpm) and bpm > 0:
                    tempos.append(bpm)
            chunk_report(100, "本段处理完成")
    if peak < 1e-5:
        raise ValueError("这个片段没有可听见的声音，请换一个时间段。")
    guitar_coverage = supported_frames / frames if mode == 'guitar' else 1.
    if mode == 'guitar' and guitar_coverage < .08:
        raise ValueError('这次没有可靠提取到吉他声部，分离结果大部分接近静音，不能据此生成可信的吉他谱。'
                         '若要弹歌曲唱出来的旋律，请选择“人声旋律改成吉他”；若要还原原曲吉他，请换一个吉他更清晰的片段或纯吉他录音。')
    notes = merge_chunk_notes(notes, polyphonic=polyphonic)
    if not notes:
        raise ValueError("没有识别到稳定的音符。请确认选对声部；纯吉他录音请选择“纯吉他录音”。")
    report(95, "合并旋律并安排整段吉他指法")
    if polyphonic:
        from .fingering import arrange_polyphonic
        arranged, shift = arrange_polyphonic(notes)
    else:
        arranged, shift = arrange_fingering(notes)
    if not arranged:
        raise ValueError("识别到的音超出吉他音域，请尝试其他片段。")
    audio_quality = write_preview(output)
    audio_quality['active_ratio'] = round(guitar_coverage, 3)
    warnings = []
    if mode == 'guitar' and guitar_coverage < .5:
        warnings.append('本模型只在部分时间提取到足够清晰的吉他信号，结果不完整；其余时间保留为空白，请先对照分离音频。')
    return {"notes": arranged, "octave_shift": shift, "omitted_notes": len(notes) - len(arranged),
            "bpm": round(float(np.median(tempos))) if tempos else None, "duration": round(frames / rate, 3),
            "song_duration": round(total, 3), "clip_start": start, "mode": mode,
            "full_song": full_song, "chunks": chunk_count,
            "polyphonic": polyphonic, "engine": 'basic-pitch-onnx' if polyphonic else 'pyin',
            'audio_quality': audio_quality, 'warnings': warnings,
            "tuning": ["E4", "B3", "G3", "D3", "A2", "E2"]}
