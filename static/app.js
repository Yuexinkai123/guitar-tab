/* No audio leaves this computer. The API is served on localhost. */
'use strict';
const $ = id => document.getElementById(id);
const TUNING = [64, 59, 55, 50, 45, 40];
const NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
let selectedFile, originalUrl, result, editingIndex = -1, busy = false;
let synthContext, playing = false, playbackStart = 0, playbackSpeed = 1, sources = [], animation;
let currentHighlight = '', playbackPosition = 0, playbackEpoch = 0;

function formatTime(seconds) {
  return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}
function pitchName(midi) { return NAMES[((midi % 12) + 12) % 12] + (Math.floor(midi / 12) - 1); }
function setError(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function setBusy(value) {
  busy = value;
  $('generate').disabled = value || !selectedFile;
  $('demo').disabled = value;
  ['mode', 'clip-start', 'clip-length', 'file-input'].forEach(id => $(id).disabled = value);
  $('clip-start').disabled = value || $('clip-length').value === '0';
  $('generate').innerHTML = value ? '正在生成…' : '生成六线谱 <span>↗</span>';
}
async function selectFile(file) {
  if (!file || busy) return;
  setError();
  if (!/\.(mp3|wav|flac|ogg)$/i.test(file.name)) return setError('请选择 MP3、WAV、FLAC 或 OGG 音频文件。');
  if (file.size > 40 * 1024 * 1024) return setError('文件超过 40 MB，请压缩或截取后上传。');
  if (!file.size) return setError('这个文件是空的。');
  selectedFile = file;
  if (originalUrl) URL.revokeObjectURL(originalUrl);
  originalUrl = URL.createObjectURL(file);
  $('original-audio').src = originalUrl;
  $('file-title').textContent = file.name;
  $('file-subtitle').textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB · 已准备好，可以换一首`;
  $('file-cta').textContent = '重新选择';
  $('file-preview').hidden = false;
  $('generate').disabled = false;
  $('clip-start').value = 0;
  // A neutral waveform is drawn until the browser has decoded the real audio.
  const canvas = $('waveform');
  canvas.hidden = false;
  const canvasContext = canvas.getContext('2d');
  canvasContext.clearRect(0, 0, canvas.width, canvas.height);
  try {
    const context = new AudioContext();
    try {
      const buffer = await context.decodeAudioData(await file.arrayBuffer());
      if (selectedFile !== file) return;
      const data = buffer.getChannelData(0);
      canvas.width = Math.max(300, Math.round(canvas.clientWidth * devicePixelRatio));
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#8b9f75';
      const bars = Math.floor(canvas.width / 5), stride = Math.max(1, Math.floor(data.length / bars));
      for (let index = 0; index < bars; index++) {
        let peak = 0;
        for (let j = index * stride; j < Math.min((index + 1) * stride, data.length); j += 16) peak = Math.max(peak, Math.abs(data[j]));
        const height = Math.max(2, peak * 60);
        ctx.fillRect(index * 5, (64 - height) / 2, 3, height);
      }
      $('file-subtitle').textContent = `${formatTime(buffer.duration)} · ${(file.size / 1024 / 1024).toFixed(1)} MB · 可以重新选择`;
    } finally { await context.close(); }
  } catch { /* The backend can decode some formats that the browser cannot preview. */ }
}
$('file-input').addEventListener('change', event => selectFile(event.target.files[0]));
$('dropzone').addEventListener('keydown', event => {
  if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); if (!busy) $('file-input').click(); }
});
['dragenter', 'dragover'].forEach(type => $('dropzone').addEventListener(type, event => {
  event.preventDefault(); if (!busy) $('dropzone').classList.add('dragover');
}));
['dragleave', 'drop'].forEach(type => $('dropzone').addEventListener(type, event => {
  event.preventDefault(); $('dropzone').classList.remove('dragover');
}));
$('dropzone').addEventListener('drop', event => selectFile(event.dataTransfer.files[0]));

$('clip-length').addEventListener('change', () => {
  const fullSong = $('clip-length').value === '0';
  $('clip-start').disabled = fullSong;
  if (fullSong) $('clip-start').value = 0;
});
$('generate').addEventListener('click', async () => {
  if (!selectedFile || busy) return;
  const start = $('clip-length').value === '0' ? 0 : Number($('clip-start').value);
  if (!Number.isFinite(start) || start < 0 || start > 7200) return setError('起始时间需要在 0～7200 秒之间。');
  stopPlayback(); $('original-audio').pause(); $('melody-audio').pause(); setError();
  setBusy(true); $('progress').hidden = false;
  $('progress-bar').style.width = '0%'; $('progress-percent').textContent = '0%';
  $('progress-message').textContent = '正在读取文件';
  const body = new FormData();
  body.append('file', selectedFile); body.append('start', start);
  body.append('seconds', $('clip-length').value); body.append('mode', $('mode').value);
  try {
    const response = await fetch('/api/transcribe', {method: 'POST', body});
    const payload = await response.json();
    if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : '上传失败，请检查输入。');
    let failures = 0;
    while (true) {
      await new Promise(resolve => setTimeout(resolve, 1200));
      let state;
      try {
        const check = await fetch(`/api/jobs/${payload.id}`, {signal: AbortSignal.timeout(15000)});
        if (!check.ok) throw new Error('无法读取任务状态');
        state = await check.json(); failures = 0;
      } catch (error) {
        if (++failures >= 5) throw new Error('与本机程序的连接中断，请确认程序窗口仍然开启。');
        $('progress-message').textContent = '连接暂时中断，正在重试…';
        continue;
      }
      $('progress-message').textContent = state.message;
      $('progress-percent').textContent = `${state.progress}%`;
      $('progress-bar').style.width = `${state.progress}%`;
      if (state.status === 'error') throw new Error(state.message);
      if (state.status === 'done') {
        result = {...state.result, title: selectedFile.name.replace(/\.[^.]+$/, ''), demo: false, edited: false};
        $('melody-audio').src = `/api/jobs/${payload.id}/melody`;
        showResult(); break;
      }
    }
  } catch (error) { setError(error.message || '分析失败，请稍后再试。'); }
  finally { setBusy(false); $('progress').hidden = true; }
});

$('demo').addEventListener('click', () => {
  stopPlayback(); $('original-audio').pause(); $('melody-audio').pause();
  const pitches = [64, 67, 69, 67, 64, 62, 60, 62, 64, 67, 71, 69, 67, 64];
  let time = 0.2;
  const notes = pitches.map((midi, index) => {
    const duration = [3, 7, 13].includes(index) ? 0.8 : 0.4;
    const string = midi >= 64 ? 1 : 2, fret = midi - TUNING[string - 1];
    const note = {midi, original_midi: midi, start: time, end: time + duration - 0.04,
      confidence: 1, name: pitchName(midi), string, fret};
    time += duration; return note;
  });
  result = {notes, title: '午后的小旋律', duration: 8, clip_start: 0,
    bpm: 150, octave_shift: 0, omitted_notes: 0, mode: 'solo', demo: true, edited: false};
  showResult();
});

function showResult() {
  $('result').hidden = false;
  $('result-label').textContent = result.demo ? 'DEMO · 示例旋律' : 'YOUR GUITAR TAB';
  $('result-heading').textContent = result.demo ? '先听听，六线谱怎么弹。' : result.polyphonic ? '吉他声部，有谱了。' : '你的旋律，有谱了。';
  $('result-subtitle').textContent = result.demo ? '手工编写的演示旋律，用来体验读谱和试听；不是音乐识别结果。'
    : `${result.title} · ${result.full_song ? '整首歌曲 · ' : ''}${formatTime(result.clip_start)}—${formatTime(result.clip_start + result.duration)} · ${result.notes.length} 个音符`;
  $('paper-title').textContent = result.title + (result.edited ? ' · 已修改' : '');
  $('paper-detail').textContent = '时间六线谱 · 第 1 弦在最上方 · 0–15 品';
  $('tempo').textContent = result.bpm ? `参考速度 ≈ ${result.bpm} BPM` : '节拍未确定';
  playbackPosition = 0;
  updatePlaybackUI();
  $('extracted').hidden = result.demo;
  const stemLabel = result.mode === 'guitar' ? '分离出的吉他 · 对照检查' : result.mode === 'song' ? '分离出的人声 · 对照检查' : '识别所用录音 · 对照检查';
  $('extracted').querySelector('span').textContent = stemLabel;
  $('melody-audio').setAttribute('aria-label', stemLabel);
  $('score-bpm').value = Math.max(30, Math.min(300, result.bpm || 120));
  $('score-meter').value = '4/4'; $('export-status').textContent = '';
  const octave = result.octave_shift ? ` 为适合吉他音域，整段旋律已${result.octave_shift > 0 ? '升高' : '降低'} ${Math.abs(result.octave_shift) / 12} 个八度。` : '';
  const omitted = result.omitted_notes ? ` 另有 ${result.omitted_notes} 个超出音域或无法安排同时发声指法的音未收录。` : '';
  $('result-note').textContent = result.demo ? '试着点击一个数字，看看同一个音可以怎样安排在不同的琴弦上。'
    : (result.polyphonic ? '这是吉他声部的多音时间谱草稿，保留原始音高和实测时间；指法是推算结果。' : '这是主旋律的时间谱草稿，保留识别到的时值。') + '网页横轴为秒；导出 MusicXML 可继续整理节奏。请对照原曲检查错音、漏音和连音。' + octave + omitted + ' ' + (result.warnings || []).join(' ');
  renderTab(); $('result').scrollIntoView({behavior: 'smooth', block: 'start'});
}

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgElement(tag, attributes = {}, text = '') {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, value));
  if (text) node.textContent = text;
  return node;
}
function renderTab() {
  const container = $('tab-container'); container.replaceChildren(); currentHighlight = '';
  const secondsPerRow = window.innerWidth <= 760 ? 2 : 4;
  const width = 1000, left = 46, right = 24, scale = (width - left - right) / secondsPerRow;
  const stringLabels = ['e', 'B', 'G', 'D', 'A', 'E'];
  for (let row = 0; row < Math.ceil(result.duration / secondsPerRow); row++) {
    const rowStart = row * secondsPerRow, rowEnd = Math.min(rowStart + secondsPerRow, result.duration);
    const svg = svgElement('svg', {viewBox: `0 0 ${width} 174`, role: 'group', 'aria-label': `${rowStart} 到 ${rowEnd} 秒的六线谱`});
    // Subtle timing divisions are seconds, not inferred musical bar lines.
    for (let second = 0; second <= secondsPerRow; second++) {
      const x = left + second * scale;
      svg.append(svgElement('line', {x1: x, y1: 35, x2: x, y2: 128, stroke: '#e9ecdf', 'stroke-dasharray': '3 5'}));
      svg.append(svgElement('text', {x, y: 151, fill: '#a1a993', 'font-size': 10, 'text-anchor': second === secondsPerRow ? 'end' : 'start'}, `${rowStart + second}s`));
    }
    stringLabels.forEach((label, string) => {
      const y = 42 + string * 16;
      svg.append(svgElement('text', {x: 15, y: y + 4, fill: '#95a184', 'font-family': 'monospace', 'font-size': 12}, label));
      svg.append(svgElement('line', {x1: left, y1: y, x2: width - right, y2: y, stroke: '#c8d0ba', 'stroke-width': 1}));
    });
    result.notes.forEach((note, index) => {
      if (note.end <= rowStart || note.start >= rowStart + secondsPerRow) return;
      const y = 42 + (note.string - 1) * 16;
      const x = left + (Math.max(note.start, rowStart) - rowStart) * scale;
      const end = left + (Math.min(note.end, rowStart + secondsPerRow) - rowStart) * scale;
      if (end - x > 14) svg.append(svgElement('line', {x1: x + 7, y1: y, x2: end, y2: y, stroke: '#8c9d73', 'stroke-width': 2, opacity: 0.65}));
      if (note.start < rowStart) return;
      const group = svgElement('g', {class: `tab-note${note.confidence < 0.65 ? ' low' : ''}`, 'data-index': index,
        tabindex: 0, role: 'button', 'aria-label': `${note.start.toFixed(2)}秒，${pitchName(note.midi)}，${note.string}弦${note.fret}品，点击修改`});
      group.append(svgElement('title', {}, `${pitchName(note.midi)} · ${note.start.toFixed(2)}–${note.end.toFixed(2)} 秒 · 点击修改`));
      group.append(svgElement('rect', {class: 'note-bg', x: x - 11, y: y - 11, width: 22, height: 22, rx: 5}));
      group.append(svgElement('text', {x, y: y + 4, 'text-anchor': 'middle'}, String(note.fret)));
      group.addEventListener('click', () => openEditor(index));
      group.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); openEditor(index); }
      });
      svg.append(group);
    });
    container.append(svg);
  }
}

function pluck(context, destination, midi, at, duration, elapsed) {
  const source = GuitarSynth.pluck(context, destination, midi, at, duration, elapsed);
  if (source) sources.push(source);
}
function playbackTime() {
  if (!result) return 0;
  return Math.min(result.duration, Math.max(playbackPosition, playing && synthContext
    ? (synthContext.currentTime - playbackStart) * playbackSpeed : playbackPosition));
}
function updatePlaybackUI(time = playbackPosition) {
  if (!result) return;
  const seek = $('play-seek'), label = `${formatTime(time)} / ${formatTime(result.duration)}`;
  seek.max = result.duration; seek.value = time;
  seek.style.setProperty('--played', `${result.duration ? time / result.duration * 100 : 0}%`);
  seek.setAttribute('aria-valuetext', label);
  $('play-time').textContent = label;
  $('play-tab').textContent = playing ? 'Ⅱ' : '▶';
  $('play-tab').setAttribute('aria-label', playing ? '暂停播放' : '播放生成的旋律');
  const indices = result.notes.flatMap((note, index) => time >= note.start && time < note.end ? [index] : []);
  const key = indices.join(',');
  if (key !== currentHighlight) {
    document.querySelectorAll('.tab-note.active').forEach(node => node.classList.remove('active'));
    indices.forEach(index => document.querySelector(`.tab-note[data-index="${index}"]`)?.classList.add('active'));
    currentHighlight = key;
  }
}
async function startPlayback() {
  if (!result) return;
  $('original-audio').pause(); $('melody-audio').pause();
  const epoch = ++playbackEpoch;
  if (playbackPosition >= result.duration) playbackPosition = 0;
  try {
    synthContext ??= new AudioContext();
    playbackSpeed = Number($('speed').value);
    playbackStart = synthContext.currentTime + 0.08 - playbackPosition / playbackSpeed;
    playing = true; updatePlaybackUI();
    await synthContext.resume();
    if (epoch !== playbackEpoch || !playing) return;
    // Re-anchor after resuming a suspended audio context. Never replay elapsed notes.
    playbackStart = synthContext.currentTime + 0.08 - playbackPosition / playbackSpeed;
    sources = [];
    for (const note of result.notes) {
      if (note.end <= playbackPosition) continue;
      const begin = Math.max(note.start, playbackPosition);
      pluck(synthContext, synthContext.destination, note.midi,
        playbackStart + begin / playbackSpeed, (note.end - begin) / playbackSpeed,
        Math.max(0, playbackPosition - note.start) / playbackSpeed);
    }
    const tick = () => {
      if (!playing || epoch !== playbackEpoch) return;
      const time = playbackTime(); updatePlaybackUI(time);
      if (time >= result.duration) stopPlayback(false);
      else animation = requestAnimationFrame(tick);
    };
    tick();
  } catch {
    if (epoch === playbackEpoch) {
      stopPlayback(false); setError('无法播放旋律，请检查浏览器的声音设置后重试。');
    }
  }
}
function stopPlayback(reset = true) {
  playbackPosition = reset ? 0 : playbackTime();
  playbackEpoch++;
  playing = false; cancelAnimationFrame(animation);
  sources.forEach(source => { try { source.stop(); } catch {} }); sources = [];
  updatePlaybackUI();
}
$('play-tab').addEventListener('click', () => { if (playing) stopPlayback(false); else startPlayback(); });
$('play-seek').addEventListener('input', () => {
  if (!result) return;
  const position = Number($('play-seek').value), resume = playing;
  stopPlayback(false);
  playbackPosition = Math.min(result.duration, Math.max(0, position));
  updatePlaybackUI();
  if (resume && playbackPosition < result.duration) startPlayback();
});
$('restart-tab').addEventListener('click', () => { const resume = playing; stopPlayback(); if (resume) startPlayback(); });
$('speed').addEventListener('change', () => { if (playing) { stopPlayback(false); startPlayback(); } });
$('original-audio').addEventListener('play', () => { stopPlayback(false); $('melody-audio').pause(); });
$('melody-audio').addEventListener('play', () => { stopPlayback(false); $('original-audio').pause(); });

function openEditor(index) {
  stopPlayback(false); editingIndex = index;
  $('edit-error').textContent = '';
  const note = result.notes[index];
  $('edit-time').textContent = `${note.start.toFixed(2)}–${note.end.toFixed(2)} 秒 · ${pitchName(note.midi)}`;
  $('edit-pitch').replaceChildren();
  for (let midi = 40; midi <= 79; midi++) $('edit-pitch').add(new Option(pitchName(midi), midi));
  $('edit-pitch').value = note.midi; updatePositions(note.string);
  $('note-dialog').showModal();
}
function updatePositions(preferredString) {
  const midi = Number($('edit-pitch').value); $('edit-position').replaceChildren();
  TUNING.forEach((base, index) => {
    if (midi - base >= 0 && midi - base <= 15) $('edit-position').add(new Option(`第 ${index + 1} 弦 · 第 ${midi - base} 品`, index + 1));
  });
  if ([...$('edit-position').options].some(option => Number(option.value) === preferredString)) $('edit-position').value = preferredString;
}
$('edit-pitch').addEventListener('change', () => updatePositions(Number($('edit-position').value)));
$('save-note').addEventListener('click', () => {
  const midi = Number($('edit-pitch').value), string = Number($('edit-position').value);
  if (!string) return;
  const current = result.notes[editingIndex];
  if (result.notes.some((note, index) => index !== editingIndex && note.string === string && note.start < current.end && note.end > current.start)) {
    $('edit-error').textContent = '这根弦在此时还有另一个音，请选择其他指法。'; return;
  }
  Object.assign(result.notes[editingIndex], {midi, string, fret: midi - TUNING[string - 1], name: pitchName(midi), confidence: 1});
  result.edited = true; $('paper-title').textContent = result.title + ' · 已修改';
  $('note-dialog').close(); renderTab();
});

async function exportScore(format) {
  if (!result) return;
  const bpm = Number($('score-bpm').value);
  if (!Number.isFinite(bpm) || bpm < 30 || bpm > 300) {
    $('export-status').textContent = '排谱速度需要在 30～300 BPM 之间。'; return;
  }
  const buttons = [$('download-midi'), $('download-musicxml')];
  buttons.forEach(button => button.disabled = true); $('export-status').textContent = '正在导出…';
  try {
    const response = await fetch(`/api/export/${format}`, {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({...result, bpm, meter: $('score-meter').value})});
    if (!response.ok) { const error = await response.json(); throw new Error(error.detail || '导出失败'); }
    download(await response.arrayBuffer(), format === 'midi' ? 'mid' : 'musicxml',
      format === 'midi' ? 'audio/midi' : 'application/vnd.recordare.musicxml+xml');
    $('export-status').textContent = format === 'musicxml' ? '已导出当前修改后的谱。节奏与拍号请在谱软件中核对。' : '已导出，保留识别到的实际时间。';
  } catch (error) { $('export-status').textContent = error.message || '导出失败，请确认本机程序仍在运行。'; }
  finally { buttons.forEach(button => button.disabled = false); }
}
$('download-midi').addEventListener('click', () => exportScore('midi'));
$('download-musicxml').addEventListener('click', () => exportScore('musicxml'));

function download(content, extension, mime) {
  const url = URL.createObjectURL(new Blob([content], {type: mime}));
  const anchor = document.createElement('a'); anchor.href = url;
  anchor.download = `${result.title.replace(/[<>:"/\\|?*]/g, '_')}-六线谱.${extension}`;
  document.body.append(anchor); anchor.click(); anchor.remove(); setTimeout(() => URL.revokeObjectURL(url), 3000);
}
$('download-svg').addEventListener('click', () => {
  if (!result) return;
  const rows = [...$('tab-container').children];
  const svg = svgElement('svg', {width: 1000, height: rows.length * 174 + 80, viewBox: `0 0 1000 ${rows.length * 174 + 80}`});
  svg.append(svgElement('rect', {width: '100%', height: '100%', fill: '#fffefa'}));
  svg.append(svgElement('style', {}, '.note-bg{fill:#fffefa}.tab-note text{fill:#38502e;font:600 13px monospace}.tab-note.low text{fill:#b68a42}'));
  svg.append(svgElement('text', {x: 20, y: 30, fill: '#38502e', 'font-size': 18}, result.title + ' · 吉他六线谱'));
  svg.append(svgElement('text', {x: 20, y: 54, fill: '#899578', 'font-size': 11}, '标准调弦 · 时间轴单位：秒 · 1 弦在最上方 · 横线表示持续时长'));
  rows.forEach((row, index) => {
    const copy = row.cloneNode(true); copy.setAttribute('x', 0); copy.setAttribute('y', 80 + index * 174);
    copy.setAttribute('width', 1000); copy.setAttribute('height', 174);
    copy.querySelectorAll('.active').forEach(node => node.classList.remove('active'));
    svg.append(copy);
  });
  download(new XMLSerializer().serializeToString(svg), 'svg', 'image/svg+xml;charset=utf-8');
});
$('download-txt').addEventListener('click', () => {
  if (!result) return;
  const lines = [result.title + ' · 吉他六线谱', '标准调弦 E A D G B e · 第1弦在最上方',
    `分析片段：原曲 ${result.clip_start.toFixed(2)} 秒起 · 时值见音符清单（片段相对时间）`,
    '每列为一次起音组（30 毫秒内近同时），列宽不代表时长。', ''];
  const groups = [];
  for (const note of [...result.notes].sort((a, b) => a.start - b.start)) {
    const last = groups.at(-1);
    if (last && note.start - last[0].start <= .03 && !last.some(n => n.string === note.string)) last.push(note);
    else groups.push([note]);
  }
  for (let offset = 0; offset < groups.length; offset += 16) {
    const part = groups.slice(offset, offset + 16);
    ['e', 'B', 'G', 'D', 'A', 'E'].forEach((label, string) => {
      lines.push(label + ' |' + part.map(group => {
        const note = group.find(note => note.string === string + 1);
        return note ? String(note.fret).padStart(2, '-') + '--' : '----';
      }).join('') + '|');
    });
    lines.push('');
  }
  lines.push('音符清单：开始秒 — 结束秒 / 音高 / 弦 / 品');
  result.notes.forEach(note => lines.push(`${note.start.toFixed(3)} — ${note.end.toFixed(3)} / ${pitchName(note.midi)} / ${note.string} / ${note.fret}`));
  download('\ufeff' + lines.join('\r\n'), 'txt', 'text/plain;charset=utf-8');
});
$('print').addEventListener('click', () => { stopPlayback(); window.print(); });
let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (result && !playing) renderTab(); }, 200);
});

// A prepared diagnosis or completed job can be opened without uploading again.
async function restoreResultFromLink() {
  const id = new URL(location.href).searchParams.get('job');
  if (!id || !/^[a-f0-9]{32}$/.test(id)) return;
  try {
    const response = await fetch(`/api/jobs/${id}`);
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || '无法恢复结果');
    if (job.status !== 'done') throw new Error(job.message || '该任务尚未完成');
    if (result || busy) return;
    result = {...job.result, title: job.result.title || job.title || '已保存的六线谱', demo: false, edited: false};
    $('melody-audio').src = `/api/jobs/${id}/melody`;
    $('original-audio').src = `/api/jobs/${id}/original`;
    $('file-preview').hidden = false; $('waveform').hidden = true;
    $('file-title').textContent = result.title;
    $('file-subtitle').textContent = '已恢复结果；选择音乐文件可重新生成。';
    showResult();
  } catch (error) { setError(error.message); }
}
restoreResultFromLink();
