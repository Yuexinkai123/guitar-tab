const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const {JSDOM} = require('jsdom');

function page(score) {
  const dom = new JSDOM(fs.readFileSync('static/index.html', 'utf8'), {
    runScripts: 'outside-only', url: 'http://127.0.0.1:8765/', pretendToBeVisual: true,
  });
  const {window} = dom;
  const downloads = [];
  const clock = {context: null, oscillators: [], frames: new Map(), nextFrame: 0};
  window.AudioContext = class {
    constructor() { this.currentTime = 0; this.sampleRate = 44100; this.destination = {}; clock.context = this; }
    async resume() {}
    createOscillator() {
      const oscillator = {frequency: {value: 0}, connect() {},
        start(time) { this.startedAt = time; }, stop(time) { this.stoppedAt = time; this.stopped = true; }};
      clock.oscillators.push(oscillator); return oscillator;
    }
    createBuffer(channels, length, rate) {
      const data = new Float32Array(length);
      return {duration: length / rate, copyToChannel(input) { data.set(input); }, getChannelData() { return data; }};
    }
    createBufferSource() {
      const source = {connect() {}, start(time, offset) { this.startedAt = time; this.offset = offset; },
        stop(time) { this.stoppedAt = time; this.stopped = true; }};
      clock.oscillators.push(source); return source;
    }
    createGain() { return {connect() {}, gain: {setValueAtTime() {}, linearRampToValueAtTime() {}, exponentialRampToValueAtTime() {}}}; }
  };
  window.requestAnimationFrame = callback => {
    const id = ++clock.nextFrame; clock.frames.set(id, callback); return id;
  };
  window.cancelAnimationFrame = id => clock.frames.delete(id);
  clock.advance = seconds => {
    clock.context.currentTime += seconds;
    const callbacks = [...clock.frames.values()]; clock.frames.clear(); callbacks.forEach(callback => callback());
  };
  window.Element.prototype.scrollIntoView = function () {};
  window.HTMLMediaElement.prototype.pause = function () {};
  window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  window.HTMLDialogElement.prototype.close = function () { this.open = false; };
  window.URL.createObjectURL = blob => { downloads.push(blob); return 'blob:test'; };
  window.URL.revokeObjectURL = () => {};
  window.HTMLAnchorElement.prototype.click = function () {};
  window.eval(fs.readFileSync('static/guitar-synth.js', 'utf8'));
  window.eval(fs.readFileSync('static/app.js', 'utf8') + (score ? `\nresult = ${JSON.stringify(score)}; showResult();` : ''));
  if (!score) window.document.getElementById('demo').click();
  return {dom, window, document: window.document, downloads, clock};
}

test('demo is explicitly labeled and renders all notes', () => {
  const {dom, document} = page();
  assert.equal(document.querySelector('#result').hidden, false);
  assert.match(document.querySelector('#result-subtitle').textContent, /不是音乐识别结果/);
  assert.equal(document.querySelectorAll('.tab-note').length, 14);
  assert.equal(document.querySelectorAll('#tab-container > svg').length, 2);
  dom.window.close();
});

const chordScore = {title: 'Chord', duration: 3, bpm: 120, mode: 'guitar', polyphonic: true, clip_start: 0,
  notes: [{midi: 48, string: 6, fret: 8, start: .2, end: 1.5, confidence: .9},
          {midi: 52, string: 5, fret: 7, start: .2, end: 1.5, confidence: .9},
          {midi: 55, string: 4, fret: 5, start: .2, end: 1.5, confidence: .9}]};

test('polyphonic playback highlights all ringing notes and editing rejects string conflicts', async () => {
  const {dom, window, document, clock} = page(chordScore);
  assert.match(document.querySelector('#extracted').textContent, /分离出的吉他/);
  document.querySelector('#play-tab').click(); await new Promise(resolve => setImmediate(resolve));
  clock.advance(.58);
  assert.equal(document.querySelectorAll('.tab-note.active').length, 3);
  assert.equal(clock.oscillators.length, 3);
  document.querySelector('.tab-note').dispatchEvent(new window.Event('click'));
  document.querySelector('#edit-pitch').value = '52';
  document.querySelector('#edit-pitch').dispatchEvent(new window.Event('change'));
  document.querySelector('#edit-position').value = '5';
  document.querySelector('#save-note').click();
  assert.equal(document.querySelector('#note-dialog').open, true);
  assert.match(document.querySelector('#edit-error').textContent, /另一个音/);
  assert.match(document.querySelector('.tab-note').getAttribute('aria-label'), /C3，6弦8品/);
  dom.window.close();
});

test('MIDI and MusicXML exports send the current edited notes and chosen meter', async () => {
  const {dom, window, document, downloads} = page();
  const requests = [];
  window.fetch = async (url, options) => {
    requests.push({url, body: JSON.parse(options.body)});
    return {ok: true, arrayBuffer: async () => new ArrayBuffer(4)};
  };
  document.querySelector('.tab-note').dispatchEvent(new window.Event('click'));
  document.querySelector('#edit-position').value = '2'; document.querySelector('#save-note').click();
  document.querySelector('#score-bpm').value = '90'; document.querySelector('#score-meter').value = '3/4';
  for (const format of ['midi', 'musicxml']) {
    document.querySelector(`#download-${format}`).click(); await new Promise(resolve => setImmediate(resolve));
    const last = requests.at(-1);
    assert.equal(last.url, `/api/export/${format}`);
    assert.equal(last.body.notes[0].string, 2); assert.equal(last.body.notes[0].fret, 5);
    assert.equal(last.body.bpm, 90); assert.equal(last.body.meter, '3/4');
  }
  assert.equal(downloads.length, 2);
  dom.window.close();
});

test('editing a fingering preserves pitch and is used in exported text and SVG', async () => {
  const {dom, window, document, downloads} = page();
  document.querySelector('.tab-note').dispatchEvent(new window.Event('click'));
  document.querySelector('#edit-position').value = '2';
  document.querySelector('#save-note').click();
  assert.match(document.querySelector('.tab-note').getAttribute('aria-label'), /E4，2弦5品/);
  document.querySelector('#download-svg').click();
  document.querySelector('#download-txt').click();
  assert.equal(downloads.length, 2);
  const readBlob = blob => new Promise(resolve => {
    const reader = new window.FileReader(); reader.onload = () => resolve(reader.result); reader.readAsText(blob);
  });
  const svg = await readBlob(downloads[0]);
  const parsed = new window.DOMParser().parseFromString(svg, 'image/svg+xml');
  assert.equal(parsed.querySelector('parsererror'), null, parsed.querySelector('parsererror')?.textContent);
  assert.equal(parsed.querySelectorAll('.tab-note').length, 14);
  assert.equal(parsed.querySelector('.tab-note text').textContent, '5');
  assert.match(await readBlob(downloads[1]), /E4 \/ 2 \/ 5/);
  dom.window.close();
});

test('mobile layout divides score into shorter time rows', () => {
  const {dom, window, document} = page();
  window.innerWidth = 390;
  document.querySelector('#demo').click();
  assert.equal(document.querySelectorAll('#tab-container > svg').length, 4);
  assert.equal(document.querySelectorAll('.tab-note').length, 14);
  dom.window.close();
});

test('full song selection resets and locks start, then restores segment controls', () => {
  const {dom, window, document} = page();
  const duration = document.querySelector('#clip-length'), start = document.querySelector('#clip-start');
  start.value = 45;
  duration.value = '0';
  duration.dispatchEvent(new window.Event('change'));
  assert.equal(start.value, '0');
  assert.equal(start.disabled, true);
  duration.value = '90';
  duration.dispatchEvent(new window.Event('change'));
  assert.equal(start.disabled, false);
  dom.window.close();
});

test('playback progress advances, pauses, and resumes without starting over', async () => {
  const {dom, document, clock} = page();
  const play = document.querySelector('#play-tab'), seek = document.querySelector('#play-seek');
  assert.equal(seek.max, '8');
  play.click(); await new Promise(resolve => setImmediate(resolve));
  clock.advance(2.08);
  assert.ok(Math.abs(Number(seek.value) - 2) < 0.001);
  assert.equal(document.querySelector('#play-time').textContent, '0:02 / 0:08');
  assert.equal(document.querySelector('.tab-note.active').dataset.index, '3');
  play.click();
  assert.equal(play.getAttribute('aria-label'), '播放生成的旋律');
  clock.advance(10);
  assert.equal(Number(seek.value), 2);
  const before = clock.oscillators.length;
  play.click(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(Number(seek.value), 2);
  assert.equal(clock.oscillators.length - before, 11); // Three completed notes stay silent.
  assert.equal(clock.oscillators[before].buffer, clock.oscillators[3].buffer); // Same G4 pitch, cached sound.
  clock.advance(1.08);
  assert.ok(Math.abs(Number(seek.value) - 3) < 0.001);
  dom.window.close();
});

test('seeking and speed changes preserve position and schedule only remaining sound', async () => {
  const {dom, window, document, clock} = page();
  const seek = document.querySelector('#play-seek'), play = document.querySelector('#play-tab');
  seek.value = 5; seek.dispatchEvent(new window.Event('input'));
  assert.equal(document.querySelector('#play-time').textContent, '0:05 / 0:08');
  play.click(); await new Promise(resolve => setImmediate(resolve));
  assert.equal(clock.oscillators.length, 4);
  assert.deepEqual(Array.from(clock.oscillators[0].buffer.getChannelData(0).slice(0, 200)),
    Array.from(window.GuitarSynth.samples(71, 44100).slice(0, 200))); // B4
  seek.value = 6.3; seek.dispatchEvent(new window.Event('input'));
  seek.value = 6.5; seek.dispatchEvent(new window.Event('input'));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(Number(seek.value), 6.5);
  assert.equal(play.getAttribute('aria-label'), '暂停播放');
  assert.equal(clock.oscillators.length, 5); // Only the last seek schedules, including the ongoing note.
  assert.ok(clock.oscillators.at(-1).offset > .2); // Seek into the decay, not a fresh attack.
  const speed = document.querySelector('#speed'); speed.value = '0.5';
  speed.dispatchEvent(new window.Event('change')); await new Promise(resolve => setImmediate(resolve));
  assert.equal(Number(seek.value), 6.5);
  clock.advance(2.08);
  assert.ok(Math.abs(Number(seek.value) - 7.5) < 0.001);
  clock.advance(2);
  assert.equal(Number(seek.value), 8);
  assert.equal(seek.style.getPropertyValue('--played'), '100%');
  assert.equal(play.getAttribute('aria-label'), '播放生成的旋律');
  document.querySelector('#restart-tab').click();
  assert.equal(Number(seek.value), 0);
  dom.window.close();
});
