/* Fractional-delay Karplus–Strong string. Local synthesis, not a recording. */
'use strict';
window.GuitarSynth = (() => {
  const cache = new Map();
  function samples(midi, sampleRate) {
    const frequency = 440 * 2 ** ((midi - 69) / 12);
    const delay = sampleRate / frequency - .5;
    const whole = Math.floor(delay), fraction = delay - whole;
    const length = Math.round(sampleRate * 5);
    const output = new Float32Array(length);
    const excitation = new Float32Array(whole + 1);
    // Seeded excitation makes the same pitch consistent on every playback.
    let seed = 12345 + midi * 171;
    for (let i = 0; i < excitation.length; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      excitation[i] = seed / 2147483648 - 1;
    }
    const mean = excitation.reduce((a, b) => a + b, 0) / excitation.length;
    for (let i = 0; i < excitation.length; i++) output[i] = excitation[i] - mean;
    let previous = 0;
    const decay = Math.exp(-3 / (frequency * 2.8));
    for (let i = excitation.length; i < length; i++) {
      const value = output[i - whole] * (1 - fraction) + output[i - whole - 1] * fraction;
      output[i] = .5 * (value + previous) * decay;
      previous = value;
    }
    let peak = .001;
    for (let i = 0; i < length; i++) peak = Math.max(peak, Math.abs(output[i]));
    for (let i = 0; i < length; i++) output[i] *= .8 / peak;
    return output;
  }
  function buffer(context, midi) {
    const key = `${context.sampleRate}:${midi}`;
    if (!cache.has(key)) {
      const data = samples(midi, context.sampleRate);
      const sound = context.createBuffer(1, data.length, context.sampleRate);
      sound.copyToChannel(data, 0); cache.set(key, sound);
    }
    return cache.get(key);
  }
  function pluck(context, destination, midi, at, duration, elapsed = 0) {
    const source = context.createBufferSource(), gain = context.createGain();
    source.buffer = buffer(context, midi);
    const remaining = Math.min(duration, Math.max(0, source.buffer.duration - elapsed));
    if (remaining <= 0) return null;
    const attack = Math.min(.006, remaining / 3), end = at + remaining;
    gain.gain.setValueAtTime(0, at);
    gain.gain.linearRampToValueAtTime(.5, at + attack);
    gain.gain.setValueAtTime(.5, Math.max(at + attack, end - .035));
    gain.gain.linearRampToValueAtTime(0, end);
    source.connect(gain); gain.connect(destination);
    // Seeking into a ringing note must not invent a new pluck at that point.
    source.start(at, elapsed, remaining); source.stop(end + .01);
    return source;
  }
  return {samples, pluck};
})();
