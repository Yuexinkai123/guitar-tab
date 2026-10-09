This directory contains the unchanged ICASSP 2022 ONNX weights distributed with
Spotify Basic Pitch 0.4.0 and four extracted decoder functions from
`basic_pitch/note_creation.py`, under Apache-2.0. See LICENSE.

Source: https://github.com/spotify/basic-pitch/tree/v0.4.0
Model SHA256: 2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec

Modifications: decoder functions extracted without TensorFlow/MIDI/sonification
imports, constants inlined. The audio preprocessing in app/polyphonic.py follows
the upstream overlap and unwrap procedure. No Songscription code or model is used.
