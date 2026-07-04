# Deej-AI Autoqueue

"Keep playing songs like this" for your local music library. Feed it one seed song, it builds a playlist of similar-sounding tracks by listening to the audio (no tags, no scrobbles) and queues them seamlessly into foobar2000 after the current track.

Fork of [teticio/Deej-AI](https://github.com/teticio/Deej-AI), which trained the model and proved the idea. This fork adds a lean pipeline for daily-driving it against your own library; upstream code is kept intact for reference.

## Features

- Runs the pretrained `speccy` model via ONNX Runtime (no TensorFlow install needed for inference)
- Vectors cached in sqlite, incremental: only new files are processed on re-runs
- Playlist chaining with lookback, noise, and multi-seed "join the dots"
- Player-agnostic m3u8 output, foobar2000 integration via [beefweb](https://github.com/hyperblast/beefweb) HTTP API
- Seamless queueing: new playlist starts after the currently playing song instead of interrupting it

## Pipeline

1. `vectorize/tf_to_onnx.py`: one-time convert of upstream's pretrained model to ONNX
2. `vectorize/1_mp3tovecs_calc.py`: mel spectrograms → model → one vector per audio slice, cached in `mp3tovecs.db`
3. `vectorize/2_tfidf_calc.py`: TF-IDF weighting of slice vectors → pickled track vectors
4. `playlist_generate.py`: seed song → cosine-similarity walk → m3u8
5. `foobar_send.py`: m3u8 → named foobar2000 playlist, queued after current track

`vectorize/vectors_cleanup.py` prunes db entries for files deleted from disk.

## Usage

```sh
python playlist_generate.py all --inputsong "D:\Music\seed.mp3" --nsongs 20 --send_foobar
```

- `vec_filename`: which pickled vector set to use (e.g. `all`)
- `--nsongs`: playlist length (default 10)
- `--noise`: randomness, 0 = strictly nearest neighbors
- `--lookback`: previous tracks factored into next pick (default 3)
- `--inputsong` with multiple songs: generate a path connecting them ("join the dots")

foobar2000 side: install beefweb (default port 8880), uncheck Preferences > Shell Integration > "Sort incoming files by" so the m3u8 order survives.

## How it works

The model maps 5-second mel-spectrogram slices to embeddings trained so that tracks co-occurring in Spotify playlists land close together (see upstream README for training details). A track is the TF-IDF weighted average of its slice vectors, which downweights generic-sounding slices. Playlist generation walks the library by cosine similarity from the seed, factoring in the last few picks so the vibe drifts naturally instead of orbiting one point.
