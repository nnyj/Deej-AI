import argparse
import logging
import os
import pickle
import re
import sys
import time

import mutagen
import numpy as np

logger = logging.getLogger(__name__)

NAME_SEPARATORS = [' - ', ' & ', ' ft. ', ' feat. ', ', ']  # mirrors foobar sort algorithm in tech.audio.md

def read_tags(track_path):
  """Normalize m4a freeform / mp3 TXXX / standard tags to plain uppercase names."""
  tags = {}
  try:
    f = mutagen.File(track_path)
  except Exception:
    return tags  # corrupt/unreadable file, caller falls back to filename
  if f is None or not f.tags:
    return tags
  standard = {'©nam': 'TITLE', '©ART': 'ARTIST', 'aART': 'ALBUM ARTIST',
              'TIT2': 'TITLE', 'TPE1': 'ARTIST', 'TPE2': 'ALBUM ARTIST'}
  for key, value in f.items():
    if key.startswith('----:com.apple.iTunes:'):
      tags[key.rsplit(':', 1)[1].upper()] = bytes(value[0]).decode('utf-8', 'replace')
    elif key.startswith('TXXX:'):
      tags[key[5:].upper()] = str(value)
    elif key in standard:
      tags[standard[key]] = str(value[0]) if isinstance(value, list) else str(value)
  return tags

def playlist_name(seed_path):
  """<source or artist> - <title>, per tagging convention in tech.audio.md"""
  tags = read_tags(seed_path)
  first = lambda *names: next((tags[n] for n in names if tags.get(n)), '')
  prefix = first('SOURCE ENGLISH', 'SOURCE', 'ALBUM ARTIST ROMANIZED', 'ALBUM ARTIST', 'ARTIST ROMANIZED', 'ARTIST')
  for sep in NAME_SEPARATORS:
    prefix = prefix.split(sep)[0]
  title = first('TITLE ENGLISH', 'TITLE ROMANIZED', 'TITLE') or os.path.splitext(os.path.basename(seed_path))[0]
  name = f'{prefix} - {title}' if prefix else title
  return re.sub(r'\s+', ' ', re.sub(r'[\\/:*?"<>|]', ' ', name)).strip(' .')

def most_similar(positive=[], negative=[], topn=5, noise=0, vector_mode=False):
  if isinstance(positive, str):
    positive = [positive] # broadcast to list
  if isinstance(negative, str):
    negative = [negative] # broadcast to list
  
  mp3_vec_i = np.sum([mp3tovec[i] for i in positive] + [-mp3tovec[i] for i in negative], axis=0) if not vector_mode else np.sum([i for i in positive] + [-i for i in negative], axis=0)

  mp3_vec_i += np.random.normal(0, noise * np.linalg.norm(mp3_vec_i), len(mp3_vec_i))
  mp3_vec_i_norm = np.linalg.norm(mp3_vec_i) # precalculate norms for mp3_vec_i
  similar = []
  mp3tovec_array = np.array(list(mp3tovec.values()))
  dot_product = np.dot(mp3tovec_array, mp3_vec_i)
  cos_proximity_array = dot_product / (mp3_vec_i_norm * mp3_vec_j_norms)
  if not vector_mode:
    similar = [(track_j, cos_proximity) for track_j, cos_proximity in zip(mp3tovec.keys(), cos_proximity_array)
             if track_j not in positive and track_j not in negative]
  else:
     similar = [(track_j, cos_proximity) for track_j, cos_proximity in zip(mp3tovec.keys(), cos_proximity_array)]
  return sorted(similar, key=lambda x:-x[1])[:topn]

def make_playlist(seed_tracks, size=10, lookback=3, noise=0):
  max_tries = size # was 10
  playlist = seed_tracks
  while len(playlist) < size:
    similar = most_similar(positive=playlist[-lookback:], topn=max_tries, noise=noise)
    # candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
    candidates = [candidate[0] for candidate in similar if candidate[0] not in playlist] # Avoid repeating
    for candidate in candidates:
      if not candidate in playlist:
        playlist.append(candidate)
        break
  return playlist

def join_the_dots(tracks, n=10, noise=0): # create a musical journey between given track "waypoints"
  max_tries = n # was 10
  playlist = []
  end = start = tracks[0]
  start_vec = mp3tovec[start]
  for end in tracks[1:]:
    end_vec = mp3tovec[end]
    playlist.append(start)
    for i in range(n):
      # print(f'{playlist[-1]}')
      similar = most_similar(positive=[(n-i+1)/n * start_vec + (i+1)/n * end_vec], topn=max_tries, noise=noise, vector_mode=True)
      # candidates = [candidate[0] for candidate in similar if candidate[0] != playlist[-1]]
      candidates = [candidate[0] for candidate in similar if candidate[0] not in playlist and candidate[0] != playlist[-1]] # Avoid repeating
      for candidate in candidates:
        if not candidate in playlist and candidate != end:
          playlist.append(candidate)
          break
    start = end
    start_vec = end_vec
  playlist.append(end)
  return playlist

def tracks_to_m3u(fileout, tracks):
  """
  using absolute path
  """
  if os.path.dirname(fileout):
    os.makedirs(os.path.dirname(fileout), exist_ok=True)
  with open(fileout, 'w', encoding="utf-8") as f:
    for item in tracks:
      f.write(item + "\n")

def main(arg_str=None):
  global mp3tovec, mp3_vec_j_norms
  os.chdir(os.path.dirname(__file__))
  
  parser = argparse.ArgumentParser()
  parser.add_argument('vec_filename', type=str, help='Filename (without extension) of pickled MP3ToVecs')
  parser.add_argument('--playlist', type=str, default='_default.m3u8', help='Location of output playlist file')
  parser.add_argument('--playlist_dir', type=str, help='Output dir; filename derived from seed song tags, overrides --playlist')
  parser.add_argument('--send_foobar', action='store_true', help='Load result into foobar as named playlist (foo_beefweb)')
  parser.add_argument('--inputsong', type=str, nargs='+', required=True, help="Song to start playlist\nIf more than 1 given, join the dots.")
  parser.add_argument("--nsongs", type=int, default=10, help="Number of songs in the playlist")
  parser.add_argument("--noise", type=int, default=0, help="Amount of randomness to throw in the mix")
  parser.add_argument("--lookback", type=int, default=3, help="Number of previous tracks to take into account")
  args = parser.parse_args() if arg_str is None else parser.parse_args(arg_str)
  inputsong = args.inputsong

  mp3tovec = pickle.load(open(f'mp3tovecs/{args.vec_filename}.p', 'rb'))
  print(f'mp3tovec contains {len(mp3tovec)} songs')
  mp3_vec_j_norms = np.array([np.linalg.norm(mp3tovec[track_j]) for track_j in mp3tovec]) # precalculate norms for mp3_vec_j

  playlist_path = os.path.join(args.playlist_dir, playlist_name(inputsong[0]) + '.m3u8') if args.playlist_dir else args.playlist
  print(f'Outfile playlist: {playlist_path}')
  print(f'Input song selected: ')
  print("\n".join(inputsong))
  print(f'Requested {args.nsongs} songs')

  start = time.time()
  playlist = []
  if len(inputsong) == 1:
    # Limit playlist size to len(mp3tovec) to avoid repeating songs
    playlist = make_playlist(inputsong, size=min(args.nsongs,len(mp3tovec)), noise=args.noise, lookback=args.lookback)
  elif len(inputsong) > 1:
    n = (args.nsongs-len(inputsong)) // (len(inputsong)-1) # Evenly spread spacing to fit nsongs
    playlist = join_the_dots(inputsong, n=n, noise=args.noise)
  end = time.time()
  print(f'Total time: {end - start}s, avg time: {(end - start)/args.nsongs}s')
  tracks_to_m3u(playlist_path, playlist)
  if args.send_foobar:
    import foobar_send
    foobar_send.send(os.path.abspath(playlist_path))

if __name__ == '__main__':
  windowless = sys.stdout is None  # pythonw: no console, print() would crash
  if windowless:
    sys.stdout = sys.stderr = open(os.devnull, 'w', encoding='utf-8')
  try:
    main()
  except Exception:
    if not windowless:
      raise
    import ctypes
    import traceback
    ctypes.windll.user32.MessageBoxW(0, traceback.format_exc(), 'Deej-AI playlist', 0x10)
    sys.exit(1)