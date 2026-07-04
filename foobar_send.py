"""Send m3u8 to foobar2000 as named playlist via foo_beefweb HTTP API.

Usage: python foobar_send.py <playlist.m3u8>
Playlist name = m3u8 filename stem. Same-name playlist gets cleared and reused.
If music already playing, playlist is queued seamlessly instead of interrupting:
next track after current song = new playlist (skipping seed if it's the current song).
Requires foobar Preferences > Shell Integration > "Sort incoming files by" unchecked,
else foobar re-sorts the expanded m3u8.
"""
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

API = 'http://127.0.0.1:8880/api'
FOOBAR_EXE = r'C:\Program Files (x86)\foobar2000\foobar2000.exe'

def api(method, path, body=None):
  data = json.dumps(body).encode() if body is not None else None
  headers = {'Content-Type': 'application/json'} if data else {}
  req = urllib.request.Request(f'{API}{path}', data=data, method=method, headers=headers)
  try:
    with urllib.request.urlopen(req, timeout=5) as resp:
      raw = resp.read()
      return json.loads(raw) if raw else None
  except urllib.error.HTTPError as e:
    raise RuntimeError(f'beefweb {method} {path} -> HTTP {e.code}: {e.read().decode(errors="replace")}') from None

def send(m3u8_path):
  playlist_name = os.path.splitext(os.path.basename(m3u8_path))[0]

  subprocess.Popen([FOOBAR_EXE])  # cold-starts foobar, or just focuses running instance
  for _ in range(30):
    try:
      api('GET', '/player')
      break
    except OSError:
      time.sleep(0.5)
  else:
    raise RuntimeError('beefweb not reachable on :8880, is foo_beefweb installed?')

  existing = next((p for p in api('GET', '/playlists')['playlists'] if p['title'] == playlist_name), None)
  if existing:
    plref = existing['id']
    api('POST', f'/playlists/{plref}/clear')
  else:
    api('POST', f'/playlists/add?{urllib.parse.urlencode({"title": playlist_name})}')
    plref = api('GET', '/playlists')['playlists'][-1]['id']
  api('POST', '/playlists', {'current': plref})  # switch UI tab to it
  api('POST', f'/playlists/{plref}/items/add', {'items': [m3u8_path]})  # foobar expands m3u8, keeps order

  for _ in range(10):  # expansion is async, wait before playing/queueing
    if next(p for p in api('GET', '/playlists')['playlists'] if p['id'] == plref)['itemCount'] > 0:
      break
    time.sleep(0.5)

  player = api('GET', '/player?columns=%path%')['player']
  if player['playbackState'] == 'playing':
    # seamless: queue new playlist after current song instead of restarting playback
    active_path = (player.get('activeItem') or {}).get('columns', [''])[0]
    first = api('GET', f'/playlists/{plref}/items/0:1?columns=%path%')['playlistItems']['items']
    start = 1 if first and first[0]['columns'][0] == active_path else 0  # skip seed if already playing it
    api('POST', f'/playqueue/add?{urllib.parse.urlencode({"plref": plref, "itemIndex": start})}')
  else:
    api('POST', f'/player/play/{plref}/0')

if __name__ == '__main__':
  send(sys.argv[1])
