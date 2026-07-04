import argparse
import concurrent.futures
import logging
import multiprocessing as mp
import os
import pickle
import sqlite3
import sys
from time import sleep
from typing import List

import librosa
import numpy as np
import onnxruntime as ort
import torch
from audiodiffusion.audio_encoder import AudioEncoder
from tqdm import tqdm

logger = logging.getLogger(__name__)

def encode_file(model, mp3_file, dir) -> List[np.ndarray]:
  """
  Encode MP3 file as list of MP3ToVecs.

  Args:
    model (torch.nn.Module): MP3Tovec model.
    mp3_file (str): Filename of MP3.
    dir (str): Directory of MP3 files.

  Returns:
    List[np.ndarray]: List of MP3ToVec vectors
  """
  y, sr = librosa.load(os.path.join(dir, mp3_file), mono=True)
  n_mels = 96
  slice_size = 216
  if y.shape[0] < slice_size:
    return 0
  S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=n_mels, fmax=sr/2)
  x = np.ndarray(shape=(S.shape[1] // slice_size, n_mels, slice_size, 1), dtype=float)
  for slice in range(S.shape[1] // slice_size):
    log_S = librosa.power_to_db(S[:, slice * slice_size : (slice+1) * slice_size], ref=np.max)
    if np.max(log_S) - np.min(log_S) != 0:
      log_S = (log_S - np.min(log_S)) / (np.max(log_S) - np.min(log_S))
    x[slice, :, :, 0] = log_S
  return model.run(None, {"input": np.transpose(x, (0, 3, 1, 2)).astype(np.float32)})[0]
  # return model.encode([os.path.join(dir, mp3_file)], pool=None)[0].cpu().numpy()

def init_session(model_path):
  EP_list = ['CPUExecutionProvider']
  sess = ort.InferenceSession(model_path, providers=EP_list)
  return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
  def __init__(self, model_path):
    self.model_path = model_path
    self.sess = init_session(self.model_path)

  def run(self, *args):
    return self.sess.run(*args)

  def __getstate__(self):
    return {'model_path': self.model_path}

  def __setstate__(self, values):
    self.model_path = values['model_path']
    self.sess = init_session(self.model_path)

def main():
  """
  Main function for the mp3tovecs_calc script.

  Encodes a directory of MP3 files as a dictionary of lists of MP3ToVec vectors

  Ags:
    --max_workers (int): Maximum number of cores to use. Default is the number of cores on the machine.
    --mp3tovec_model_file (str): Path to the MP3ToVec model file. Default is "models/mp3tovec.ckpt".
    --mp3tovecs_file (str): Path to the output file where the MP3ToVec vectors will be saved. Default is "models/mp3tovecs.p".
    --mp3s_dir (str): Path to the directory containing the MP3 files to be encoded. Default is "previews".
  """
  os.chdir(os.path.dirname(__file__))
  # audioread needs ffmpeg for m4a/aac, not on system PATH; inherited by worker processes
  os.environ["PATH"] = r"C:\N\software\media\video\ffmpeg" + os.pathsep + os.environ["PATH"]
  # windows console defaults to cp1252, japanese filenames in prints would crash
  sys.stdout.reconfigure(encoding="utf-8", errors="replace")

  parser = argparse.ArgumentParser()
  parser.add_argument("--max_workers", type=int, default=os.cpu_count() if os.cpu_count() is not None else 1, help="Maximum number of cores to use")
  parser.add_argument("--mp3tovec_model_file", type=str, default="../models/speccy_model_20220202.onnx", help="MP3ToVec model file")
  parser.add_argument("--mp3tovecs_file", type=str, default="../models/mp3tovecs.db", help="Mp3ToVecs output file")
  parser.add_argument("--mp3s_dir", type=str, action='append', help="Directory of MP3 files (Argument can be specified multiple times)")
  parser.add_argument("--save_every", type=int, default=100, help="Save MP3ToVecs every N MP3s")
  # args = parser.parse_args()
  args = parser.parse_args(['--max_workers', '20'
                ,'--mp3s_dir', 'D:\\media\\music\\Japanese'
                ,'--mp3s_dir', 'D:\\media\\music\\Korean'
                ,'--mp3s_dir', 'D:\\media\\music\\English'
                ])

  # model = AudioEncoder()
  # model.load_state_dict(
  #   {
  #     k.replace("model.", ""): v
  #     for k, v in torch.load(args.mp3tovec_model_file)["state_dict"].items()
  #   }
  # )
  model = PickableInferenceSession(args.mp3tovec_model_file)

  con = sqlite3.connect(args.mp3tovecs_file)
  cur = con.cursor()
  cur.execute("""CREATE TABLE IF NOT EXISTS vectors
          (fullpath text PRIMARY KEY, vector blob)""")
  cur.execute("""CREATE UNIQUE INDEX IF NOT EXISTS "index_vectors" ON "vectors" (
	"fullpath"	ASC
  )""")

  formats = ["mp3","wav","m4a","flac","opus","alac","ogg","oga"]
  formats = set([f'.{format}' for format in formats])

  mp3_files = []
  for mp3_dir in args.mp3s_dir:
    mp3_files.extend([
      os.path.join(root, file)
      for root, _, files in os.walk(mp3_dir)
      for file in files
      if file[file.rfind(".") :].lower() in formats
      # Filter from DB to skip existing vectors
      and cur.execute("""SELECT fullpath FROM vectors WHERE fullpath = (?)""", [os.path.join(root, file)]).fetchone() is None
    ])
 
  # To enable GPU, model and inputs must be copied to GPU
  # PyTorch/ONNX is better than TensorFlow in memory allocation
  # Parallel-GPU inferencing eats ~1.5GB RAM/process for not much speedup
  # model.cuda()
  torch.multiprocessing.set_start_method("spawn")
  with concurrent.futures.ProcessPoolExecutor(
    max_workers=args.max_workers
  ) as executor:
    futures = {
      executor.submit(encode_file, model, mp3_file, ""): mp3_file
      for mp3_file in tqdm(mp3_files, desc="Setting up jobs")
      if sleep(1e-4) is None
    }
    for i, future in enumerate(
      tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Encoding MP3s",
      )
    ):
      mp3_file = futures[future]
      try:
        cur.execute("INSERT OR IGNORE INTO vectors VALUES (?, ?)", (mp3_file, future.result()))
        if (i + 1) % args.save_every == 0:
          con.commit()
      except KeyboardInterrupt:
        break
      except Exception as e:
        print(f"Skipping {mp3_file}: {e}")

  con.commit()
  con.close()

if __name__ == "__main__":
  main()
