import os
import sys
import lmdb
import time
import numpy as np
import librosa
import os
import soundfile as sf
import io
import re
# import torchaudio
# import torchaudio.transforms as T

from datum_all_pb2 import Datum_all as Datum_out

device = 'cpu'
count = 0
total_hours = 0

def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs):
    """Load and resample audio using librosa."""
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    # src, sr = _resample_load_torchaudio(path, sample_rate, downmix_to_mono)
    return src

# Define paths
lmdb_file = './jay_val_lmdb'
toy_path = './jay_music_dataset'
lmdb_key = os.path.join(lmdb_file, 'data_key.key')

# Open LMDB environment
env = lmdb.open(lmdb_file, map_size=64 * 1024 * 1024 * 1024)
txn = env.begin(write=True)
final_keys = []

start_time = time.time()

def get_caption_from_file(filename):
    caption_filename = os.path.splitext(filename)[0]
    caption_filename += ".txt"
    if not os.path.exists(caption_filename):
        raise FileNotFoundError(f"对应的txt文件不存在: {caption_filename}")

    with open(caption_filename, "r", encoding="utf-8") as f:
        content = f.read()
        
    return content

# Walk through the dataset directory
idx = 0
for root, _, files in os.walk(toy_path):
    for file in files:
        if "._" in file or not file.endswith(".mp3"):
            continue

        if idx >= 3:
            break
        
        audio_path = os.path.join(root, file)
        key_tmp = audio_path.replace('/', '_')
        audio = _resample_load_librosa(audio_path, 16000, True)
        print(">>> audio_path=", audio_path)

        # Create a new Datum object
        datum = Datum_out()
        datum.wav_file.extend(audio)
        caption = get_caption_from_file(audio_path)
        # print(">>> caption=", caption)
        datum.caption_original = caption.encode()
        datum.caption_generated.append(caption.encode())
        datum.mos = -1

        # Write to LMDB
        txn.put(key_tmp.encode(), datum.SerializeToString())
        final_keys.append(key_tmp)

        count += 1
        total_hours += 1.00 / 60 / 10

        if count % 1 == 0:
            elapsed_time = time.time() - start_time
            print(f'{count} files written, time: {elapsed_time:.2f}s')
            txn.commit()
            txn = env.begin(write=True)
        
        idx += 1

# Finalize transaction
try:
    total_time = time.time() - start_time
    print(f'Packing completed: {count} files written, total_hours: {total_hours:.2f}, time: {total_time:.2f}s')
    txn.commit()
except:
    pass

env.close()

# Save the LMDB keys
with open(lmdb_key, 'w') as f:
    for key in final_keys:
        f.write(key + '\n')
