import os
import lmdb
import time
import numpy as np
import librosa
import os
import soundfile as sf
import io
import re

from datum_all_pb2 import Datum_all as Datum_out

device = 'cpu'
count = 0
total_hours = 0

# Define paths
lmdb_file = './toy_lmdb'
toy_path = './audioset'
lmdb_key = os.path.join(lmdb_file, 'data_key.key')

# Open LMDB environment
env = lmdb.open(lmdb_file, map_size=64 * 1024 * 1024 * 1024)
txn = env.begin(write=True)
final_keys = []

def _resample_load_librosa(path: str, sample_rate: int, downmix_to_mono: bool, **kwargs):
    """Load and resample audio using librosa."""
    src, sr = librosa.load(path, sr=sample_rate, mono=downmix_to_mono, **kwargs)
    return src

start_time = time.time()

def get_caption_from_file(filename):
    name = os.path.splitext(filename)[0]
    name = re.sub(r'-\d+$', '', name)
    name = name.replace('-', ' ')
    return name

# Walk through the dataset directory
for root, _, files in os.walk(toy_path):
    for file in files:
        audio_path = os.path.join(root, file)
        key_tmp = audio_path.replace('/', '_')
        audio = _resample_load_librosa(audio_path, 16000, True)
        
        # Create a new Datum object
        datum = Datum_out()
        datum.wav_file.extend(audio)
        caption = get_caption_from_file(file)
        print(">>> caption=", caption)
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
