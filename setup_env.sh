CUR_DIR=$(pwd)

sudo apt install -y ffmpeg

pip install pytorch_lightning
pip install einops
pip install h5py
pip install torchlibrosa
pip install braceexpand
pip install webdataset
pip install wget
pip install timm
pip install matplotlib
pip install wandb
pip install lmdb
pip install pydub
pip install fairscale
pip install xformers
pip install positional_encodings
pip install taming

if [ ! -d "taming-transformers" ]; then
  echo "📥 Cloning taming-transformers from GitHub..."
  git clone https://github.com/CompVis/taming-transformers.git
else
  echo "✅ taming-transformers directory already exists. Skipping clone."
fi
cd taming-transformers
pip install -e .
cd "$CUR_DIR"


pip install kagglehub
