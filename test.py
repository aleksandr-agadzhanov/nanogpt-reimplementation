import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "minbpe_reimplementation"))
from minbpe_tokenizers import RegexTokenizer

tokenizer = RegexTokenizer("fineweb_edu_100mb_16384.pkl")

shard_path = Path("training_shards/fineweb_edu_10bt_val_0000.npy")
tokens = np.load(shard_path).tolist()

decoded_text = tokenizer.decode(tokens)

Path("decoded.txt").write_text(decoded_text)
