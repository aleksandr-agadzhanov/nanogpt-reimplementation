# training_shards

This folder is where tokenized training data shards are written to and read from. Its
contents are **not committed to the repository** due to their large size (the full
`sample-10BT` split of `HuggingFaceFW/fineweb-edu`, tokenized, is tens of gigabytes on
disk) - only this README is tracked, so the folder still shows up in the repo.

## How shards get here

Run [`load_training_dataset.py`](../load_training_dataset.py) from the repo root:

```bash
python load_training_dataset.py
```

This downloads `HuggingFaceFW/fineweb-edu` (`sample-10BT`), tokenizes it in parallel
using the custom `RegexTokenizer` (vendored via the `minbpe_reimplementation` git
submodule), and writes fixed-size `.npy` shards here.

## File naming

Each shard is saved as:

```
fineweb_edu_10bt_{split}_{shard_index:04d}.npy
```

- `split` is `val` for shard `0000` and `train` for every other shard - the first shard
  is always reserved for validation.
- Each shard holds up to `MAX_TOKENS_PER_SHARD` (100,000,000) `uint16` token ids.
- Documents are never split across two shards, so `ShardDataLoader` can shuffle whole
  shards without ever separating a document from part of itself.
- Every document is prefixed with the `<|endoftext|>` special token id, which delimits
  document boundaries within a shard.

## Regenerating vs. re-downloading

Regenerating these shards requires network access (to download the dataset from
HuggingFace) and can take a long time depending on `num_workers` and available
bandwidth/compute - there's no need to do this unless you're training from scratch or
the shards have been deleted/corrupted.
