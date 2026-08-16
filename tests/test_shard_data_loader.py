import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import functools
import operator

from data_loaders.shard_data_loader import END_OF_TEXT_TOKEN, ShardDataLoader


def _make_document(marker: int, length: int) -> list[int]:
    """Builds a document of `length` tokens (all equal to `marker`), prefixed with
    the end-of-text delimiter every real document starts with (see
    load_training_dataset.py)."""
    return [END_OF_TEXT_TOKEN] + [marker] * length


def _write_shard(folder: Path, file_name: str, documents: list[list[int]]) -> Path:
    tokens = functools.reduce(operator.iadd, documents, [])
    shard_path = folder / file_name
    np.save(shard_path, np.array(tokens, dtype=np.uint16))
    return shard_path


@pytest.fixture
def shard_folder(tmp_path):
    """Writes two 'train' shards (4 documents of 8 tokens each) and one 'val' shard
    into tmp_path."""
    _write_shard(
        tmp_path, "shard_train_000000.npy", [_make_document(m, 8) for m in range(4)]
    )
    _write_shard(
        tmp_path,
        "shard_train_000001.npy",
        [_make_document(m, 8) for m in range(100, 104)],
    )
    _write_shard(
        tmp_path,
        "shard_val_000000.npy",
        [_make_document(m, 8) for m in range(200, 204)],
    )
    return tmp_path


def test_init_finds_and_sorts_shards_matching_split(shard_folder):
    loader = ShardDataLoader(
        str(shard_folder),
        batch_size=1,
        context_size=2,
        process_index=0,
        num_processes=1,
        split="train",
    )

    assert [os.path.basename(p) for p in loader.shard_paths] == [
        "shard_train_000000.npy",
        "shard_train_000001.npy",
    ]
    assert loader.shard_order == [0, 1]


def test_init_raises_when_no_shards_match_split(tmp_path):
    _write_shard(tmp_path, "shard_val_000000.npy", [_make_document(0, 5)])

    with pytest.raises(ValueError, match="no shard files matching split"):
        ShardDataLoader(
            str(tmp_path),
            batch_size=1,
            context_size=2,
            process_index=0,
            num_processes=1,
            split="train",
        )


def test_load_shard_raises_when_shard_has_too_few_tokens(tmp_path):
    _write_shard(tmp_path, "shard_train_000000.npy", [_make_document(0, 1)])

    with pytest.raises(ValueError, match="tokens are needed for a single batch"):
        ShardDataLoader(
            str(tmp_path),
            batch_size=10,
            context_size=10,
            process_index=0,
            num_processes=1,
            split="train",
        )


def test_reset_rewinds_to_first_shard_and_process_offset(shard_folder):
    loader = ShardDataLoader(
        str(shard_folder),
        batch_size=1,
        context_size=2,
        process_index=1,
        num_processes=2,
        split="train",
    )
    loader.get_next_batch()

    loader.reset()

    assert loader.current_shard_index == 0
    assert loader.current_index == loader.tokens_per_microbatch * loader.process_index


def test_get_next_batch_returns_expected_shapes(shard_folder):
    loader = ShardDataLoader(
        str(shard_folder),
        batch_size=2,
        context_size=4,
        process_index=0,
        num_processes=1,
        split="train",
    )

    inputs, outputs = loader.get_next_batch()

    assert inputs.shape == (2, 4)
    assert outputs.shape == (2, 4)


def test_get_next_batch_outputs_are_inputs_shifted_by_one_token(shard_folder):
    loader = ShardDataLoader(
        str(shard_folder),
        batch_size=2,
        context_size=4,
        process_index=0,
        num_processes=1,
        split="train",
    )

    inputs, outputs = loader.get_next_batch()

    assert torch.equal(inputs.flatten()[1:], outputs.flatten()[:-1])


def test_get_next_batch_slices_process_specific_offset_within_same_round(shard_folder):
    loader0 = ShardDataLoader(
        str(shard_folder),
        batch_size=1,
        context_size=4,
        process_index=0,
        num_processes=2,
        split="train",
        seed=3,
    )
    loader1 = ShardDataLoader(
        str(shard_folder),
        batch_size=1,
        context_size=4,
        process_index=1,
        num_processes=2,
        split="train",
        seed=3,
    )
    # Same seed => identical document shuffle, so both processes see the same shard content.
    assert torch.equal(loader0.shard_tokens, loader1.shard_tokens)
    shard_tokens = loader0.shard_tokens

    inputs0, outputs0 = loader0.get_next_batch()
    inputs1, outputs1 = loader1.get_next_batch()

    expected0 = shard_tokens[0:5]
    expected1 = shard_tokens[4:9]
    assert torch.equal(inputs0.flatten(), expected0[:-1])
    assert torch.equal(outputs0.flatten(), expected0[1:])
    assert torch.equal(inputs1.flatten(), expected1[:-1])
    assert torch.equal(outputs1.flatten(), expected1[1:])


def test_seed_and_shard_order_change_only_after_full_pass(tmp_path):
    _write_shard(tmp_path, "shard_train_000000.npy", [_make_document(0, 2)])
    _write_shard(tmp_path, "shard_train_000001.npy", [_make_document(1, 2)])

    loader = ShardDataLoader(
        str(tmp_path),
        batch_size=1,
        context_size=2,
        process_index=0,
        num_processes=1,
        split="train",
        seed=10,
    )
    initial_seed = loader.seed
    assert loader.current_shard_index == 0

    loader.get_next_batch()  # exhausts shard 0 -> moves to shard 1 (no full pass yet)
    assert loader.current_shard_index == 1
    assert loader.seed == initial_seed

    loader.get_next_batch()  # exhausts shard 1 -> wraps to shard 0 (full pass completed)
    assert loader.current_shard_index == 0
    assert loader.seed == initial_seed + 1
    assert sorted(loader.shard_order) == [0, 1]


def test_multiple_processes_stay_synchronized_across_full_passes(tmp_path):
    shard_specs = [
        [(m, 5) for m in range(9)],
        [(m, 20) for m in range(100, 104)],
        [(m, 3) for m in range(200, 230)],
    ]
    for idx, spec in enumerate(shard_specs):
        _write_shard(
            tmp_path,
            f"shard_train_{idx:06d}.npy",
            [_make_document(marker, length) for marker, length in spec],
        )

    num_processes = 3
    loaders = []
    for process_index in range(num_processes):
        # Deliberately different global RNG state per simulated process: this class
        # must never depend on the global RNG for cross-process determinism to hold.
        torch.manual_seed(process_index * 111)
        loaders.append(
            ShardDataLoader(
                str(tmp_path),
                batch_size=1,
                context_size=4,
                process_index=process_index,
                num_processes=num_processes,
                split="train",
                seed=7,
            )
        )
    assert all(loader.shard_order == loaders[0].shard_order for loader in loaders)

    seen_full_pass_wraps = 0
    prev_shard_index = loaders[0].current_shard_index
    for round_num in range(40):
        for process_index, loader in enumerate(loaders):
            torch.manual_seed(round_num * 37 + process_index * 991)
            loader.get_next_batch()

        assert all(
            loader.current_shard_index == loaders[0].current_shard_index
            for loader in loaders
        )
        assert all(loader.shard_order == loaders[0].shard_order for loader in loaders)
        assert all(loader.seed == loaders[0].seed for loader in loaders)
        assert all(
            torch.equal(loader.shard_tokens, loaders[0].shard_tokens)
            for loader in loaders
        )

        if loaders[0].current_shard_index == 0 and prev_shard_index != 0:
            seen_full_pass_wraps += 1
        prev_shard_index = loaders[0].current_shard_index

    assert seen_full_pass_wraps >= 2


def test_derive_seed_combines_seed_and_part(shard_folder):
    loader = ShardDataLoader(
        str(shard_folder),
        batch_size=1,
        context_size=2,
        process_index=0,
        num_processes=1,
        split="train",
        seed=5,
    )

    assert loader._derive_seed(0) == 5 * 1_000_003 + 0
    assert loader._derive_seed(3) == 5 * 1_000_003 + 3
    assert loader._derive_seed(3) != loader._derive_seed(4)


def test_permute_documents_reorders_but_preserves_document_content():
    documents = [_make_document(marker, 3) for marker in range(5)]
    tokens = torch.tensor(functools.reduce(operator.iadd, documents, []), dtype=torch.long)

    generator = torch.Generator().manual_seed(123)
    permuted = ShardDataLoader._permute_documents(tokens, generator)

    assert permuted.shape == tokens.shape
    assert sorted(permuted.tolist()) == sorted(tokens.tolist())

    # Matches independently re-deriving the same permutation from an identically-seeded generator.
    expected_generator = torch.Generator().manual_seed(123)
    expected_order = torch.randperm(
        len(documents), generator=expected_generator
    ).tolist()
    expected_tokens = torch.cat(
        [torch.tensor(documents[i], dtype=torch.long) for i in expected_order]
    )
    assert torch.equal(permuted, expected_tokens)


def test_load_tokens_from_shard_loads_uint16_npy_as_long_tensor(tmp_path):
    shard_path = _write_shard(
        tmp_path, "shard_train_000000.npy", [_make_document(7, 5)]
    )

    tokens = ShardDataLoader._load_tokens_from_shard(str(shard_path))

    assert tokens.dtype == torch.long
    assert tokens[0].item() == END_OF_TEXT_TOKEN
    assert tokens[1:].tolist() == [7] * 5
