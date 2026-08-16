import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_loaders.basic_data_loader import TRAINING_DATASETS_DIR, BasicDataLoader


@pytest.fixture
def sample_dataset_file():
    """Writes a small text file into TRAINING_DATASETS_DIR and removes it afterward."""
    created_dir = not TRAINING_DATASETS_DIR.exists()
    TRAINING_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    file_name = "test_basic_data_loader_sample.txt"
    file_path = TRAINING_DATASETS_DIR / file_name
    file_path.write_text("hello world " * 200)

    yield file_name

    file_path.unlink()
    if created_dir:
        TRAINING_DATASETS_DIR.rmdir()


def test_init_tokenizes_file_and_initializes_cursor(sample_dataset_file):
    loader = BasicDataLoader(sample_dataset_file, batch_size=2, context_size=8)

    assert isinstance(loader.tokens, torch.Tensor)
    assert loader.tokens.dtype == torch.long
    assert len(loader.tokens) > 0
    assert loader.tokens_per_batch == 2 * 8
    assert loader.current_index == 0


def test_init_rejects_file_name_that_escapes_training_datasets_dir():
    with pytest.raises(ValueError, match="must resolve to a path inside"):
        BasicDataLoader("../gpt.py", batch_size=1, context_size=1)


def test_init_rejects_file_with_too_few_tokens_for_one_batch(sample_dataset_file):
    with pytest.raises(ValueError, match="tokens are needed for a single batch"):
        BasicDataLoader(sample_dataset_file, batch_size=1000, context_size=1000)


def test_get_next_batch_returns_expected_shapes(sample_dataset_file):
    loader = BasicDataLoader(sample_dataset_file, batch_size=4, context_size=16)

    inputs, outputs = loader.get_next_batch()

    assert inputs.shape == (4, 16)
    assert outputs.shape == (4, 16)


def test_get_next_batch_outputs_are_inputs_shifted_by_one_token(sample_dataset_file):
    loader = BasicDataLoader(sample_dataset_file, batch_size=4, context_size=16)

    inputs, outputs = loader.get_next_batch()

    # Flattened in row-major order, outputs is inputs shifted right by exactly one token.
    assert torch.equal(inputs.flatten()[1:], outputs.flatten()[:-1])


def test_get_next_batch_advances_current_index_by_tokens_per_batch(sample_dataset_file):
    loader = BasicDataLoader(sample_dataset_file, batch_size=2, context_size=8)

    loader.get_next_batch()

    assert loader.current_index == loader.tokens_per_batch


def test_get_next_batch_wraps_around_when_tokens_run_out(sample_dataset_file):
    loader = BasicDataLoader(sample_dataset_file, batch_size=1, context_size=4)
    # Replace with a small, known token sequence so the wraparound point is deterministic:
    # exactly enough tokens for two batches (tokens_per_batch=4) before it must loop back.
    loader.tokens = torch.arange(9)

    first_inputs, first_outputs = loader.get_next_batch()
    loader.get_next_batch()
    assert loader.current_index == 0

    third_inputs, third_outputs = loader.get_next_batch()
    assert torch.equal(third_inputs, first_inputs)
    assert torch.equal(third_outputs, first_outputs)
