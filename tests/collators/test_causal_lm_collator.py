import pytest
import torch
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM


@pytest.fixture
def sample_batch():
    return [
        {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
        {"input_ids": [5, 6, 7], "attention_mask": [1, 1, 1]},
        {"input_ids": [8, 9], "attention_mask": [1, 1]},
    ]


@pytest.fixture
def tokenizer():
    class MockTokenizer:
        def __init__(self):
            self.pad_token_id = 12800

    return MockTokenizer()


def test_causal_lm_collator_padding(sample_batch, tokenizer):
    collator = DataCollatorForCausalLM(tokenizer)
    batch = collator(sample_batch)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == (3, 16)
    assert batch["attention_mask"].shape == (3, 16)
    assert batch["labels"].shape == (3, 16)


def test_different_pad_token_id(sample_batch, tokenizer):
    tokenizer.pad_token_id = 99
    collator = DataCollatorForCausalLM(tokenizer)
    batch = collator(sample_batch)
    assert (batch["input_ids"] == 99).sum() > 0
    assert (batch["input_ids"] == 99).sum() > 0
