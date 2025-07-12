import pytest
import torch
from nnt.collators.data_collators import PlainDataCollator


def test_sample_to_tensor_with_list():
    collator = PlainDataCollator(["input_ids"])
    feature = [1, 2, 3]
    tensor = collator._sample_to_tensor(feature)
    assert isinstance(tensor, torch.Tensor)
    assert torch.equal(tensor, torch.tensor([1, 2, 3]))


def test_sample_to_tensor_with_tensor():
    collator = PlainDataCollator(["input_ids"])
    feature = torch.tensor([1, 2, 3])
    tensor = collator._sample_to_tensor(feature)
    assert tensor is feature


def test_stack_tensors():
    collator = PlainDataCollator(["input_ids"])
    tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    stacked = collator._stack(tensors)
    assert stacked.shape == (2, 2)
    assert torch.equal(stacked, torch.tensor([[1, 2], [3, 4]]))


def test_call_with_single_variable():
    collator = PlainDataCollator(["input_ids"])
    samples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
    batch = collator(samples)
    assert "input_ids" in batch
    assert batch["input_ids"].shape == (2, 3)
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2, 3], [4, 5, 6]]))


def test_call_with_multiple_variables():
    collator = PlainDataCollator(["input_ids", "attention_mask"])
    samples = [{"input_ids": [1, 2], "attention_mask": [1, 0]}, {"input_ids": [3, 4], "attention_mask": [0, 1]}]
    batch = collator(samples)
    assert set(batch.keys()) == {"input_ids", "attention_mask"}
    assert batch["input_ids"].shape == (2, 2)
    assert batch["attention_mask"].shape == (2, 2)
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 4]]))
    assert torch.equal(batch["attention_mask"], torch.tensor([[1, 0], [0, 1]]))


def test_call_ignores_extra_keys():
    collator = PlainDataCollator(["input_ids"])
    samples = [{"input_ids": [1, 2], "extra": [9, 9]}, {"input_ids": [3, 4], "extra": [8, 8]}]
    batch = collator(samples)
    assert "input_ids" in batch
    assert "extra" not in batch
    assert batch["input_ids"].shape == (2, 2)


def test_call_with_tensor_inputs():
    collator = PlainDataCollator(["input_ids"])
    samples = [{"input_ids": torch.tensor([1, 2])}, {"input_ids": torch.tensor([3, 4])}]
    batch = collator(samples)
    assert batch["input_ids"].shape == (2, 2)
    assert torch.equal(batch["input_ids"], torch.tensor([[1, 2], [3, 4]]))
