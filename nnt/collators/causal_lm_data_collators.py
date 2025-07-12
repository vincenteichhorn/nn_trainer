import torch
from typing import Any, List, Dict
from transformers import PreTrainedTokenizer


class DataCollatorForCausalLM:
    """
    Collator for batching and padding samples for causal language modeling tasks.
    Handles input_ids, attention_mask, and labels, with support for left/right padding and padding to multiples.

    Args:
        tokenizer: Tokenizer object with pad_token_id.

    Example:
        collator = DataCollatorForCausalLM(tokenizer)
        batch = collator([
            {'input_ids': [1,2,3], 'labels': [1,2,3]},
            {'input_ids': [4,5], 'labels': [4,5]}
        ])
        print(batch['input_ids'].shape)  # torch.Size([2, padded_length])
    """

    tokenizer: PreTrainedTokenizer
    padding_token: int
    loss_mask_token: int

    def __init__(self, tokenizer: PreTrainedTokenizer):
        """
        Initialize the DataCollatorForCausalLM with a tokenizer.

        Args:
            tokenizer: Tokenizer object with pad_token_id.
        """
        self.tokenizer = tokenizer
        self.padding_token = tokenizer.pad_token_id
        self.loss_mask_token = -100

    def _stack_and_pad(
        self,
        tensors: list,
        padding_side: str = "right",
        pad_token: int = None,
        pad_to_multiple_of: int = 16,
    ) -> torch.Tensor:
        """
        Stack and pad tensors to the same length, optionally padding to a multiple of a given value.

        Args:
            tensors (list): List of tensors to stack and pad (N).
            padding_side (str): "right" or "left".
            pad_token (int): Padding token id.
            pad_to_multiple_of (int): Pad the sequence length to a multiple of this value.
        Returns:
            torch.Tensor: Stacked and padded tensor of shape (N, max_length).
        """
        if pad_token is None:
            pad_token = self.padding_token

        max_length = max(tensor.size(0) for tensor in tensors)

        if pad_to_multiple_of is not None:
            max_length = (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of * pad_to_multiple_of

        padded_tensors = []
        for tensor in tensors:
            padding = (0, max_length - tensor.size(0)) if padding_side == "right" else (max_length - tensor.size(0), 0)
            padded_tensors.append(torch.nn.functional.pad(tensor, padding, value=pad_token))

        return torch.stack(padded_tensors, dim=0)

    def _feature_to_tensor(self, feature) -> torch.Tensor:
        """
        Convert a feature (input_ids, attention_mask) to a tensor if it is not already a tensor.

        Args:
            feature (Union[list, torch.Tensor]): The feature to convert.
        Returns:
            torch.Tensor: The converted feature.
        """
        return torch.tensor(feature) if not isinstance(feature, torch.Tensor) else feature

    def __call__(self, samples: List[Dict[str, Any]], padding_side: str = "left") -> Dict[str, torch.Tensor]:
        """
        Collate, stack, and pad a list of samples into a batch dictionary for causal LM training.

        Args:
            samples (list): List of sample dictionaries.
            padding_side (str): "right" or "left".
        Returns:
            dict: Dictionary of stacked and padded features (input_ids, attention_mask, labels).
        """
        input_ids = [self._feature_to_tensor(sample["input_ids"]) for sample in samples]
        labels = [
            (
                self._feature_to_tensor(sample["labels"])
                if "labels" in sample
                else self._feature_to_tensor(sample["input_ids"])
            )
            for sample in samples
        ]
        attention_masks = [torch.ones_like(inp) for inp in input_ids]
        batch = {
            "input_ids": self._stack_and_pad(input_ids, padding_side=padding_side),
            "attention_mask": self._stack_and_pad(attention_masks, padding_side=padding_side, pad_token=0),
            "labels": self._stack_and_pad(labels, pad_token=self.loss_mask_token, padding_side=padding_side),
        }
        return batch
