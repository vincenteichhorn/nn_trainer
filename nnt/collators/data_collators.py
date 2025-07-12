import torch
from typing import List, Dict, Any


class PlainDataCollator:
    """
    Collator for batching samples into tensors for model input.
    Converts lists or tensors in samples to batched tensors for each input variable.

    Args:
        input_variable_names (list): List of variable names to collate from each sample.

    Example:
        collator = PlainDataCollator(['input_ids', 'attention_mask'])
        batch = collator([{'input_ids': [1,2], 'attention_mask': [1,1]}, {'input_ids': [3,4], 'attention_mask': [1,0]}])
        print(batch['input_ids'].shape)  # torch.Size([2, 2])
    """

    input_variable_names: List[str]

    def __init__(self, input_variable_names: List[str]):
        """
        Initialize the PlainDataCollator with input variable names.

        Args:
            input_variable_names (list): Names of variables to collate.
        """
        self.input_variable_names = input_variable_names

    def _sample_to_tensor(self, feature) -> torch.Tensor:
        """
        Convert a feature (e.g., input_ids, attention_mask) to a tensor if it is not already a tensor.

        Args:
            feature (Union[list, torch.Tensor]): The feature to convert.
        Returns:
            torch.Tensor: The converted feature.
        """
        return torch.tensor(feature) if not isinstance(feature, torch.Tensor) else feature

    def _stack(self, tensors: list) -> torch.Tensor:
        """
        Stack tensors to the same length along a new batch dimension.

        Args:
            tensors (list): List of tensors to stack (N).
        Returns:
            torch.Tensor: Stacked tensor of shape (N, max_length).
        """
        return torch.stack(tensors, dim=0)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a list of samples into a batch dictionary of tensors for each input variable.

        Args:
            samples (list): List of sample dictionaries.
        Returns:
            dict: Dictionary mapping variable names to batched tensors.
        """
        input_vars = set(self.input_variable_names).intersection(samples[0].keys())
        batch = {name: [] for name in input_vars}

        for name in input_vars:
            for sample in samples:
                sample_tensor = self._sample_to_tensor(sample[name])
                batch[name].append(sample_tensor)
        for name in input_vars:
            batch[name] = self._stack(batch[name])
        return batch
