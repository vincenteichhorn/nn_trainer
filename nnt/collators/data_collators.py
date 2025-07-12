import torch


class PlainDataCollator:

    def __init__(self, input_variable_names):
        self.input_variable_names = input_variable_names

    def _sample_to_tensor(self, feature):
        """
        convert a feature (input_ids, attention_mask) to a tensor if it is not already a tensor
        Args:
            feature (Union[list, torch.Tensor]): the feature to convert
        Returns:
            torch.Tensor: the converted feature
        """
        return torch.tensor(feature) if not isinstance(feature, torch.Tensor) else feature

    def _stack(
        self,
        tensors,
    ):
        """
        Stack tensors to the same length.
        Args:
            tensors (list): list of tensors to stack (N)
        Returns:
            torch.Tensor: stacked tensor of shape (N, max_length)
        """
        return torch.stack(tensors, dim=0)

    def __call__(self, samples):
        input_vars = set(self.input_variable_names).intersection(samples[0].keys())
        batch = {name: [] for name in input_vars}

        for name in input_vars:
            for sample in samples:
                sample_tensor = self._sample_to_tensor(sample[name])
                batch[name].append(sample_tensor)
        for name in input_vars:
            batch[name] = self._stack(batch[name])
        return batch
