import torch


class DataCollatorForCausalLM:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.padding_token = tokenizer.pad_token_id
        self.loss_mask_token = -100

    def _stack_and_pad(
        self,
        tensors,
        padding_side: str = "right",
        pad_token: int = None,
        pad_to_multiple_of: int = 16,
    ):
        """
        Stack and pad tensors to the same length, optionally padding to a multiple of a given value.
        Args:
            tensors (list): list of tensors to stack and pad (N)
            padding_side (str): "right" or "left"
            pad_token (int): padding token id
            pad_to_multiple_of (int): pad the sequence length to a multiple of this value
        Returns:
            torch.Tensor: stacked and padded tensor of shape (N, max_length)
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

    def _feature_to_tensor(self, feature):
        """
        convert a feature (input_ids, attention_mask) to a tensor if it is not already a tensor
        Args:
            feature (Union[list, torch.Tensor]): the feature to convert
        Returns:
            torch.Tensor: the converted feature
        """
        return torch.tensor(feature) if not isinstance(feature, torch.Tensor) else feature

    def __call__(self, samples, padding_side: str = "left"):
        """
        Callable function to stack and pad features to max length of a batch
        Args:
            features (list): list of features to stack and pad
            padding_side (str): "right" or "left"
        Returns:
            dict: dictionary of stacked and padded features (input_ids, attention_mask, labels)
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
