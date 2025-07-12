import numpy as np
import torch


from nnt.datasets.dataset import DataSplit, Dataset


class ToyClassificationDataset(Dataset):
    """
    A simple dataset for toy_model.py.
    Generates random input features and targets for regression or classification.
    """

    def __init__(self, num_samples=1000, input_size=10, output_size=2):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            input_size (int): Size of the input features.
            output_size (int): Size of the output targets.
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        assert output_size > 1, "Output size must be greater than 1 for classification."
        super().__init__()

    def load(self):
        """
        Load the dataset.
        Returns:
            tuple: (inputs, targets) where inputs is a tensor of shape (num_samples, input_size)
                   and targets is a tensor of shape (num_samples, output_size).
        """

        weights = np.arange(1, self.input_size + 1, dtype=np.float32)

        # Precompute bin edges using theoretical min/max of logits
        min_logit = np.dot(np.zeros(self.input_size), weights)
        max_logit = np.dot(np.ones(self.input_size), weights)
        bins = np.linspace(min_logit, max_logit, self.output_size - 1)

        def get_random_sample():
            inputs = np.random.rand(self.input_size).astype(np.float32)
            logits = np.dot(inputs, weights)
            class_idx = np.digitize(logits, bins)
            class_idx = max(0, min(class_idx, self.output_size - 1))
            targets = np.zeros(self.output_size, dtype=np.float32)
            targets[class_idx] = 1.0
            return {"x": inputs, "y": targets}

        self["train"] = DataSplit([get_random_sample() for _ in range(self.num_samples)])
        self["validation"] = DataSplit([get_random_sample() for _ in range(self.num_samples // 10)])
