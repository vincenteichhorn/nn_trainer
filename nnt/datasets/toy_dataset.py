import numpy as np
import torch


from nnt.datasets.dataset import DataSplit, Dataset


class ToyClassificationDataset(Dataset):
    """
    A simple dataset for toy_model.py.
    Generates random input features and one-hot targets for classification tasks.

    Args:
        num_samples (int): Number of samples in the dataset.
        input_size (int): Size of the input features.
        output_size (int): Number of output classes.

    Example:
        ds = ToyClassificationDataset(num_samples=100, input_size=5, output_size=3)
        ds.load()
        print(ds['train'][0])
    """

    num_samples: int
    input_size: int
    output_size: int

    def __init__(self, num_samples: int = 1000, input_size: int = 10, output_size: int = 2):
        """
        Initialize the ToyClassificationDataset.

        Args:
            num_samples (int): Number of samples in the dataset.
            input_size (int): Size of the input features.
            output_size (int): Number of output classes.
        Raises:
            AssertionError: If output_size is not greater than 1.
        """
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        assert output_size > 1, "Output size must be greater than 1 for classification."
        super().__init__()

    def load(self) -> None:
        """
        Load the dataset by generating random samples and assigning them to train and validation splits.

        Returns:
            None
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
