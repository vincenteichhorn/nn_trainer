from abc import abstractmethod
from typing import Dict, List, Literal, Tuple, Union
import numpy as np
import torch


class ClassificationMetrics:
    """
    Base class for computing and storing classification metrics.

    Attributes:
        num_classes (int): Number of classes in the classification task.
        confusion_matrix (np.ndarray): Confusion matrix for the predictions.
    """

    def __init__(self, num_classes: int):
        """
        Initialize the ClassificationMetrics object.

        Args:
            num_classes (int): Number of classes in the classification task.
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    @property
    def name(self) -> str:
        """
        Returns the name of the metric class.

        Returns:
            str: Name of the metric class.
        """
        return self.__class__.__name__

    @abstractmethod
    def check_classification(self, predicted_batch) -> List[Tuple[int, int]]:
        """
        For each sample in the batch, return a list of tuples (true_label, predicted_label).

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and true labels.

        Returns:
            List[Tuple[int, int]]: List of (true_label, predicted_label) pairs.
        """

        raise NotImplementedError("Subclasses must implement this method.")

    def compute(self, predicted_batch) -> None:
        """
        Update the confusion matrix based on the predicted batch.

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and true labels.
        """

        classification_results = self.check_classification(predicted_batch)
        for true_label, predicted_label in classification_results:
            self.confusion_matrix[true_label, predicted_label] += 1

    def finalize(self) -> Dict[str, float]:
        """
        Finalize the metric computation and return the confusion matrix and metrics.

        Returns:
            Dict[str, float]: Dictionary containing accuracy, precision, recall, f1_score, and mcc.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            accuracy = np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)
            precision = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=0)
            precision = np.nan_to_num(precision, nan=0.0)
            recall = np.diag(self.confusion_matrix) / np.sum(self.confusion_matrix, axis=1)
            recall = np.nan_to_num(recall, nan=0.0)
            f1_score = 2 * (precision * recall) / (precision + recall)
            f1_score = np.nan_to_num(f1_score, nan=0.0)
            mcc = None
            if self.num_classes == 2:
                # For binary classification, return single float for each metric
                precision = precision[1]
                recall = recall[1]
                f1_score = f1_score[1]
                # Matthews correlation coefficient for binary classification
                tn = self.confusion_matrix[0, 0]
                fp = self.confusion_matrix[0, 1]
                fn = self.confusion_matrix[1, 0]
                tp = self.confusion_matrix[1, 1]
                numerator = (tp * tn) - (fp * fn)
                denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
                mcc = numerator / denominator if denominator != 0 else 0.0
            self.confusion_matrix = np.zeros_like(self.confusion_matrix)
            return {
                "accuracy": float(accuracy),
                "precision": float(precision) if not isinstance(precision, np.ndarray) else precision.tolist(),
                "recall": float(recall) if not isinstance(recall, np.ndarray) else recall.tolist(),
                "f1_score": float(f1_score) if not isinstance(f1_score, np.ndarray) else f1_score.tolist(),
                "mcc": float(mcc) if mcc is not None else None,
            }


class OneHotClassificationMetrics(ClassificationMetrics):
    """
    Metrics class for evaluating classification tasks with one-hot encoded targets.
    Supports both standard and sequence classification tasks, with options for class filtering and label padding.

    Attributes:
        sequence_offset (int): Offset for sequence classification tasks, indicating which token in the sequence to evaluate.
        classes (Union[Literal["all"], List[int]]): List of class indices to consider for evaluation. If "all", all classes are considered.
        label_padding_value (int): Padding value for labels in sequence tasks, used to ignore padded tokens.
    """

    def __init__(
        self,
        num_classes: int,
        logits_key: str = "logits",
        targets_key: str = "y",
        sequence_offset: int = 0,
        classes: Union[Literal["all"], List[int]] = None,
        label_padding_value: int = -100,
    ):
        """
        Initialize the OneHotClassificationMetrics object.

        Args:
            num_classes (int): Number of classes.
            logits_key (str): Key for logits in prediction.
            targets_key (str): Key for targets in batch.
            sequence_offset (int): Offset for sequence classification.
            classes (Union[Literal["all"], List[int]], optional): Classes to consider.
            label_padding_value (int, optional): Padding value for labels.
        """
        super().__init__(num_classes)
        self.classes = classes if isinstance(classes, list) else list(range(num_classes))
        self.logits_key = logits_key
        self.targets_key = targets_key
        self.sequence_offset = sequence_offset
        self.label_padding_value = label_padding_value

    def check_classification(self, predicted_batch) -> List[Tuple[int, int]]:
        """
        For each sample in the batch, return a list of tuples (true_label, predicted_label).

        Handles both standard and sequence classification tasks. Applies class filtering and checks for unsupported shapes or invalid labels.

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and true labels.

        Returns:
            List[Tuple[int, int]]: List of (true_label, predicted_label) pairs.

        Raises:
            ValueError: If logits shape is unsupported or label not in classes.
        """

        logits = vars(predicted_batch.prediction).get(self.logits_key)
        labels = predicted_batch.batch.get(self.targets_key)

        if logits.ndim == 2:
            predicted_labels = torch.argmax(logits, dim=1)
            true_labels = torch.argmax(labels, dim=1)
            output = [(int(t.item()), int(p.item())) for t, p in zip(true_labels, predicted_labels)]
        elif logits.ndim == 3:
            predicted_labels, true_labels = [], []
            logits = logits[:, :, self.classes]

            for pad_logits_sequence, pad_label_sequence in zip(logits, labels):
                label_sequence = pad_label_sequence[pad_label_sequence != self.label_padding_value]
                true_label = label_sequence[len(label_sequence) - self.sequence_offset - 1].cpu()
                if not torch.isin(true_label, torch.tensor(self.classes)):
                    raise ValueError(
                        f"Label {true_label.item()} not in classes {self.classes}. Ensure that the sequence offset is correct."
                    )
                true_labels.append(self.classes.index(int(true_label.item())))

                logits_sequence = pad_logits_sequence[pad_label_sequence != self.label_padding_value, :]
                predicted_label = logits_sequence[len(logits_sequence) - self.sequence_offset - 2]
                predicted_labels.append(int(torch.argmax(predicted_label).item()))
            output = [(int(t), int(p)) for t, p in zip(true_labels, predicted_labels)]
        else:
            raise ValueError(f"Unsupported logits shape: {logits.shape}")

        return output
