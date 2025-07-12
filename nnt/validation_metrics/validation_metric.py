from abc import abstractmethod
from typing import Any, Dict, List


class ValidationMetric:
    """
    Base class for validation metrics.
    This class should be extended to implement specific validation metrics.
    Handles computation and finalization of metric results for model validation.
    """

    def __init__(self) -> None:
        """
        Initialize the ValidationMetric object and set up storage for validation results.
        """
        self.validation_results = []

    @abstractmethod
    def compute(self, predicted_batch) -> None:
        """
        Compute the metric based on the predicted batch.
        This method should be implemented by subclasses.

        Args:
            predicted_batch (PredictedBatch): Batch containing predictions and reference data.
        """
        from nnt.validators.validator import PredictedBatch

        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def finalize(self) -> Dict[str, float]:
        """
        Finalize the metric computation and return the result.
        This method should be implemented by subclasses.

        Returns:
            Dict[str, float]: Finalized metric results after all batches have been processed.
        """
        raise NotImplementedError("Subclasses must implement this method.")
