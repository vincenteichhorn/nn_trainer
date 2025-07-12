from abc import abstractmethod
from typing import Any, Dict, List
from nnt.validators.validator import PredictedBatch


class ValidationMetric:
    """
    Base class for validation metrics.
    This class should be extended to implement specific validation metrics.
    """

    def __init__(self):
        self.validation_results = []

    @abstractmethod
    def compute(self, predicted_batch: PredictedBatch) -> Dict[str, List[Any]]:
        """
        Compute the metric based on the predicted batch.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize the metric computation and return the result.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")
