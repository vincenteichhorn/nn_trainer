from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import Any, Dict, List, Callable, Iterable

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from nnt.collators.data_collators import PlainDataCollator
from nnt.util.functions import iter_batchwise
from nnt.util.monitor import Monitor
from nnt.validation_metrics.validation_metric import ValidationMetric


@dataclass
class ValidationArguments:
    """
    Data class for storing validation configuration arguments.

    Attributes:
        batch_size (int): Batch size for validation.
        data_collator (callable, optional): Function to collate validation data batches.
    """

    batch_size: int = 32
    data_collator: Callable = None  # type: ignore


@dataclass
class PredictedBatch:
    """
    Data class for storing a batch's predictions and reference data.

    Attributes:
        batch (Dict[str, Any]): Input batch data.
        prediction (Any): Model predictions for the batch.
        reference_data (Dict[str, Any], optional): Reference data for evaluation.
    """

    batch: Dict[str, Any]
    prediction: Any
    reference_data: Dict[str, Any] = None


class Validator:
    """
    Base class for validators. Handles validation loop, batching, and metric computation.
    """

    def __init__(
        self,
        model: Module,
        validation_args: ValidationArguments,
        validation_data: Iterable[Dict[str, Any]],
        metrics: List[ValidationMetric] = None,
    ):
        """
        Initialize the Validator object and set up model, arguments, data, and metrics.

        Args:
            model: The model to validate.
            validation_args (ValidationArguments): Validation configuration arguments.
            validation_data: Validation dataset.
            metrics (list, optional): List of metric objects for evaluation.
        """
        self.metrics = metrics or []
        self.model = model
        self.validation_args = validation_args
        self.validation_data = validation_data
        self.validation_batches = None
        self.device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_input_parameter_names = inspect.signature(self.model.forward).parameters.keys()

    def _prepare_data(self) -> None:
        """
        Prepare the validation data loader and set up the data collator.
        """
        if self.validation_args.data_collator is None:
            self.validation_args.data_collator = PlainDataCollator(input_variable_names=self.model_input_parameter_names)
        self.validation_batches = iter_batchwise(self.validation_data, self.validation_args.batch_size)

    def _batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Move batch tensors to the appropriate device (CPU or GPU).

        Args:
            batch (dict): Batch of data.
        Returns:
            dict: Batch with tensors moved to device.
        """
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @abstractmethod
    def model_predict(self, batch: Dict[str, Any]) -> PredictedBatch:
        """
        Predict the output of the model for a given batch.
        This method should be implemented by subclasses.

        Args:
            batch (dict): Batch of input data.
        Returns:
            PredictedBatch: Object containing predictions and reference data.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def validate(self) -> Dict[str, Any]:
        """
        Validate the model on the validation data.
        Prepares the data, runs predictions, and computes metrics.

        Returns:
            dict: Results from all metrics after validation.
        """
        self._prepare_data()
        self.model.eval()

        with torch.no_grad():
            for batch in Monitor().tqdm(
                self.validation_batches,
                desc=self.__class__.__name__,
                total=len(self.validation_data) // self.validation_args.batch_size,
            ):
                collated_batch = self.validation_args.data_collator(batch)
                collated_batch = self._batch_to_device(collated_batch)
                predicted_batch = self.model_predict(collated_batch)
                reference_data = {k: [d[k] for d in batch] for k in batch[0].keys()}
                predicted_batch.reference_data = reference_data
                for metric in self.metrics:
                    metric.compute(predicted_batch)

        results = {metric.__class__.__name__: metric.finalize() for metric in self.metrics}

        return results
