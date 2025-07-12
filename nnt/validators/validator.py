from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from nnt.collators.data_collators import PlainDataCollator
from nnt.util.functions import iter_batchwise
from nnt.util.monitor import Monitor


@dataclass
class ValidationArguments:
    batch_size: int = 32
    data_collator: callable = None


@dataclass
class PredictedBatch:
    batch: Dict[str, Any]
    prediction: Any
    reference_data: Dict[str, Any] = None


class Validator:
    """
    Base class for validators.
    """

    def __init__(self, model, validation_args, validation_data, metrics=None):
        self.metrics = metrics or []
        self.model = model
        self.validation_args = validation_args
        self.validation_data = validation_data
        self.validation_batches = None
        self.device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_input_parameter_names = inspect.signature(self.model.forward).parameters.keys()

    def _prepare_data(self):
        if self.validation_args.data_collator is None:
            self.validation_args.data_collator = PlainDataCollator(input_variable_names=self.model_input_parameter_names)
        self.validation_batches = iter_batchwise(self.validation_data, self.validation_args.batch_size)

    def _batch_to_device(self, batch):
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @abstractmethod
    def model_predict(self, batch) -> PredictedBatch:
        """
        Predict the output of the model for a given batch.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def validate(self):
        """
        Validate the model on the validation data.
        This method prepares the data and calls the model_predict method.
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
