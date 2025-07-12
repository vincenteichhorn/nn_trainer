from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import Dict, List

import numpy as np
import torch

from nnt.validators.validator import PredictedBatch, Validator


class ForwardValidator(Validator):
    """
    Validator class for direct forward pass evaluation. Uses the model's forward method to produce outputs for each batch.
    """

    def model_predict(self, batch: Dict[str, torch.Tensor]) -> PredictedBatch:
        """
        Run a forward pass of the model on the given batch and return predictions.

        Args:
            batch (dict): Batch of input data for the model.
        Returns:
            PredictedBatch: Object containing the batch and model outputs as predictions.
        """

        with torch.no_grad():
            outputs = self.model(**batch)

        return PredictedBatch(batch=batch, prediction=outputs)
