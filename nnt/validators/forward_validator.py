from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import List

import numpy as np
import torch

from nnt.validators.validator import PredictedBatch, Validator


class ForwardValidator(Validator):

    def model_predict(self, batch) -> PredictedBatch:

        with torch.no_grad():
            outputs = self.model(**batch)

        return PredictedBatch(batch=batch, prediction=outputs)
