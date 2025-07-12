from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import List

import numpy as np
import torch

from nnt.validators.validator import PredictedBatch, Validator


class GenerationValidator(Validator):

    def __init__(self, tokenizer, max_length=10, temperature=1.0, *args, **kwargs):
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def model_predict(self, batch) -> PredictedBatch:
        input_ids = batch["input_ids"]
        input_seq_len = input_ids.size(1)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=batch["input_ids"],
                max_length=self.max_length,
                temperature=self.temperature,
            )
        new_token_ids = generated[:, input_seq_len:]
        generated_text = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return PredictedBatch(batch=batch, prediction=generated_text)
