from abc import abstractmethod
from dataclasses import dataclass
from email.headerregistry import DateHeader
import inspect
from typing import Any, Dict, List

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from nnt.validators.validator import PredictedBatch, Validator


class GenerationValidator(Validator):
    """
    Validator class for sequence generation tasks. Uses a model's generate method to produce outputs and decodes them using a tokenizer.

    Args:
        tokenizer: Tokenizer for decoding generated token IDs.
        max_length (int): Maximum length of generated sequences.
        temperature (float): Sampling temperature for generation.
        *args, **kwargs: Additional arguments passed to the base Validator.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 10,
        temperature: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the GenerationValidator with tokenizer, generation parameters, and base validator arguments.

        Args:
            tokenizer: Tokenizer for decoding generated sequences.
            max_length (int): Maximum length for generated sequences.
            temperature (float): Sampling temperature for generation.
            *args, **kwargs: Additional arguments for Validator.
        """
        self.max_length = max_length
        self.temperature = temperature
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def model_predict(self, batch: Dict[str, torch.Tensor]) -> PredictedBatch:
        """
        Generate predictions for a batch using the model's generate method and decode the results.

        Args:
            batch (dict): Batch of input data containing 'input_ids'.
        Returns:
            PredictedBatch: Object containing the batch and generated text predictions.
        """
        input_ids = batch["input_ids"]
        input_seq_len = input_ids.size(1)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=batch["input_ids"],
                max_length=self.max_length,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        new_token_ids = generated[:, input_seq_len:]
        generated_text = self.tokenizer.batch_decode(new_token_ids, skip_special_tokens=True)
        return PredictedBatch(batch=batch, prediction=generated_text)
