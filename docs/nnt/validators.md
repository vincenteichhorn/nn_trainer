# Validation 

## Overview

This framework provides a flexible way to validate deep learning models by running predictions on validation data batches and computing metrics. It is designed for both classification and generation tasks.

---

## Core Components

### `Validator` (Abstract Base Class)

Base class for validation logic.

* Manages batch preparation and device placement.
* Defines an abstract method `model_predict` to be implemented by subclasses.
* Runs validation loop over data batches.
* Computes metrics for each batch via metric objects implementing `compute()` and `finalize()` methods.

#### Key methods

* `__init__(model, validation_args, validation_data, metrics=None)`

  Initialize with a model, validation dataset, batch configuration, and metrics.

* `_prepare_data()`

  Prepares an iterator over validation batches using the configured data collator.

* `_batch_to_device(batch)`

  Moves tensors in a batch to the model’s device (GPU/CPU).

* `model_predict(batch) -> PredictedBatch`

  Abstract method to run model inference on a batch and return predictions.

* `validate() -> Dict[str, Any]`

  Runs the full validation loop, applies metrics, and returns aggregated results.

---

### `ForwardValidator`

Subclass of `Validator` for typical supervised tasks.

* Implements `model_predict` by running a forward pass with no gradient tracking.

```python
def model_predict(self, batch) -> PredictedBatch:
    with torch.no_grad():
        outputs = self.model(**batch)
    return PredictedBatch(batch=batch, prediction=outputs)
```

---

### `GenerationValidator`

Subclass of `Validator` for autoregressive generation models.

* Uses the model’s `.generate()` method of the model
* Decodes generated token sequences to text using a tokenizer.

```python
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
```

---

## Supporting Classes

### `ValidationArguments`

Holds validation configuration such as batch size and data collator function.

```python
@dataclass
class ValidationArguments:
    batch_size: int = 32
    data_collator: callable = None
```

### `PredictedBatch`

Data container for a single batch’s input, model prediction, and optionally ground-truth references (useful for generation).

```python
@dataclass
class PredictedBatch:
    batch: Dict[str, Any]
    prediction: Any
    reference_data: Dict[str, Any] = None
```

---

## Usage Example

```python
# Setup validation parameters and dataset
val_args = ValidationArguments(batch_size=16)
validation_data = my_dataset

# Define metrics to compute
metrics = [MyCustomMetric(), AnotherMetric()]

# Initialize validator (e.g., ForwardValidator or GenerationValidator)
validator = ForwardValidator(model=my_model, validation_args=val_args, validation_data=validation_data, metrics=metrics)

# Run validation and collect metric results
results = validator.validate()

print(results)
```

---

## Notes

* `model_predict` must be implemented by subclasses to handle model-specific prediction logic.
* The validation loop uses a progress bar (`Monitor().tqdm`) for convenience.
* Metrics receive a `PredictedBatch` object and should implement `compute` and `finalize`.
* Device placement of inputs is automatically handled.
* Reference data is extracted from the raw batch and attached to `PredictedBatch` for metric use (e.g., for generation metrics needing references).
* Not all metrics are compatable with all types of validation; ensure metrics are appropriate for the task (e.g., classification vs. generation).
