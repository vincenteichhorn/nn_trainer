# Tutorial: Training a Toy Classification Model with Callbacks and Validation

This tutorial walks through training a simple classification model using the `Trainer` class and demonstrates how to add monitoring, validation, and resource tracking callbacks.

---

## Step 1: Import Required Modules

The following modules provide datasets, models, training framework, metrics, validators, and useful callbacks:

```python
from nnt.callbacks.energy_callback import EnergyCallback
from nnt.callbacks.flops_budget_callback import FLOPsBudgetControllCallback
from nnt.callbacks.logging_callback import LoggingCallback
from nnt.callbacks.validator_callback import ValidatorCallback
from nnt.datasets.toy_dataset import ToyClassificationDataset
from nnt.models.toy_models import ToyClassificationModel
from nnt.trainer import Trainer, TrainingArguments
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.validator import ValidationArguments
```

---

## Step 2: Prepare Dataset and Model

* Create a toy dataset with 10-dimensional input and 2 output classes.
* Define a simple feedforward classification model.

```python
num_classes = 2

dataset = ToyClassificationDataset(input_size=10, output_size=num_classes, num_samples=1000)

model = ToyClassificationModel(
    input_size=10,
    hidden_size=20,
    output_size=num_classes,
)
```

* The dataset contains 1000 samples with train and validation splits.
* The model has one hidden layer with 20 units.

---

## Step 3: Define Training Arguments

Specify hyperparameters and training behavior:

```python
training_args = TrainingArguments(
    num_epochs=5,
    batch_size=1,
    learning_rate=0.001,
    weight_decay=0.01,
    monitor_strategy="steps",
    monitor_every=1000,
    checkpoint_strategy="steps",
    checkpoint_every=1000,
)
```

* Training will run for 5 epochs with batch size 1.
* Monitor and checkpoint every 1000 steps.
* AdamW optimizer default parameters will be used.

---

## Step 4: Setup Validation Metrics and Validator

* Use one-hot classification metrics to evaluate model performance.
* Initialize a `ForwardValidator` to run validation batches through the model.

```python
metrics = [OneHotClassificationMetrics(num_classes=num_classes, targets_key="y", logits_key="logits")]

validator = ForwardValidator(
    model=model,
    validation_args=ValidationArguments(batch_size=32),
    validation_data=dataset["validation"],
    metrics=metrics,
)
```

* Validation runs on batches of 32.
* The validator computes metrics like accuracy or F1 during validation.

---

## Step 5: Define Callbacks

Callbacks enhance training with logging, validation, resource monitoring, and FLOPs control.

```python
output_dir = "./out/toy_classification_model"

callbacks = [
    LoggingCallback(output_dir=output_dir),  # Logs training progress
    ValidatorCallback(                      # Runs validation every 500 steps and logs results
        output_dir=output_dir,
        log_file="validation_log.csv",
        validator=validator,
        validate_strategy="steps",
        validate_every=500,
    ),
    FLOPsBudgetControllCallback(            # Monitors FLOPs, optionally stops training if budget exceeded
        output_dir=output_dir,
        budget=1e9,
        should_stop_training=False,
    ),
    EnergyCallback(                         # Monitors system energy consumption (NVIDIA GPU only)
        output_dir=output_dir,
        nvidia_query_interval=10,
    ),
]
```

---

## Step 6: Initialize Trainer and Start Training

Create the `Trainer` instance with all components and kick off training:

```python
trainer = Trainer(
    output_dir=output_dir,
    model=model,
    training_args=training_args,
    train_data=dataset["train"],
    callbacks=callbacks,
)

trainer.train()
```

* Training runs for configured epochs with real-time monitoring.
* Checkpoints and logs are saved to the specified `output_dir`.

---

## Summary

This example demonstrates:

* How to load or create a toy dataset and model.
* Configure training hyperparameters.
* Use validators and metrics for evaluation during training.
* Attach callbacks to log progress, validate the model periodically, monitor FLOPs usage, and track energy consumption.
* Run the full training loop using the flexible `Trainer` class.

---

**Feel free to modify the dataset, model, or callbacks to customize training behavior by creating creating new subclasses!**
