# Trainer Class Documentation

## Overview

The `Trainer` class provides a flexible and extensible framework for training PyTorch models with:

* Configurable training parameters
* Dataset batching and collation
* Optimizer management
* Callback hooks for custom logic at various training stages
* Checkpoint saving and monitoring progress

---

## Classes and Data Structures

### `TrainingArguments`

Configuration holder for training hyperparameters and strategies.

| Field                 | Type                    | Default      | Description                                              |
| --------------------- | ----------------------- | ------------ | -------------------------------------------------------- |
| `num_epochs`          | `int`                   | **Required** | Number of training epochs                                |
| `batch_size`          | `int`                   | **Required** | Batch size per training step                             |
| `data_collator`       | `callable`              | `None`       | Function to collate raw samples into batches             |
| `learning_rate`       | `float`                 | 2e-4         | Learning rate for optimizer                              |
| `weight_decay`        | `float`                 | 0.01         | Weight decay regularization factor                       |
| `monitor_strategy`    | `'steps'` or `'epochs'` | `'steps'`    | Frequency unit for monitoring progress                   |
| `monitor_every`       | `int`                   | 1000         | Interval (in steps or epochs) to print training progress |
| `checkpoint_strategy` | `'steps'` or `'epochs'` | `'steps'`    | Frequency unit for saving checkpoints                    |
| `checkpoint_every`    | `int`                   | 1000         | Interval (in steps or epochs) to save model checkpoints  |
| `model_save_function` | `callable`              | `None`       | Custom function to save model state                      |

---

### `Trainer`

Main training loop manager.

#### Constructor

```python
Trainer(
    output_dir: str,
    model,
    training_args: TrainingArguments,
    train_data: DataSplit,
    optimizer: torch.optim.Optimizer = None,
    callbacks: List[TrainerCallback] = [],
)
```

* `output_dir`: Directory to save checkpoints and model artifacts.
* `model`: PyTorch model instance to train.
* `training_args`: Instance of `TrainingArguments` defining hyperparameters.
* `train_data`: Dataset for training (supports PyTorch `DataSplit` interface).
* `optimizer`: Optional optimizer; defaults to AdamW with specified LR and weight decay.
* `callbacks`: List of callback objects implementing training event hooks.

#### Key Methods

* `_prepare_data()`

  Sets up the training data loader with batching and optional data collator.

* `_batch_to_device(batch)`

  Moves input tensors in a batch to the configured device (GPU or CPU).

* `_save_model(global_step: int, checkpoint: bool = False)`

  Saves model state either to checkpoint folder (if `checkpoint=True`) or final model folder.

* `stop()`

  Request training to stop gracefully after the current batch.

* `train()`

  Runs the full training loop over configured epochs and batches.

---

## Training Loop Details (`train()`)

1. Prepares batched training data loader.
2. Sets model to training mode.
3. Initializes step counters and progress tracking.
4. Iterates over epochs:

   * Seeds data loader worker for reproducibility per epoch.
   * Calls `on_epoch_begin` callback.
   * Iterates over batches:

     * Calls `on_step_begin` callback.
     * Moves batch to device.
     * Runs forward pass and calculates loss.
     * Backpropagates loss.
     * Steps optimizer.
     * Calls `on_step_end` callback.
     * Updates progress bar and counters.
     * Periodically logs training info and saves checkpoints based on configured frequency.
   * Calls `on_epoch_end` callback.
5. Calls `on_training_end` callback at the end.
6. Saves final model state.
7. Returns trained model instance.

---

## Callback System

Callbacks are user-defined objects implementing any subset of the following methods:

* `on_training_begin(info: dict, trainer: Trainer)`
* `on_epoch_begin(info: dict, trainer: Trainer)`
* `on_step_begin(info: dict, trainer: Trainer)`
* `on_step_end(info: dict, trainer: Trainer)`
* `on_checkpoint(info: dict, trainer: Trainer)`
* `on_epoch_end(info: dict, trainer: Trainer)`
* `on_training_end(info: dict, trainer: Trainer)`

Each method receives:

* `info`: Dictionary containing current state info such as epoch, step, learning rate, timestamp, current batch, and loss.
* `trainer`: The current `Trainer` instance.

---

## Example Usage

```python
training_args = TrainingArguments(
    num_epochs=3,
    batch_size=16,
    learning_rate=3e-4,
    monitor_strategy="steps",
    monitor_every=500,
    checkpoint_strategy="steps",
    checkpoint_every=1000,
)

trainer = Trainer(
    output_dir="./training_output",
    model=my_model,
    training_args=training_args,
    train_data=my_train_dataset,
    optimizer=None,  # default AdamW will be used
    callbacks=[LoggingCallback(), EnergyCallback()],
)

trained_model = trainer.train()
```

---

## Notes

* The trainer automatically detects and uses GPU if available.
* Model saving can be customized by providing a `model_save_function` in `TrainingArguments`.
* Batch seeding ensures reproducibility between epochs.
* Progress bars and monitoring are integrated with `tqdm` for user-friendly feedback.
* Callbacks provide extensibility to inject behavior without modifying core logic.
