# TrainerCallback System Documentation

## 📌 Overview

The `TrainerCallback` system enables you to inject **custom logic** at specific training lifecycle points without modifying the core training loop. Simply subclass `TrainerCallback` and override relevant hooks to integrate:

* Logging
* Monitoring
* Validation
* Custom utilities

---

## Base Interface: `TrainerCallback`

```python
class TrainerCallback:
    def on_step_begin(self, info: dict, trainer: "Trainer"): pass
    def on_step_end(self, info: dict, trainer: "Trainer"): pass
    def on_epoch_begin(self, info: dict, trainer: "Trainer"): pass
    def on_epoch_end(self, info: dict, trainer: "Trainer"): pass
    def on_training_begin(self, info: dict, trainer: "Trainer"): pass
    def on_training_end(self, info: dict, trainer: "Trainer"): pass
    def on_checkpoint(self, info: dict, trainer: "Trainer"): pass
```

### Parameters

* `info (dict)`:

  * `global_step`: Current global step.
  * `epoch`: Current epoch number.
  * `current_batch`: Actual batch data *(optional, often excluded from logs)*.
  * Any other trainer-generated metadata or metrics.

* `trainer (Trainer)`:

  * Reference to the main `Trainer` instance.
  * Allows access to:

    * `model`, `optimizer`, `train_data`
    * Control methods like `trainer.stop()`

---

## How to Implement a Custom Callback

1. **Subclass** `TrainerCallback`
2. **Override** the desired hook methods
3. Use `info` to track steps, epochs, or metrics
4. Use `trainer` to:

   * Access training components
   * Trigger actions (e.g., `stop()`)
5. *(Optional)* Log or integrate with external tools

---

## Example: Custom Print Callback

Prints a message at the start and end of each epoch:

```python
from nnt.callbacks.trainer_callback import TrainerCallback

class PrintCallback(TrainerCallback):
    def on_epoch_begin(self, info: dict, trainer: "Trainer"):
        print(f"Epoch {info['epoch']} started.")

    def on_epoch_end(self, info: dict, trainer: "Trainer"):
        print(f"Epoch {info['epoch']} ended. Loss: {info.get('loss', 'N/A')}")

    def on_training_end(self, info: dict, trainer: "Trainer"):
        print("Training complete.")
```

---

## Example: Registering Callbacks with Trainer

```python
from nnt.trainer import Trainer
from my_project.callbacks import PrintCallback, LoggingCallback, EnergyCallback

trainer = Trainer(
    model=my_model,
    train_data=train_dataset,
    val_data=val_dataset,
    callbacks=[
        PrintCallback(),
        LoggingCallback(output_dir="./logs"),
        EnergyCallback(output_dir="./logs")
    ]
)

trainer.train()
```

---

## Best Practices

* Avoid logging large objects like raw tensors — filter them from `info` if needed.
* Prefer structured logs with tools like `FastCSV`.
* Keep callback logic **side-effect-free** unless intentional.
* Use `trainer.stop()` inside `on_step_begin` or `on_epoch_end` for early stopping (e.g., based on FLOPs or custom criteria).

---

## Related Utilities

| Utility          | Purpose                                      |
| ---------------- | -------------------------------------------- |
| `FastCSV`        | Fast, structured CSV writer for metrics      |
| `flatten_dict`   | Flattens nested dicts for easy CSV logging   |
| `Monitor`        | Colored, readable console logs               |
| `TorchProfiler`  | Performance profiling for PyTorch            |
| `NvidiaProfiler` | Energy and performance metrics (NVIDIA GPUs) |

---

Let me know if you want this exported to a `.md` file or converted to HTML for docs!
