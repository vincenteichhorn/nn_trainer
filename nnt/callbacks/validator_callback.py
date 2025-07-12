import os
from typing import Literal
from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.util.fast_csv import FastCSV
from nnt.util.functions import flatten_dict
from nnt.util.monitor import Monitor
from nnt.validators.validator import Validator


class ValidatorCallback(TrainerCallback):
    """
    Callback for validation during training.

    This callback is used to perform validation at the end of each epoch or at specified intervals.
    It utilizes the Validator class to compute metrics on the validation dataset.
    """

    def __init__(
        self,
        output_dir: str,
        validator: Validator,
        log_file: str = "validation_log.csv",
        validate_strategy: Literal["epochs", "steps"] = "epoch",
        validate_every: int = 1,
    ):
        self.output_dir = output_dir
        self.validation_log_file = os.path.join(output_dir, log_file)
        self.validation_strategy = validate_strategy
        self.validation_interval = validate_every

        self.validator = validator

        os.makedirs(output_dir, exist_ok=True)
        self.fast_csv_writer = FastCSV(self.validation_log_file, force=True)
        self.writer_has_set_columns = False
        self.skip_info_keys = ["current_batch"]

    def on_step_begin(self, info, trainer):
        train_set_size = len(trainer.train_data)
        self.validate_every = (
            self.validation_interval if self.validation_strategy == "steps" else (train_set_size * self.validation_interval)
        )
        global_step = info["global_step"]
        if global_step % self.validate_every == 0:
            results = self.validator.validate()
            row = {
                **{k: v for k, v in info.items() if k not in self.skip_info_keys},
                **flatten_dict(results),
            }
            if not self.writer_has_set_columns:
                self.fast_csv_writer.set_columns(row.keys())
                self.writer_has_set_columns = True
            Monitor().print(f"Validation results at step {global_step}: {' '.join(f'{k}: {v}' for k, v in row.items())}")

            self.fast_csv_writer.append(row)
