import os
from typing import TYPE_CHECKING, Literal
from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.util.fast_csv import FastCSV
from nnt.util.functions import flatten_dict
from nnt.util.monitor import Monitor
from nnt.validators.validator import Validator

if TYPE_CHECKING:
    from nnt.trainer import Trainer


class ValidatorCallback(TrainerCallback):
    """
    Callback for performing validation during training using a Validator.
    Computes metrics on the validation dataset at specified intervals and logs results to a CSV file.

    Args:
        output_dir (str): Directory to save validation logs.
        validator (Validator): Validator instance for computing metrics.
        log_file (str): Name of the log file.
        validate_strategy (Literal["epochs", "steps"]): Validation interval strategy.
        validate_every (int): Validation interval.

    Example:
        callback = ValidatorCallback(output_dir="./logs", validator=my_validator)
        trainer = Trainer(..., callbacks=[callback])
        trainer.train()
    """

    output_dir: str
    validation_log_file: str
    validation_strategy: str
    validation_interval: int
    validator: Validator
    fast_csv_writer: FastCSV
    writer_has_set_columns: bool
    skip_info_keys: list

    def __init__(
        self,
        output_dir: str,
        validator: Validator,
        log_file: str = "validation_log.csv",
        validate_strategy: str = "epoch",
        validate_every: int = 1,
    ) -> None:
        """
        Initialize the ValidatorCallback and set up logging and validation strategy.

        Args:
            output_dir (str): Directory to save validation logs.
            validator (Validator): Validator instance for computing metrics.
            log_file (str): Name of the log file.
            validate_strategy (Literal["epochs", "steps"]): Validation interval strategy.
            validate_every (int): Validation interval.
        """
        self.output_dir = output_dir
        self.validation_log_file = os.path.join(output_dir, log_file)
        self.validation_strategy = validate_strategy
        self.validation_interval = validate_every

        self.validator = validator

        os.makedirs(output_dir, exist_ok=True)
        self.fast_csv_writer = FastCSV(self.validation_log_file, force=True)
        self.writer_has_set_columns = False
        self.skip_info_keys = ["current_batch"]

    def validate(self, info: dict, trainer: "Trainer") -> None:
        global_step = info["global_step"]
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

    def on_step_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Perform validation and log results at the beginning of a step if the interval is met.

        Args:
            info (dict): Training info for the step.
            trainer (Trainer): Trainer instance.
        """
        train_set_size = len(trainer.train_data) // trainer.training_args.batch_size
        self.validate_every = (
            self.validation_interval if self.validation_strategy == "steps" else (train_set_size * self.validation_interval)
        )
        global_step = info["global_step"]
        if (
            global_step % self.validate_every == 0
            and global_step != 0
            and global_step != train_set_size * trainer.training_args.num_epochs
        ):
            self.validate(info, trainer)

    def on_training_end(self, info, trainer):
        self.validate(info, trainer)

    def on_training_begin(self, info, trainer):
        self.validate(info, trainer)
