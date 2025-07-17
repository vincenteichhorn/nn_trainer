from abc import abstractmethod
import os
from typing import TYPE_CHECKING

import pandas as pd
import torch

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.profiling.torch_profiler import TorchProfiler
from nnt.util.fast_csv import FastCSV
from nnt.util.monitor import Monitor


if TYPE_CHECKING:
    from nnt.trainer import Trainer


class FLOPsBudgetControllCallback(TrainerCallback):
    """
    Callback for monitoring and controlling FLOPs budget during training.
    Tracks FLOPs per step, accumulates total FLOPs, and stops training if budget is exceeded.
    Logs step and cumulative FLOPs to a CSV file for analysis.

    Args:
        output_dir (str): Directory to save FLOPs logs.
        budget (int, optional): FLOPs budget for training. If exceeded, training stops.
        should_stop_training (bool): Whether to stop training when budget is exceeded.

    Example:
        callback = FLOPsBudgetControllCallback(output_dir="./logs", budget=1e12)
        trainer = Trainer(..., callbacks=[callback])
        trainer.train()
    """

    output_dir: str
    budget: int
    flops_memorization_table: dict
    should_stop_training: bool
    flop_budget_file: str
    fast_csv_writer: FastCSV
    fast_writer_has_set_columns: bool
    skip_info_keys: list
    cumulative_flops: int

    def __init__(self, output_dir: str, budget: int = None, should_stop_training: bool = True):
        """
        Initialize the FLOPsBudgetControllCallback and set up logging.

        Args:
            output_dir (str): Directory to save FLOPs logs.
            budget (int, optional): FLOPs budget for training.
            should_stop_training (bool): Whether to stop training when budget is exceeded.
        """
        super().__init__()
        self.output_dir = output_dir
        self.budget = budget
        self.flops_memorization_table = {}
        self.should_stop_training = should_stop_training
        self.flop_budget_file = os.path.join(output_dir, "flops_budget_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        self.fast_csv_writer = FastCSV(self.flop_budget_file, force=True)
        self.fast_writer_has_set_columns = False

        self.skip_info_keys = ["current_batch", "train_layers_with_gradients"]
        self.cumulative_flops = 0

    def on_step_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Compute and log FLOPs for the current step, update cumulative FLOPs, and stop training if budget exceeded.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        batch = info["current_batch"]
        model_signature = "-".join([name for name, param in trainer.model.named_parameters() if param.requires_grad])
        batch_signature = "-".join([str(batch[k].shape) for k in batch if isinstance(batch[k], torch.Tensor)])
        signature = f"{model_signature}-{batch_signature}"
        model = trainer.model
        if signature not in self.flops_memorization_table:
            with TorchProfiler() as prof:
                with prof.record_context("forward"):
                    out = model(**batch)
                with prof.record_context("backward"):
                    out.loss.backward()
            flops = prof.get_total_flops()
            self.flops_memorization_table[signature] = flops
        else:
            flops = self.flops_memorization_table[signature]
        self.cumulative_flops += flops
        ratio_of_budget = self.cumulative_flops / self.budget if self.budget is not None else -1

        if self.budget is not None and self.cumulative_flops > self.budget and self.should_stop_training:
            trainer.stop()

        if self.fast_writer_has_set_columns is False:
            self.info_keys = list(k for k in info.keys() if k not in self.skip_info_keys)
            self.fast_csv_writer.set_columns(self.info_keys + ["step_flops", "cumulative_flops", "ratio_of_budget"])
            self.fast_writer_has_set_columns = True
        self.fast_csv_writer.append(
            {key: info[key] for key in self.info_keys if key not in self.skip_info_keys}
            | {"step_flops": flops, "cumulative_flops": self.cumulative_flops, "ratio_of_budget": ratio_of_budget}
        )
