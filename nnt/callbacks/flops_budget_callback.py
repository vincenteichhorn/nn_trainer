from abc import abstractmethod
import os
from typing import TYPE_CHECKING

import pandas as pd
import torch

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.profiling.torch_profiler import TorchProfiler
from nnt.util.fast_csv import FastCSV


if TYPE_CHECKING:
    from nnt.trainer import Trainer


class FLOPsBudgetControllCallback(TrainerCallback):

    def __init__(self, output_dir: str, budget: int = None, should_stop_training: bool = True):
        super().__init__()
        self.output_dir = output_dir
        self.budget = budget
        self.flops_memorization_table = {}
        self.should_stop_training = should_stop_training
        self.flop_budget_file = os.path.join(output_dir, "flops_budget_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        self.fast_csv_writer = FastCSV(self.flop_budget_file, force=True)
        self.fast_writer_has_set_columns = False

        self.skip_info_keys = ["current_batch"]
        self.cumulative_flops = 0

    def _batch_signature(self, batch):
        str(tuple(batch[k].shape for k in batch if isinstance(batch[k], torch.Tensor)))

    def on_step_begin(self, info, trainer: "Trainer"):
        batch = info["current_batch"]
        signature = self._batch_signature(batch)
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
