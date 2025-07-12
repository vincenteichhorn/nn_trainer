from abc import abstractmethod
import os
from typing import TYPE_CHECKING

import pandas as pd

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.util.fast_csv import FastCSV


if TYPE_CHECKING:
    from nnt.trainer import Trainer


class LoggingCallback(TrainerCallback):

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_file = os.path.join(output_dir, "train_log.csv")
        os.makedirs(output_dir, exist_ok=True)
        self.fast_csv_writer = FastCSV(self.log_file, force=True)
        self.skip_info_keys = ["current_batch"]
        self.writer_has_set_columns = False

    def on_step_end(self, info: dict, trainer: "Trainer"):
        row = {k: v for k, v in info.items() if k not in self.skip_info_keys}
        if not self.writer_has_set_columns:
            self.fast_csv_writer.set_columns(list(row.keys()))
            self.writer_has_set_columns = True
        self.fast_csv_writer.append(row)
