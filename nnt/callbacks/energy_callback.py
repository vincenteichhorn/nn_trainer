import os
import subprocess
from typing import TYPE_CHECKING
import warnings

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.profiling.nvidia_profiler import NvidiaProfiler

if TYPE_CHECKING:
    from nnt.trainer import Trainer


class EnergyCallback(TrainerCallback):

    def __init__(self, output_dir, nvidia_query_interval: int = 10):
        energy_log = os.path.join(output_dir, "energy_log.csv")
        # check if nvidia-smi is available by calling it
        self.prof = None
        if subprocess.getstatusoutput("nvidia-smi")[0] == 0:
            self.prof = NvidiaProfiler(
                interval=nvidia_query_interval,
                cache_file=energy_log,
            )
            self.prof.start()
        else:
            warnings.warn(
                "NVIDIA GPU not detected or nvidia-smi not available. EnergyCallback will not be active.",
                UserWarning,
            )

    def __del__(self):
        if self.prof is None:
            return
        self.prof.stop()

    def on_step_begin(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("step_begin")

    def on_step_end(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("step_end")

    def on_epoch_begin(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("epoch_begin")

    def on_epoch_end(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("epoch_end")

    def on_training_begin(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("training_begin")

    def on_training_end(self, info: dict, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("training_end")

    def on_checkpoint(self, info, trainer: "Trainer"):
        if self.prof is None:
            return
        self.prof.record_step("checkpoint")
