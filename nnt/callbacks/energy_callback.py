import os
import subprocess
from typing import TYPE_CHECKING
import warnings

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.profiling.nvidia_profiler import NvidiaProfiler

if TYPE_CHECKING:
    from nnt.trainer import Trainer


class EnergyCallback(TrainerCallback):
    """
    Callback for tracking and logging GPU energy consumption during training using NvidiaProfiler.
    Records energy usage at key training steps and saves results to a CSV file if nvidia-smi is available.

    Args:
        output_dir (str): Directory to save energy logs.
        nvidia_query_interval (int): Interval in milliseconds for querying nvidia-smi.

    Example:
        callback = EnergyCallback(output_dir="./logs", nvidia_query_interval=10)
        trainer = Trainer(..., callbacks=[callback])
        trainer.train()
    """

    prof: NvidiaProfiler | None

    def __init__(self, output_dir: str, nvidia_query_interval: int = 10):
        """
        Initialize the EnergyCallback and start NvidiaProfiler if available.

        Args:
            output_dir (str): Directory to save energy logs.
            nvidia_query_interval (int): Interval in milliseconds for querying nvidia-smi.
        """
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
        """
        Destructor to stop the profiler when the callback is deleted.
        """
        if self.prof is None:
            return
        self.prof.stop()

    def on_step_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the beginning of a training step.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("step_begin")

    def on_step_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the end of a training step.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("step_end")

    def on_epoch_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the beginning of an epoch.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("epoch_begin")

    def on_epoch_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the end of an epoch.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("epoch_end")

    def on_training_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the beginning of training.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("training_begin")

    def on_training_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at the end of training.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("training_end")

    def on_checkpoint(self, info: dict, trainer: "Trainer") -> None:
        """
        Record energy usage at checkpoint events.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        if self.prof is None:
            return
        self.prof.record_step("checkpoint")
