from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nnt.trainer import Trainer


class TrainerCallback:
    """
    Abstract base class for training callbacks.
    Defines hooks for key training events to be implemented by subclasses.

    Example:
        class MyCallback(TrainerCallback):
            def on_step_begin(self, info, trainer):
                print("Step begin")
            # Implement other methods...
    """

    @abstractmethod
    def on_step_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the beginning of each training step.

        Args:
            info (dict): Training info for the step.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_step_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the end of each training step.

        Args:
            info (dict): Training info for the step.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_epoch_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the beginning of each epoch.

        Args:
            info (dict): Training info for the epoch.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_epoch_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the end of each epoch.

        Args:
            info (dict): Training info for the epoch.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_training_begin(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the beginning of training.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_training_end(self, info: dict, trainer: "Trainer") -> None:
        """
        Called at the end of training.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        pass

    @abstractmethod
    def on_checkpoint(self, info: dict, trainer: "Trainer") -> None:
        """
        Called when a checkpoint is saved during training.

        Args:
            info (dict): Training info.
            trainer (Trainer): Trainer instance.
        """
        pass
