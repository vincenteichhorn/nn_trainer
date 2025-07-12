from abc import abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from nnt.trainer import Trainer


class TrainerCallback:

    @abstractmethod
    def on_step_begin(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_step_end(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_epoch_begin(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_epoch_end(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_training_begin(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_training_end(self, info: dict, trainer: "Trainer"):
        pass

    @abstractmethod
    def on_checkpoint(self, info: dict, trainer: "Trainer"):
        pass
