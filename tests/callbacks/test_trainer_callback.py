import pytest
from nnt.callbacks.trainer_callback import TrainerCallback


class DummyTrainer:
    pass


class ConcreteTrainerCallback(TrainerCallback):
    def on_step_begin(self, info, trainer):
        return "step_begin"

    def on_step_end(self, info, trainer):
        return "step_end"

    def on_epoch_begin(self, info, trainer):
        return "epoch_begin"

    def on_epoch_end(self, info, trainer):
        return "epoch_end"

    def on_training_begin(self, info, trainer):
        return "training_begin"

    def on_training_end(self, info, trainer):
        return "training_end"

    def on_checkpoint(self, info, trainer):
        return "checkpoint"


def test_concrete_trainer_callback_methods():
    cb = ConcreteTrainerCallback()
    trainer = DummyTrainer()
    info = {"step": 1, "epoch": 2}

    assert cb.on_step_begin(info, trainer) == "step_begin"
    assert cb.on_step_end(info, trainer) == "step_end"
    assert cb.on_epoch_begin(info, trainer) == "epoch_begin"
    assert cb.on_epoch_end(info, trainer) == "epoch_end"
    assert cb.on_training_begin(info, trainer) == "training_begin"
    assert cb.on_training_end(info, trainer) == "training_end"
    assert cb.on_checkpoint(info, trainer) == "checkpoint"
