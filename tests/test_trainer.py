import os
from types import SimpleNamespace
import torch
import pytest
from torch.utils.data import Dataset
from nnt.trainer import Trainer, TrainingArguments
from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.callbacks.logging_callback import LoggingCallback


class DummyDataset(Dataset):
    def __init__(self, length=10):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return a dict compatible with ToyLanguageModel or similar
        return {"input_ids": torch.tensor([idx]), "labels": torch.tensor([idx])}


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, input_ids, labels):
        # Ensure input_ids is float and requires grad for autograd
        input_ids = input_ids.float().unsqueeze(-1)
        input_ids.requires_grad_()
        logits = self.linear(input_ids)
        loss = torch.mean((logits.squeeze(-1) - labels.float()) ** 2)
        return SimpleNamespace(loss=loss, logits=logits)


class DummyCallback(TrainerCallback):
    def __init__(self):
        self.calls = []

    def on_step_begin(self, info, trainer):
        self.calls.append("step_begin")

    def on_step_end(self, info, trainer):
        self.calls.append("step_end")

    def on_epoch_begin(self, info, trainer):
        self.calls.append("epoch_begin")

    def on_epoch_end(self, info, trainer):
        self.calls.append("epoch_end")

    def on_training_begin(self, info, trainer):
        self.calls.append("training_begin")

    def on_training_end(self, info, trainer):
        self.calls.append("training_end")

    def on_checkpoint(self, info, trainer):
        self.calls.append("checkpoint")


@pytest.fixture
def dummy_dataset():
    return DummyDataset(length=5)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def training_args(tmp_path):
    return TrainingArguments(
        num_epochs=2,
        batch_size=2,
        data_collator=None,
        learning_rate=1e-3,
        weight_decay=0.0,
        monitor_strategy="steps",
        monitor_every=2,
        checkpoint_strategy="steps",
        checkpoint_every=2,
        model_save_function=None,
    )


def test_trainer_runs_and_saves_model(tmp_path, dummy_dataset, dummy_model, training_args):
    output_dir = str(tmp_path)
    trainer = Trainer(
        output_dir=output_dir,
        model=dummy_model,
        training_args=training_args,
        train_data=dummy_dataset,
        callbacks=[],
    )
    trained_model = trainer.train()
    assert isinstance(trained_model, DummyModel)
    # Check model save
    assert os.path.isfile(os.path.join(output_dir, "model", "model.pth"))
    # Check checkpoint save
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    found = False
    for root, dirs, files in os.walk(checkpoint_dir):
        if any("model.pth" in f for f in files):
            found = True
            break
    assert found


def test_trainer_calls_callbacks(tmp_path, dummy_dataset, dummy_model, training_args):
    cb = DummyCallback()
    trainer = Trainer(
        output_dir=str(tmp_path),
        model=dummy_model,
        training_args=training_args,
        train_data=dummy_dataset,
        callbacks=[cb],
    )
    trainer.train()
    # Check that all callback events were called at least once
    for event in ["step_begin", "step_end", "epoch_begin", "epoch_end", "training_begin", "training_end", "checkpoint"]:
        assert event in cb.calls


def test_trainer_stop_stops_training(tmp_path, dummy_dataset, dummy_model, training_args):
    class StopCallback(DummyCallback):
        def on_step_begin(self, info, trainer):
            super().on_step_begin(info, trainer)
            trainer.stop()

    cb = StopCallback()
    trainer = Trainer(
        output_dir=str(tmp_path),
        model=dummy_model,
        training_args=training_args,
        train_data=dummy_dataset,
        callbacks=[cb],
    )
    trainer.train()
    # Should only run one step before stopping
    assert cb.calls.count("step_begin") == 1
    assert cb.calls.count("training_end") == 1


def test_trainer_with_logging_callback_creates_log_file(tmp_path, dummy_dataset, dummy_model, training_args):
    cb = LoggingCallback(output_dir=str(tmp_path))
    trainer = Trainer(
        output_dir=str(tmp_path),
        model=dummy_model,
        training_args=training_args,
        train_data=dummy_dataset,
        callbacks=[cb],
    )
    trainer.train()
    log_file = os.path.join(str(tmp_path), "train_log.csv")
    assert os.path.isfile(log_file)
    with open(log_file, "r") as f:
        content = f.read()
        assert "step" in content or "global_step" in content


def test_trainer_custom_model_save_function(tmp_path, dummy_dataset, dummy_model, training_args):
    saved = {}

    def custom_save(model, folder):
        saved["called"] = True
        saved["folder"] = folder

    training_args.model_save_function = custom_save
    trainer = Trainer(
        output_dir=str(tmp_path),
        model=dummy_model,
        training_args=training_args,
        train_data=dummy_dataset,
        callbacks=[],
    )
    trainer.train()
    assert saved["called"]
    assert "model" in saved["folder"] or "checkpoint" in saved["folder"]
