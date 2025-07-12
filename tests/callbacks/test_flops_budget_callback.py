import torch
import pytest
from nnt.callbacks.flops_budget_callback import FLOPsBudgetControllCallback
import os
import tempfile

import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def model():
    return SimpleModel()


class DummyLoss:
    def backward(self):
        pass


class DummyModel(torch.nn.Module):
    def forward(self, x):
        return type("DummyOutput", (), {"loss": DummyLoss()})()


class DummyTrainer:
    def __init__(self, model):
        self.model = model
        self.stopped = False

    def stop(self):
        self.stopped = True


@pytest.fixture
def dummy_trainer():
    return DummyTrainer(DummyModel())


def test_flops_budget_callback_creates_file(dummy_trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = FLOPsBudgetControllCallback(output_dir=tmpdir, budget=1000)
        info = {"current_batch": {"x": torch.randn(2, 10)}, "step": 1}
        callback.on_step_begin(info, dummy_trainer)
        expected_file = os.path.join(tmpdir, "flops_budget_log.csv")
        assert os.path.isfile(expected_file)
        with open(expected_file, "r") as f:
            content = f.read()
            assert "step_flops" in content
            assert "cumulative_flops" in content
