import pytest
import torch
from nnt.profiling.torch_profiler import TorchProfiler
import pandas as pd


import torch.nn as nn
import torch.optim as optim


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def dummy_model():
    return DummyModel()


@pytest.fixture
def dummy_input():
    return torch.randn(4, 10)


def test_profiler_basic_usage(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    df = prof.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert "name" in df.columns
    assert "flops" in df.columns
    assert df.shape[0] > 0


def test_profiler_summary(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    summary_df = prof.summary()
    assert isinstance(summary_df, pd.DataFrame)
    assert "flops" in summary_df.columns
    assert summary_df.index.is_unique


def test_profiler_totals(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    totals = prof.totals()
    assert isinstance(totals, pd.Series)
    for col in prof.numeric_columns:
        assert col in totals.index


def test_profiler_total_time(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    cpu_time = prof.get_total_time(device="CPU")
    assert isinstance(cpu_time, float)
    assert cpu_time >= 0
    if torch.cuda.is_available():
        gpu_time = prof.get_total_time(device="CUDA")
        assert isinstance(gpu_time, float)
        assert gpu_time >= 0


def test_profiler_total_flops(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    total_flops = prof.get_total_flops()
    assert isinstance(total_flops, int)
    assert total_flops >= 0


def test_profiler_flops_by_step(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    df = prof.get_flops_by_step()
    assert isinstance(df, pd.DataFrame)
    assert "flops" in df.columns


def test_profiler_time_by_step(dummy_model, dummy_input):
    with TorchProfiler() as prof:
        out = dummy_model(dummy_input)
        loss = out.sum()
        loss.backward()
    df = prof.get_time_by_step()
    assert isinstance(df, pd.DataFrame)
    assert "cpu_time" in df.columns
    assert "gpu_time" in df.columns
