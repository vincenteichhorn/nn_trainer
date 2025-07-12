import pytest
from datetime import datetime
from nnt.profiling.nvidia_profiler import NvidiaProfiler
from plotly import graph_objects as go


@pytest.fixture
def dummy_data():
    # gpu_id, timestamp, power, memory, record_step
    return [
        (0, datetime(2024, 1, 1, 12, 0, 0, 0), 100.0, 5000.0, "step1"),
        (0, datetime(2024, 1, 1, 12, 0, 1, 0), 110.0, 5100.0, "step1"),
        (0, datetime(2024, 1, 1, 12, 0, 2, 0), 120.0, 5200.0, "step2"),
        (0, datetime(2024, 1, 1, 12, 0, 3, 0), 130.0, 5300.0, "step2"),
    ]


@pytest.fixture
def profiler(dummy_data):
    prof = NvidiaProfiler()
    prof.data = dummy_data
    return prof


def test_to_pandas(profiler):
    df = profiler.to_pandas()
    assert not df.empty
    assert set(df.columns) == {"gpu_id", "timestamp", "power", "memory", "record_step"}


def test_get_profiled_gpus(profiler):
    gpus = profiler.get_profiled_gpus()
    assert gpus == [0]


def test_get_total_energy(profiler):
    energy = profiler.get_total_energy()
    assert isinstance(energy, float)
    assert energy > 0


def test_get_total_time(profiler):
    total_time = profiler.get_total_time()
    assert total_time == 3.0


def test_get_max_memory(profiler):
    max_mem = profiler.get_max_memory()
    assert max_mem == 5300.0


def test_get_mean_memory(profiler):
    mean_mem = profiler.get_mean_memory()
    assert mean_mem == pytest.approx((5000.0 + 5100.0 + 5200.0 + 5300.0) / 4)


def test_get_total_energy_with_record_steps(profiler):
    energy = profiler.get_total_energy(record_steps=["step1"])
    assert energy > 0


def test_get_total_time_with_record_steps(profiler):
    time = profiler.get_total_time(record_steps=["step2"])
    assert time == 1.0


def test_get_max_memory_with_record_steps(profiler):
    max_mem = profiler.get_max_memory(record_steps=["step2"])
    assert max_mem == 5300.0


def test_get_mean_memory_with_record_steps(profiler):
    mean_mem = profiler.get_mean_memory(record_steps=["step1"])
    assert mean_mem == pytest.approx((5000.0 + 5100.0) / 2)


def test_time_series_plot_returns_figure(profiler):

    fig = profiler.get_time_series_plot()
    assert isinstance(fig, go.Figure)


def test_from_cache(monkeypatch, dummy_data):
    class DummyResultHandler:
        def get_all(self):
            return dummy_data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def set_columns(self, *a, **kw):
            pass

    monkeypatch.setattr("nnt.profiling.nvidia_profiler.FileCacheResultHandler", lambda *a, **kw: DummyResultHandler())
    prof = NvidiaProfiler.from_cache("dummy.csv")
    assert prof.data == dummy_data
