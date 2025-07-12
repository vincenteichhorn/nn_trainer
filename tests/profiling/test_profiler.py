import pytest
from nnt.profiling.profiler import Profiler
from datetime import datetime, timedelta


class TestProfiler(Profiler):
    """Concrete Profiler for testing (since Profiler is abstract in docstring only)"""

    def __init__(self):
        super().__init__()


def test_profiler_initialization_records_init():
    profiler = TestProfiler()
    assert len(profiler.record_steps) == 1
    assert profiler.record_steps[0][1] == "__init__"
    assert isinstance(profiler.record_steps[0][0], datetime)


def test_record_step_adds_step():
    profiler = TestProfiler()
    profiler.record_step("step1")
    assert profiler.record_steps[-1][1] == "step1"
    assert isinstance(profiler.record_steps[-1][0], datetime)
    assert len(profiler.record_steps) == 2


def test_record_context_records_steps():
    profiler = TestProfiler()
    with profiler.record_context("context1"):
        pass
    # Should record "context1" and "__other__"
    names = [step[1] for step in profiler.record_steps]
    assert "context1" in names
    assert "__other__" in names
    # "__init__", "context1", "__other__"
    assert names == ["__init__", "context1", "__other__"]


def test_record_context_records_steps_on_exception():
    profiler = TestProfiler()
    with pytest.raises(ValueError):
        with profiler.record_context("context2"):
            raise ValueError("Test exception")
    names = [step[1] for step in profiler.record_steps]
    assert names == ["__init__", "context2", "__other__"]


def test_multiple_steps_and_contexts():
    profiler = TestProfiler()
    profiler.record_step("stepA")
    with profiler.record_context("contextA"):
        profiler.record_step("stepB")
    names = [step[1] for step in profiler.record_steps]
    assert names == ["__init__", "stepA", "contextA", "stepB", "__other__"]
