import os
import tempfile
import pytest
from nnt.callbacks.logging_callback import LoggingCallback


class DummyTrainer:
    pass


@pytest.fixture
def dummy_trainer():
    return DummyTrainer()


def test_logging_callback_creates_log_file(dummy_trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = LoggingCallback(output_dir=tmpdir)
        info = {"step": 1, "loss": 0.123, "current_batch": "should_skip"}
        callback.on_step_end(info, dummy_trainer)
        expected_file = os.path.join(tmpdir, "train_log.csv")
        assert os.path.isfile(expected_file)
        with open(expected_file, "r") as f:
            content = f.read()
            assert "step" in content
            assert "loss" in content
            assert "current_batch" not in content


def test_logging_callback_appends_rows(dummy_trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = LoggingCallback(output_dir=tmpdir)
        info1 = {"step": 1, "loss": 0.1, "current_batch": "skip"}
        info2 = {"step": 2, "loss": 0.2, "current_batch": "skip"}
        callback.on_step_end(info1, dummy_trainer)
        callback.on_step_end(info2, dummy_trainer)
        expected_file = os.path.join(tmpdir, "train_log.csv")
        with open(expected_file, "r") as f:
            lines = f.readlines()
            assert len(lines) >= 3  # header + 2 rows
            assert "1,0.1" in lines[1]
            assert "2,0.2" in lines[2]


def test_logging_callback_sets_columns_once(dummy_trainer):
    with tempfile.TemporaryDirectory() as tmpdir:
        callback = LoggingCallback(output_dir=tmpdir)
        info = {"step": 1, "loss": 0.123, "current_batch": "skip"}
        callback.on_step_end(info, dummy_trainer)
        # Should not raise or reset columns on second call
        callback.on_step_end(info, dummy_trainer)
        assert callback.writer_has_set_columns is True
