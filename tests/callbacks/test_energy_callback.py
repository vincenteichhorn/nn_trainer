import os
import tempfile
import pytest
import warnings
from unittest import mock
from nnt.callbacks.energy_callback import EnergyCallback


class DummyTrainer:
    pass


@pytest.fixture
def dummy_trainer():
    return DummyTrainer()


def test_energy_callback_initializes_profiler_if_nvidia_smi_available(monkeypatch):
    # Simulate nvidia-smi available
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (0, "NVIDIA-SMI"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("nnt.callbacks.energy_callback.NvidiaProfiler") as MockProfiler:
            cb = EnergyCallback(output_dir=tmpdir)
            assert cb.prof is not None
            MockProfiler.assert_called_once()
            cb.prof.start.assert_called_once()


def test_energy_callback_warns_if_nvidia_smi_not_available(monkeypatch):
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (1, "not found"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with warnings.catch_warnings(record=True) as w:
            cb = EnergyCallback(output_dir=tmpdir)
            assert cb.prof is None
            assert any("EnergyCallback will not be active" in str(warn.message) for warn in w)


def test_energy_callback_calls_prof_record_step(monkeypatch, dummy_trainer):
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (0, "NVIDIA-SMI"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("nnt.callbacks.energy_callback.NvidiaProfiler") as MockProfiler:
            mock_prof = MockProfiler.return_value
            cb = EnergyCallback(output_dir=tmpdir)
            cb.prof = mock_prof
            for method, arg in [
                (cb.on_step_begin, "step_begin"),
                (cb.on_step_end, "step_end"),
                (cb.on_epoch_begin, "epoch_begin"),
                (cb.on_epoch_end, "epoch_end"),
                (cb.on_training_begin, "training_begin"),
                (cb.on_training_end, "training_end"),
                (cb.on_checkpoint, "checkpoint"),
            ]:
                method({}, dummy_trainer)
                mock_prof.record_step.assert_called_with(arg)


def test_energy_callback_methods_do_nothing_if_prof_none(monkeypatch, dummy_trainer):
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (1, "not found"))
    with tempfile.TemporaryDirectory() as tmpdir:
        cb = EnergyCallback(output_dir=tmpdir)
        cb.prof = None
        # Should not raise
        cb.on_step_begin({}, dummy_trainer)
        cb.on_step_end({}, dummy_trainer)
        cb.on_epoch_begin({}, dummy_trainer)
        cb.on_epoch_end({}, dummy_trainer)
        cb.on_training_begin({}, dummy_trainer)
        cb.on_training_end({}, dummy_trainer)
        cb.on_checkpoint({}, dummy_trainer)


def test_energy_callback_del_calls_prof_stop(monkeypatch):
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (0, "NVIDIA-SMI"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("nnt.callbacks.energy_callback.NvidiaProfiler") as MockProfiler:
            cb = EnergyCallback(output_dir=tmpdir)
            cb.prof = MockProfiler.return_value
            cb.prof.stop = mock.Mock()
            del cb
            # Can't assert after del, but no error should occur


def test_energy_callback_creates_energy_file(monkeypatch, dummy_trainer):
    monkeypatch.setattr("subprocess.getstatusoutput", lambda cmd: (0, "NVIDIA-SMI"))
    with tempfile.TemporaryDirectory() as tmpdir:
        with mock.patch("nnt.callbacks.energy_callback.NvidiaProfiler") as MockProfiler:
            mock_prof = MockProfiler.return_value
            # Simulate record_step writing to file
            energy_file = os.path.join(tmpdir, "energy_log.csv")

            def fake_record_step(event):
                with open(energy_file, "a") as f:
                    f.write(f"{event}\n")

            mock_prof.record_step.side_effect = fake_record_step
            cb = EnergyCallback(output_dir=tmpdir)
            cb.prof = mock_prof
            cb.on_step_end({}, dummy_trainer)
            assert os.path.isfile(energy_file)
            with open(energy_file, "r") as f:
                content = f.read()
                assert "step_end" in content
