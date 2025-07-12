import pytest
import torch
from torch.utils.data import Dataset
from nnt.datasets.dataset import DataSplit
from nnt.validators.validator import Validator, ValidationArguments, PredictedBatch


@pytest.fixture
def mock_datasplit():
    # Create a sample DataSplit instance
    data_split = DataSplit()

    # Add some sample data with 'x' key as tensor, matching test expectations
    data_split.append({"x": torch.tensor([1.0])})
    data_split.append({"x": torch.tensor([2.0])})
    data_split.append({"x": torch.tensor([3.0])})
    data_split.append({"x": torch.tensor([4.0])})
    data_split.append({"x": torch.tensor([5.0])})

    return data_split


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cpu"

    def forward(self, x):
        return x + 1


class DummyMetric:
    def __init__(self):
        self.name = "dummy_metric"
        self.values = []

    def compute(self, predicted_batch):
        self.values.append(predicted_batch.prediction)

    def finalize(self):
        return torch.cat(self.values).mean().item()


class TestValidator(Validator):
    def model_predict(self, batch):
        prediction = self.model(batch["x"])
        return PredictedBatch(batch=batch, prediction=prediction)


def test_validator_validate_returns_metric_result(mock_datasplit):
    model = DummyModel()
    metric = DummyMetric()
    val_args = ValidationArguments(batch_size=2)
    validator = TestValidator(model, val_args, mock_datasplit, metrics=[metric])

    results = validator.validate()
    assert "dummy_metric" in results
    assert isinstance(results["dummy_metric"], float)
    assert abs(results["dummy_metric"] - 4.0) < 1e-5


def test_validator_batch_to_device_moves_tensor(mock_datasplit):
    model = DummyModel()
    val_args = ValidationArguments(batch_size=1)
    validator = TestValidator(model, val_args, mock_datasplit)
    batch = {"x": torch.tensor([1.0])}
    batch_on_device = validator._batch_to_device(batch)
    assert isinstance(batch_on_device["x"], torch.Tensor)
    assert batch_on_device["x"].device.type == "cpu"


def test_validator_prepare_data_sets_validation_batches(mock_datasplit):
    model = DummyModel()
    val_args = ValidationArguments(batch_size=2)
    validator = TestValidator(model, val_args, mock_datasplit)
    validator._prepare_data()
    assert validator.validation_batches is not None
    batches = list(validator.validation_batches)
    assert len(batches) == 3  # 5 items, batch_size=2 -> 3 batches


def test_validator_model_predict_not_implemented():
    class IncompleteValidator(Validator):
        pass

    model = DummyModel()
    val_args = ValidationArguments(batch_size=1)
    validator = IncompleteValidator(model, val_args, mock_datasplit)
    with pytest.raises(NotImplementedError):
        validator.model_predict({"x": torch.tensor([1.0])})
