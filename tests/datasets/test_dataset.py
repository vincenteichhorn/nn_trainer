import pytest

from nnt.datasets.dataset import DataSplit, Dataset


def mock_datasplit():
    # Create a sample DataSplit instance
    data_split = DataSplit()

    # Add some sample data
    data_split.append({"id": 1, "value": "A"})
    data_split.append({"id": 2, "value": "B"})

    return data_split


class ConcreteDataset(Dataset):
    def load(self):
        # No-op for testing, as data is already initialized
        pass


def test_dataset():
    datasplit = mock_datasplit()
    dataset = ConcreteDataset(train=datasplit)

    # Check if the dataset contains the DataSplit
    assert isinstance(dataset["train"], DataSplit)

    # Access the DataSplit through the dataset
    assert dataset["train"][0] == {"id": 1, "value": "A"}

    # Modify the DataSplit through the dataset
    dataset["train"][0] = {"id": 1, "value": "C"}
    assert dataset["train"][0] == {"id": 1, "value": "C"}

    # Delete an item in the DataSplit through the dataset
    del dataset["train"][1]
    assert len(dataset["train"]) == 1

    # Check if an item exists in the DataSplit through the dataset
    assert {"id": 1, "value": "C"} in dataset["train"]

    # Iterate over the DataSplit through the dataset
    items = [item for item in dataset["train"]]
    assert items == [{"id": 1, "value": "C"}]

    # Test setting a new DataSplit
    new_data_split = DataSplit([{"id": 3, "value": "D"}])
    dataset["test"] = new_data_split
    assert dataset["test"][0] == {"id": 3, "value": "D"}

    # Test deleting a DataSplit
    del dataset["test"]
    assert "test" not in dataset._datasplits

    # Test accessing a non-existent DataSplit using pytest
    with pytest.raises(KeyError):
        _ = dataset["non_existent"]
