import pytest
import pandas as pd
from nnt.datasets.dataset import DataSplit, Dataset


def test_datasplit():

    # Create a sample DataSplit instance
    data_split = DataSplit()

    # Add some sample data
    data_split.append({"id": 1, "value": "A"})
    data_split.append({"id": 2, "value": "B"})

    # Check if the data is correctly appended
    assert len(data_split) == 2

    # Access the first item
    assert data_split[0] == {"id": 1, "value": "A"}

    # Modify an item
    data_split[0] = {"id": 1, "value": "C"}
    assert data_split[0] == {"id": 1, "value": "C"}

    # Delete an item
    del data_split[1]
    assert len(data_split) == 1

    # Check if an item exists
    assert {"id": 1, "value": "C"} in data_split

    # Iterate over the DataSplit
    items = [item for item in data_split]
    assert items == [{"id": 1, "value": "C"}]

    # Test unified columns
    unified_data = [{"id": 3}, {"value": "D"}]
    data_split.unify_columns(unified_data)
    assert len(data_split) == 2
    assert data_split[0] == {"id": 3, "value": None}

    # Test from_iterable method
    iterable_data = [{"id": 4, "value": "E"}, {"id": 5, "value": "F"}]
    new_data_split = DataSplit.from_iterable(iterable_data)
    assert len(new_data_split) == 2
    assert new_data_split[0] == {"id": 4, "value": "E"}

    # Test from_dict method
    dict_data = {"train": [{"id": 6, "value": "G"}], "test": [{"id": 7, "value": "H"}]}
    data_split_from_dict = DataSplit.from_dict(dict_data)
    assert len(data_split_from_dict) == 2
    assert data_split_from_dict[0] == {"train": {"id": 6, "value": "G"}}
    assert data_split_from_dict[1] == {"test": {"id": 7, "value": "H"}}

    # Test from_json method
    json_data = '[{"id": 8, "value": "I"}, {"id": 9, "value": "J"}]'
    data_split_from_json = DataSplit.from_json(json_data)
    assert len(data_split_from_json) == 2
    assert data_split_from_json[0] == {"id": 8, "value": "I"}
    assert data_split_from_json[1] == {"id": 9, "value": "J"}

    # Test from_pandas method
    df = pd.DataFrame([{"id": 10, "value": "K"}, {"id": 11, "value": "L"}])
    data_split_from_pandas = DataSplit.from_pandas(df)
    assert len(data_split_from_pandas) == 2
    assert data_split_from_pandas[0] == {"id": 10, "value": "K"}
    assert data_split_from_pandas[1] == {"id": 11, "value": "L"}
