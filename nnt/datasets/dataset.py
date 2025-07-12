from abc import abstractmethod
import json
from typing import Dict, Iterable, Union


class DataSplit:

    def __init__(self, data: Iterable[Dict] = None):
        self.data = data
        self.name = None
        self.unify_columns(data)

    def set_name(self, name: str):
        """
        Set the name of the DataSplit.
        Args:
            name (str): The name to set.
        """
        self.name = name

    def add_sample(self, sample: Dict):
        if self.data is None:
            self.data = []
        if not isinstance(sample, Dict):
            raise TypeError("Sample must be a dictionary.")
        self.data.append(sample)

    def append(self, sample: Dict):
        self.add_sample(sample)

    def _get_columns(self):
        columns = set()
        if self.data is not None:
            for sample in self.data:
                if isinstance(sample, Dict):
                    columns.update(sample.keys())
        return list(columns)

    def unify_columns(self, data: Iterable[Dict]):
        if data is None:
            return None
        columns = self._get_columns()
        unified_data = []
        for sample in data:
            unified_sample = {col: sample.get(col, None) for col in columns}
            unified_data.append(unified_sample)
        self.data = unified_data

    def __getitem__(self, index):
        if self.data is None:
            raise IndexError("Data is not available.")
        return self.data[index]

    def __setitem__(self, index, value):
        if self.data is None:
            raise IndexError("Data is not available.")
        self.data[index] = value

    def __delitem__(self, index):
        if self.data is None:
            raise IndexError("Data is not available.")
        del self.data[index]

    def __contains__(self, item):
        if self.data is None:
            return False
        return item in self.data

    def __repr__(self):
        return f"DataSplit(columns={self._get_columns()}, num_samples={len(self.data) if self.data else 0})"

    def __iter__(self):
        return iter(self.data) if self.data is not None else iter([])

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    @classmethod
    def from_iterable(cls, iterable: Iterable[Dict]):
        return cls(data=iterable)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Iterable[Dict]]):
        data_split = cls()
        for key, value in data_dict.items():
            if isinstance(value, Iterable):
                for item in value:
                    data_split.append({key: item})
            else:
                raise TypeError(f"Value for key '{key}' must be an iterable of dictionaries.")
        return data_split

    @classmethod
    def from_pandas(cls, df):
        if not hasattr(df, "to_dict"):
            raise TypeError("Input must be a pandas DataFrame.")
        return cls(data=df.to_dict(orient="records"))

    @classmethod
    def from_json(cls, json_data):
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        if isinstance(json_data, list):
            return cls(data=json_data)
        elif isinstance(json_data, dict):
            return cls.from_dict(json_data)
        else:
            raise TypeError("Input must be a JSON string, list, or dictionary.")


class Dataset:

    def __init__(self, datasplits: Union[Dict[str, Union[DataSplit, Iterable[Dict]]], None] = None, **kwargs):
        self._datasplits = datasplits or {}
        self._is_loaded = False

        if kwargs:
            for name, data in kwargs.items():
                if isinstance(data, DataSplit):
                    self._datasplits[name] = data
                elif isinstance(data, Iterable):
                    self._datasplits[name] = DataSplit(data)

                else:
                    raise TypeError(f"Invalid type for {name}: {type(data)}. Must be DataSplit or iterable of dictionaries.")

    @abstractmethod
    def load(self):
        """
        Load datasets instance.
        Example:
            return datasets.load_dataset("my_dataset")
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def load_if_not_loaded(self):
        """
        Load the dataset if it has not been loaded yet.
        """
        if not self._is_loaded:
            self._is_loaded = True
            self.load()

    def splits(self) -> Dict[str, DataSplit]:
        """
        Returns a dictionary of DataSplit instances for each split.
        """
        return self._datasplits

    def update_split_names(self):
        """
        Update the names of the DataSplit instances based on their keys.
        """
        for name, split in self._datasplits.items():
            if isinstance(split, DataSplit):
                split.set_name(name)

    def __getitem__(self, name: str) -> DataSplit:
        self.load_if_not_loaded()
        if name not in self._datasplits:
            raise KeyError(f"DataSplit '{name}' does not exist.")
        return self._datasplits[name]

    def __setitem__(self, name: str, data: Union[DataSplit, Iterable[Dict]]):
        self.load_if_not_loaded()
        if not isinstance(data, (DataSplit, Iterable)):
            raise TypeError("Data must be a DataSplit instance or an iterable of dictionaries.")
        self._datasplits[name] = DataSplit(data) if isinstance(data, Iterable) else data
        self.update_split_names()

    def __delitem__(self, name: str):
        if name not in self._datasplits:
            raise KeyError(f"DataSplit '{name}' does not exist.")
        del self._datasplits[name]

    def __contains__(self, name: str) -> bool:
        self.load_if_not_loaded()
        return name in self._datasplits

    def __repr__(self):
        return f"Dataset(train={self.train}, validation={self.validation}, test={self.test})"
