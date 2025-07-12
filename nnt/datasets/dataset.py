from abc import abstractmethod
import json
from typing import Dict, List, Any, Iterable, Union


class DataSplit:
    """
    Container for a split of dataset samples, with unified columns and utility methods for access and manipulation.

    Args:
        data (Iterable[Dict], optional): Iterable of sample dictionaries.

    Example:
        split = DataSplit([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        print(split[0])
        split.append({'a': 5, 'b': 6})
        print(len(split))
    """

    data: List[Dict[str, Any]]
    name: str

    def __init__(self, data: List[Dict[str, Any]] = None):
        """
        Initialize the DataSplit and unify columns across samples.
        """
        self.data = data
        self.name = None
        self.unify_columns(data)

    def set_name(self, name: str) -> None:
        """
        Set the name of the DataSplit.

        Args:
            name (str): The name to set.
        """
        self.name = name

    def add_sample(self, sample: Dict) -> None:
        """
        Add a sample dictionary to the split.

        Args:
            sample (Dict): Sample to add.
        Raises:
            TypeError: If sample is not a dictionary.
        """
        if self.data is None:
            self.data = []
        if not isinstance(sample, Dict):
            raise TypeError("Sample must be a dictionary.")
        self.data.append(sample)

    def append(self, sample: Dict) -> None:
        """
        Append a sample to the split (alias for add_sample).

        Args:
            sample (Dict): Sample to append.
        """
        self.add_sample(sample)

    def _get_columns(self) -> list:
        """
        Get the list of column names present in the split.

        Returns:
            list: List of column names.
        """
        columns = set()
        if self.data is not None:
            for sample in self.data:
                if isinstance(sample, Dict):
                    columns.update(sample.keys())
        return list(columns)

    def unify_columns(self, data: Iterable[Dict]) -> None:
        """
        Ensure all samples have the same columns, filling missing values with None.

        Args:
            data (Iterable[Dict]): Iterable of sample dictionaries.
        """
        if data is None:
            return None
        columns = self._get_columns()
        unified_data = []
        for sample in data:
            unified_sample = {col: sample.get(col, None) for col in columns}
            unified_data.append(unified_sample)
        self.data = unified_data

    def __getitem__(self, index: int) -> dict:
        """
        Get a sample by index.

        Args:
            index (int): Index of the sample.
        Returns:
            dict: Sample at the given index.
        Raises:
            IndexError: If data is not available.
        """
        if self.data is None:
            raise IndexError("Data is not available.")
        return self.data[index]

    def __setitem__(self, index: int, value: dict) -> None:
        """
        Set a sample at a given index.

        Args:
            index (int): Index to set.
            value (dict): Sample to set.
        Raises:
            IndexError: If data is not available.
        """
        if self.data is None:
            raise IndexError("Data is not available.")
        self.data[index] = value

    def __delitem__(self, index: int) -> None:
        """
        Delete a sample at a given index.

        Args:
            index (int): Index to delete.
        Raises:
            IndexError: If data is not available.
        """
        if self.data is None:
            raise IndexError("Data is not available.")
        del self.data[index]

    def __contains__(self, item: dict) -> bool:
        """
        Check if a sample is in the split.

        Args:
            item (dict): Sample to check.
        Returns:
            bool: True if item is in the split.
        """
        if self.data is None:
            return False
        return item in self.data

    def __repr__(self) -> str:
        """
        String representation of the DataSplit.
        """
        return f"DataSplit(columns={self._get_columns()}, num_samples={len(self.data) if self.data else 0})"

    def __iter__(self):
        """
        Iterate over samples in the split.
        """
        return iter(self.data) if self.data is not None else iter([])

    def __len__(self) -> int:
        """
        Get the number of samples in the split.
        """
        return len(self.data) if self.data is not None else 0

    @classmethod
    def from_iterable(cls, iterable: Iterable[Dict]) -> "DataSplit":
        """
        Create a DataSplit from an iterable of sample dictionaries.
        """
        return cls(data=iterable)

    @classmethod
    def from_dict(cls, data_dict: Dict[str, Iterable[Dict]]) -> "DataSplit":
        """
        Create a DataSplit from a dictionary of key to iterable of samples.

        Args:
            data_dict (Dict[str, Iterable[Dict]]): Dictionary mapping keys to iterables of samples.
        Returns:
            DataSplit: The created DataSplit.
        Raises:
            TypeError: If value is not an iterable of dictionaries.
        """
        data_split = cls()
        for key, value in data_dict.items():
            if isinstance(value, Iterable):
                for item in value:
                    data_split.append({key: item})
            else:
                raise TypeError(f"Value for key '{key}' must be an iterable of dictionaries.")
        return data_split

    @classmethod
    def from_pandas(cls, df) -> "DataSplit":
        """
        Create a DataSplit from a pandas DataFrame.

        Args:
            df (DataFrame): Pandas DataFrame.
        Returns:
            DataSplit: The created DataSplit.
        Raises:
            TypeError: If input is not a DataFrame.
        """
        if not hasattr(df, "to_dict"):
            raise TypeError("Input must be a pandas DataFrame.")
        return cls(data=df.to_dict(orient="records"))

    @classmethod
    def from_json(cls, json_data) -> "DataSplit":
        """
        Create a DataSplit from JSON data (string, list, or dict).

        Args:
            json_data (str, list, or dict): JSON data to load.
        Returns:
            DataSplit: The created DataSplit.
        Raises:
            TypeError: If input is not a valid JSON type.
        """
        if isinstance(json_data, str):
            json_data = json.loads(json_data)
        if isinstance(json_data, list):
            return cls(data=json_data)
        elif isinstance(json_data, dict):
            return cls.from_dict(json_data)
        else:
            raise TypeError("Input must be a JSON string, list, or dictionary.")


class Dataset:
    """
    Container for multiple DataSplit instances, representing different splits of a dataset (e.g., train, validation, test).

    Args:
        datasplits (dict, optional): Dictionary mapping split names to DataSplit or iterable of samples.
        **kwargs: Additional splits as keyword arguments.

    Example:
        train_split = DataSplit([{'x': 1}, {'x': 2}])
        ds = Dataset(train=train_split)
        print(ds['train'])
    """

    _datasplits: Dict[str, DataSplit]
    _is_loaded: bool

    def __init__(self, datasplits: Union[Dict[str, DataSplit], None] = None, **kwargs):
        """
        Initialize the Dataset with given splits.
        """
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
        Load the dataset instance. Should be implemented by subclasses.

        Example:
            return datasets.load_dataset("my_dataset")
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def load_if_not_loaded(self) -> None:
        """
        Load the dataset if it has not been loaded yet.
        """
        if not self._is_loaded:
            self._is_loaded = True
            self.load()

    def splits(self) -> Dict[str, DataSplit]:
        """
        Get a dictionary of DataSplit instances for each split.

        Returns:
            dict: Dictionary of split names to DataSplit instances.
        """
        return self._datasplits

    def update_split_names(self) -> None:
        """
        Update the names of the DataSplit instances based on their keys.
        """
        for name, split in self._datasplits.items():
            if isinstance(split, DataSplit):
                split.set_name(name)

    def __getitem__(self, name: str) -> DataSplit:
        """
        Get a DataSplit by name, loading the dataset if needed.

        Args:
            name (str): Name of the split.
        Returns:
            DataSplit: The requested DataSplit.
        Raises:
            KeyError: If the split does not exist.
        """
        self.load_if_not_loaded()
        if name not in self._datasplits:
            raise KeyError(f"DataSplit '{name}' does not exist.")
        return self._datasplits[name]

    def __setitem__(self, name: str, data: Union[DataSplit, Iterable[Dict]]) -> None:
        """
        Set a DataSplit by name, loading the dataset if needed.

        Args:
            name (str): Name of the split.
            data (DataSplit or iterable): Data to set for the split.
        Raises:
            TypeError: If data is not a DataSplit or iterable of dictionaries.
        """
        self.load_if_not_loaded()
        if not isinstance(data, (DataSplit, Iterable)):
            raise TypeError("Data must be a DataSplit instance or an iterable of dictionaries.")
        self._datasplits[name] = DataSplit(data) if isinstance(data, Iterable) else data
        self.update_split_names()

    def __delitem__(self, name: str) -> None:
        """
        Delete a DataSplit by name.

        Args:
            name (str): Name of the split to delete.
        Raises:
            KeyError: If the split does not exist.
        """
        if name not in self._datasplits:
            raise KeyError(f"DataSplit '{name}' does not exist.")
        del self._datasplits[name]

    def __contains__(self, name: str) -> bool:
        """
        Check if a split exists in the dataset.

        Args:
            name (str): Name of the split.
        Returns:
            bool: True if the split exists.
        """
        self.load_if_not_loaded()
        return name in self._datasplits

    def __repr__(self) -> str:
        """
        String representation of the Dataset.
        """
        return f"Dataset(train={self.train}, validation={self.validation}, test={self.test})"
