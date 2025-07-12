from abc import ABC
from itertools import zip_longest
from multiprocessing import Manager, Process, Queue
import multiprocessing
import os
from typing import Any, Callable, List, Tuple, Union
import warnings


class ResultHandler(ABC):
    """
    Abstract base class for multiprocessing result handlers.
    Works like a queue but can be implemented in different ways (e.g., in-memory, file-based).

    Attributes:
        column_names (Tuple[str, ...]): Names of columns for stored data.
        dtypes (Tuple[type, ...]): Data types for each column.
    """

    column_names: Tuple[str, ...]
    dtypes: Tuple[type, ...]

    def __init__(self) -> None:
        """
        Initialize the ResultHandler with empty column names and dtypes.
        """
        self.column_names: Tuple[str, ...] = ()
        self.dtypes: Tuple[type, ...] = ()

    def __enter__(self) -> "ResultHandler":
        """
        Enter the context manager.
        """
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context manager.
        """
        pass

    def set_columns(self, names: Tuple[str, ...], dtypes: Tuple[type, ...] = None) -> None:
        """
        Set the column names and data types for the handler.

        Args:
            names (Tuple[str, ...]): Column names.
            dtypes (Tuple[type, ...], optional): Data types for each column.
        """
        assert len(names) == len(dtypes), "Length of names and dtypes must be equal"
        self.column_names = names
        self.dtypes = dtypes

    def put(self, data: Tuple[Any, ...]) -> None:
        """
        Put a data tuple into the handler.
        """
        raise NotImplementedError

    def get(self) -> Union[Tuple[Any, ...], None]:
        """
        Get a single data tuple from the handler.
        """
        raise NotImplementedError

    def get_all(self) -> List[Tuple[Any, ...]]:
        """
        Get all data tuples from the handler.
        """
        raise NotImplementedError


class MPQueueResultHandler(ResultHandler):
    """
    Multiprocessing Queue ResultHandler.
    Wrapper for a multiprocessing.Queue.

    Example:
        handler = MPQueueResultHandler()
        handler.put((1, 2, 3))
        print(handler.get())
    """

    def __init__(self) -> None:
        """
        Initialize the MPQueueResultHandler with a multiprocessing.Queue.
        """
        super().__init__()
        self.queue = Queue()

    def put(self, data: Tuple[Any, ...]) -> None:
        """
        Put a data tuple into the queue.
        """
        return self.queue.put(data)

    def get(self) -> Union[Tuple[Any, ...], None]:
        """
        Get a single data tuple from the queue.
        """
        return self.queue.get()

    def get_all(self) -> List[Tuple[Any, ...]]:
        """
        Get all data tuples from the queue until None is received.
        """
        data = []
        for el in iter(self.queue.get, None):
            data.append(el)
        return data


class FileCacheResultHandler(ResultHandler):
    """
    File Cache ResultHandler.
    Writes data to a file and reads it from disk.

    Args:
        file_path (str): Path to the file.
        force (bool): If True, overwrite the file if it already exists.

    Example:
        handler = FileCacheResultHandler('results.csv', force=True)
        handler.set_columns(('a', 'b', 'c'), (int, int, float))
        handler.put((1, 2, 3.5))
        print(handler.get_all())
    """

    file_path: str

    def __init__(self, file_path: str, force: bool = False):
        """
        Initialize the FileCacheResultHandler, set up file path and handle overwriting.
        """
        super().__init__()
        self.file_path = file_path
        if os.path.dirname(file_path) != "":
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if force and os.path.isfile(file_path):
            os.remove(file_path)
        # if os.path.isfile(file_path) and os.path.getsize(file_path) > 0 and not force:
        #     warnings.warn(f"Warning: File {file_path} already exists and is not empty")

        self.file_obj = None

    def __enter__(self) -> "FileCacheResultHandler":
        """
        Enter the context manager and open the file for appending.
        """
        self.file_obj = open(self.file_path, "a", encoding="utf-8")
        return self

    def __exit__(self, *args) -> None:
        """
        Exit the context manager and close the file.
        """
        self.file_obj.close()

    def set_columns(self, header: Tuple[str, ...], dtypes: Tuple[type, ...] = None) -> None:
        """
        Set the column names and data types, and write the header to the file if needed.

        Args:
            header (Tuple[str, ...]): Column names.
            dtypes (Tuple[type, ...], optional): Data types for each column.
        """
        super().set_columns(header, dtypes)
        if os.path.exists(self.file_path) and os.path.getsize(self.file_path) > 0:
            return
        with open(self.file_path, "w", encoding="utf-8") as file_obj:
            file_obj.write(",".join(header) + "\n")

    def put(self, data: Tuple[Any, ...]) -> None:
        """
        Write a data tuple to the file.
        """
        if not isinstance(data, tuple):
            return
        with open(self.file_path, "a", encoding="utf-8") as file_obj:
            file_obj.write(",".join([str(el) for el in data]) + "\n")

    def _read_line_tuple(self, line: str) -> Tuple[Any, ...]:
        """
        Convert a line from the file to a tuple using the specified data types.
        """
        return tuple(t(el) for el, t in zip_longest(line.strip().split(","), self.dtypes, fillvalue=str))

    def get(self) -> Union[Tuple[Any, ...], None]:
        """
        Get the last data tuple from the file.
        """
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            if not lines:
                return None
            return self._read_line_tuple(lines[-1])

    def get_all(self) -> List[Tuple[Any, ...]]:
        """
        Get all data tuples from the file.
        """
        with open(self.file_path, "r", encoding="utf-8") as file_obj:
            lines = file_obj.readlines()[1 if len(self.column_names) > 0 else 0 :]
            data = [self._read_line_tuple(line) for line in lines]
        return data


def start_seprate_process(target: Callable, other_args: List[Any]) -> Any:
    """
    Start a separate process for evaluation, passing a queue as the first argument to the target function.

    Args:
        target (Callable): Target function. Should take a result_queue as the first argument and put the result in the queue.
        other_args (List): List of arguments for the target function after the result_queue.
    Returns:
        Any: The result from the queue after the process finishes.

    Example:
        def worker(queue, x):
            queue.put(x * 2)
        result = start_seprate_process(worker, [5])
        print(result)  # Output: 10
    """
    multiprocessing.set_start_method("spawn", force=True)
    manager = Manager()
    queue = manager.Queue()
    process = Process(
        target=target,
        args=[queue] + other_args,
    )
    process.start()
    process.join()
    return queue.get()
