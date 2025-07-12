from tqdm import tqdm
from threading import RLock


class Monitor:
    """
    Singleton class for managing multiple tqdm progress bars and thread-safe console output.

    Attributes:
        pbars (list): List of active tqdm progress bars.
        lock (RLock): Thread lock for safe progress bar and print operations.
    """

    _instance: "Monitor" = None
    _lock: RLock = RLock()  # Lock for thread-safe singleton access

    def __new__(cls, *args, **kwargs) -> "Monitor":
        """
        Create or return the singleton instance of Monitor.
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the Monitor instance, setting up progress bar list and thread lock.
        """
        if self._initialized:
            return
        self.pbars = []
        self.lock = RLock()
        tqdm.set_lock(self.lock)
        self._initialized = True

    def tqdm(
        self,
        iterable=None,
        total: int = None,
        leave: bool = None,
        desc: str = "",
        **kwargs,
    ) -> tqdm:
        """
        Create and track a new tqdm progress bar, automatically assigning its position.

        Args:
            iterable: Iterable to wrap with tqdm.
            total (int, optional): Total number of iterations.
            leave (bool, optional): Whether to leave the bar after completion.
            desc (str, optional): Description for the progress bar.
            **kwargs: Additional tqdm arguments.
        Returns:
            tqdm: The created tqdm progress bar.
        """
        self.clear_closed_bars()
        position = len(self.pbars)
        pbar = tqdm(iterable=iterable, total=total, desc=desc, position=position, leave=leave, **kwargs)
        self.pbars.append(pbar)
        return pbar

    def print(self, message: str) -> None:
        """
        Print a message to the console without breaking the progress bars.

        Args:
            message (str): Message to print.
        """
        with self.lock:
            tqdm.write(message)

    def clear_closed_bars(self) -> None:
        """
        Remove closed progress bars from the list to keep it clean.
        """
        self.pbars = [pbar for pbar in self.pbars if not pbar.disable]

    def close_all(self) -> None:
        """
        Close all active progress bars and clear the list.
        """
        for p in self.pbars:
            p.close()
        self.pbars.clear()
