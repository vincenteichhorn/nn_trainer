from tqdm import tqdm
from threading import RLock


class Monitor:
    _instance = None
    _lock = RLock()  # Lock for thread-safe singleton access

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.pbars = []
        self.lock = RLock()
        tqdm.set_lock(self.lock)
        self._initialized = True

    def tqdm(self, iterable=None, total=None, leave=None, desc="", **kwargs):
        """
        Automatically assigns position based on the number of existing bars.
        """
        self.clear_closed_bars()
        position = len(self.pbars)
        pbar = tqdm(iterable=iterable, total=total, desc=desc, position=position, leave=leave, **kwargs)
        self.pbars.append(pbar)
        return pbar

    def print(self, message):
        """
        Print a message to the console without breaking the progress bars.
        """
        with self.lock:
            tqdm.write(message)

    def clear_closed_bars(self):
        """
        Remove closed progress bars from the list.
        This helps to keep the list of progress bars clean.
        """
        self.pbars = [pbar for pbar in self.pbars if not pbar.disable]

    def close_all(self):
        """
        Close all progress bars.
        """
        for p in self.pbars:
            p.close()
        self.pbars.clear()
