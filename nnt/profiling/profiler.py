from contextlib import contextmanager
from typing import List, Tuple
from datetime import datetime


class Profiler:
    """
    Abstract Class for Profilers
    """

    def __init__(self):
        self.record_steps: List[Tuple[datetime, str]] = []
        self.record_step("__init__")

    def record_step(self, name: str) -> None:
        """
        saves a record step with a name

        Args:
            name (str): the name of the step
        """
        self.record_steps.append((datetime.now(), name))

    @contextmanager
    def record_context(self, name: str):
        """
        record step as a context manager.
        Starts a record_step with the given name,
        after execution of the context it records a step with the name "__other__"

        Args:
            name (str): the name of the context
        """
        try:
            self.record_step(name)
            yield None
        finally:
            self.record_step("__other__")
