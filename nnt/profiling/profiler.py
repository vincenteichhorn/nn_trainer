from contextlib import contextmanager
from typing import List, Tuple
from datetime import datetime


class Profiler:
    """
    Abstract base class for profiling. Tracks named record steps and provides context management for profiling code sections.

    Attributes:
        record_steps (List[Tuple[datetime, str]]): List of tuples containing timestamps and step names.
    """

    record_steps: List[Tuple[datetime, str]]

    def __init__(self) -> None:
        """
        Initialize the Profiler and record the initial step.
        """
        self.record_steps = []
        self.record_step("__init__")

    def record_step(self, name: str) -> None:
        """
        Save a record step with a given name and current timestamp.

        Args:
            name (str): The name of the step to record.
        """
        """
        saves a record step with a name

        Args:
            name (str): the name of the step
        """
        self.record_steps.append((datetime.now(), name))

    @contextmanager
    def record_context(self, name: str):
        """
        Context manager for recording profiling steps. Records a step with the given name at entry, and a step named "__other__" at exit.

        Args:
            name (str): The name of the context step.
        """
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
