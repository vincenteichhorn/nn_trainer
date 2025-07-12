import os
import warnings


class FastCSV:
    """
    Utility class for fast CSV file writing and appending.
    Handles column setup and row appending, with options to overwrite or append to existing files.

    Attributes:
        file_path (str): Path to the CSV file.
        columns (list): List of column names for the CSV file.
    """

    file_path: str
    columns: list

    def __init__(self, file_path: str, force: bool = False):
        """
        Initialize the FastCSV object, set file path, and handle file overwrite or append.

        Args:
            file_path (str): Path to the CSV file.
            force (bool, optional): If True, overwrite the file if it exists. Otherwise, append.
        """
        self.file_path = file_path
        self.columns = []

        if os.path.exists(file_path) and force:
            os.remove(file_path)
        elif os.path.exists(file_path):
            warnings.warn(
                f"File {file_path} already exists. Use force=True to overwrite. Will append to the existing file regardless of content.",
                UserWarning,
            )

    def set_columns(self, columns: list) -> None:
        """
        Set the column names for the CSV file and write the header if the file does not exist.

        Args:
            columns (list): List of column names.
        """
        self.columns = columns
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                f.write(",".join(columns) + "\n")

    def append(self, row: dict) -> None:
        """
        Append a row to the CSV file. The row must contain all columns.

        Args:
            row (dict): Dictionary mapping column names to values.
        Raises:
            AssertionError: If the row does not contain all columns.
        """
        assert all(key in row for key in self.columns), "Row must contain all columns"
        with open(self.file_path, "a") as f:
            f.write(",".join(str(v) for v in row.values()) + "\n")
