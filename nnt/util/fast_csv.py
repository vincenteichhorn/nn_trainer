import os
import warnings


class FastCSV:

    def __init__(self, file_path: str, force: bool = False):
        self.file_path = file_path
        self.columns = []

        if os.path.exists(file_path) and force:
            os.remove(file_path)
        elif os.path.exists(file_path):
            warnings.warn(
                f"File {file_path} already exists. Use force=True to overwrite. Will append to the existing file regardless of content.",
                UserWarning,
            )

    def set_columns(self, columns: list):
        self.columns = columns
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                f.write(",".join(columns) + "\n")

    def append(self, row: dict):
        assert all(key in row for key in self.columns), "Row must contain all columns"
        with open(self.file_path, "a") as f:
            f.write(",".join(str(v) for v in row.values()) + "\n")
