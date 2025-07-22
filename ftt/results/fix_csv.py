import io
import os
import re

import pandas as pd
from nnt.util.monitor import Monitor


def mask_brackets_in_csv(path: str) -> str:
    """
    Reads a CSV file and wraps each '[' with '"[', and each ']' with ']"'
    to protect list-like content from being split on commas.
    Returns the modified CSV data as a single string.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace all '[' and ']' at once
    # Only replace [ if not already preceded by a quote, and ] if not already followed by a quote
    content = re.sub(r'(?<!")\[', '"[', content)
    content = re.sub(r'\](?!")', ']"', content)

    return content


if __name__ == "__main__":

    base_dir = "/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/out"

    # walk through all subdirectories searching for csv files
    csv_files = []
    for root, dirs, files in Monitor().tqdm(os.walk(base_dir), desc="Searching for CSV files"):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    print(f"Found {len(csv_files)} csv files.")

    for csv in Monitor().tqdm(csv_files):
        Monitor().print(f"Processing {csv}")
        df = pd.read_csv(io.StringIO(mask_brackets_in_csv(csv)), quotechar='"')
        new_file_name = csv.replace(".csv", "_fixed.csv")
        df.to_csv(new_file_name, index=False)
