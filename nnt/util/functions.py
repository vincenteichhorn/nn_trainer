from datetime import datetime
from itertools import islice
import json
import os
from typing import Any, Dict, Tuple, Iterable
import plotly.express as px


def parse_args_string(args_string: str) -> Dict[str, Any]:
    """
    Parse a string of arguments into a dictionary.
    Args:
        args_string (str): The string of arguments.
    Returns:
        Dict[str, Any]: The dictionary of arguments.
    """
    args = {}
    if args_string is not None:
        for arg in args_string.split(","):
            key, value = arg.split("=")
            if '"' in value or "'" in value:
                value = value.replace('"', "").replace("'", "")
            elif "." in value:
                value = float(value)
            else:
                value = int(value)
            args[key] = value
    return args


def parse_class_initialisation_str(class_initialisation_str: str) -> Tuple[str, Dict[str, Any]]:
    """
    Parse a string of class initialisation arguments into a dictionary.
    Args:
        class_initialisation_str (str): The string of class initialisation arguments. Like Class(arg1=val1, arg2=val2).
    Returns:
        Dict[str, Any]: The dictionary of class initialisation arguments.
    """
    class_name = class_initialisation_str.split("(")[0]
    args_string = class_initialisation_str.split("(")[1].replace(")", "")
    return class_name, parse_args_string(args_string)


def save_json(data: dict, path: str) -> None:
    """
    Save a dictionary to a JSON file.
    Args:
        data (dict): The dictionary to save.
        path (str): The path to save the dictionary to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json(path: str) -> dict:
    """
    Load a dictionary from a JSON file.
    Args:
        path (str): The path to the JSON file.
    Returns:
        dict: The loaded dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)


def get_current_time() -> str:
    """
    Get the current time as a string.
    Returns:
        str: The current time as a string.
    """
    return datetime.now().strftime("%Y-%m-%d %H-%M-%S.%f")


def iter_batchwise(iterable, n: int = 1) -> Iterable[Tuple[Any, ...]]:
    """
    Batch an iterable into chunks of size n.
    Args:
        iterable: The iterable to batch.
        n: The size of the batches.
    """
    it = iter(iterable)
    while True:
        batch = tuple(islice(it, n))
        if not batch:
            return
        yield batch


def get_plotly_color_scale(n: int, base_colors=px.colors.qualitative.Plotly) -> list:
    """
    Get a color scale for Plotly.
    Args:
        n (int): The number of colors to generate.
        base_colors: The base color scale to use.
    """
    colors = (base_colors * ((n // len(base_colors)) + 1))[:n]
    return colors


def flatten_dict(d: dict, parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """
    Flatten a nested dictionary into a single-level dictionary with compound keys.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str, optional): The base key to prepend to each key. Defaults to "".
        sep (str, optional): Separator to use between parent and child keys. Defaults to "_".
    Returns:
        dict: A flattened dictionary with compound keys.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
