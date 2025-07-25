import argparse
import ast
import io
import os
import random
import time
from typing import Dict
import warnings
import pandas as pd

from nnt.profiling.nvidia_profiler import NvidiaProfiler
from nnt.util.monitor import Monitor
import re
import multiprocessing


correction = {
    "mapping": {"gx29@NVIDIA-A40-45G": 0, "gx02@NVIDIA-A100-SXM4-80GB-80G": 0},
    "correction": {
        0: {
            "time": lambda x: x,
            "energy": lambda x: x,
        },
    },
}


def try_float(inp):
    """
    Attempts to convert the input to a float.
    If it fails, returns the input unchanged.
    """
    try:
        return float(inp)
    except ValueError:
        return None


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


def get_csv_data(path: str) -> pd.DataFrame:

    file_name = os.path.basename(path)
    file_to_open = path
    if "validation" in file_name:
        if os.path.exists(os.path.join(os.path.dirname(path), "re_validation_log.csv")):
            file_to_open = os.path.join(os.path.dirname(path), "re_validation_log.csv")

    if os.path.exists(os.path.join(os.path.dirname(path), file_name.replace(".csv", "_fixed.csv"))):
        file_to_open = os.path.join(os.path.dirname(path), file_name.replace(".csv", "_fixed.csv"))

    # if not "_fixed" in file_to_open:
    #     content = mask_brackets_in_csv(file_to_open)
    #     df = pd.read_csv(io.StringIO(content), quotechar='"')
    # else:
    df = pd.read_csv(file_to_open)
    return df, file_to_open


def get_run_result(run_folder: str) -> Dict[str, float]:

    results = {}
    donefile_path = os.path.join(run_folder, "donefile")
    if not os.path.exists(donefile_path):
        return results

    validation_df, _ = get_csv_data(os.path.join(run_folder, "validation_log.csv"))
    for col in validation_df.columns:
        if col in ["timestamp", "learning_rate"]:
            continue
        min_val = validation_df[col].min()
        max_val = validation_df[col].max()
        results[f"{col}_max"] = try_float(max_val)
        results[f"{col}_min"] = try_float(min_val)
    hardware_signature = "unknown"
    if os.path.exists(os.path.join(run_folder, "hardware_signature")):
        with open(os.path.join(run_folder, "hardware_signature"), "r") as f:
            hardware_signature = f.read().strip()
        assert hardware_signature != "unknown"
    energy_correction = correction["correction"][correction["mapping"].get(hardware_signature, 0)]["energy"]
    time_correction = correction["correction"][correction["mapping"].get(hardware_signature, 0)]["time"]
    _, energy_log_path = get_csv_data(os.path.join(run_folder, "energy_log.csv"))
    energy_prof = NvidiaProfiler.from_cache(energy_log_path)
    results["train_energy"] = energy_correction(float(energy_prof.get_total_energy(record_steps=["step_begin"])))
    results["total_energy"] = energy_correction(float(energy_prof.get_total_energy()))
    results["train_time"] = time_correction(float(energy_prof.get_total_time(record_steps=["step_begin"])))
    results["total_time"] = time_correction(float(energy_prof.get_total_time()))

    flops_budget_log_path = os.path.join(run_folder, "flops_budget_log.csv")
    if os.path.exists(flops_budget_log_path):
        flops_budget_df, _ = get_csv_data(flops_budget_log_path)
        results["total_flops"] = float(flops_budget_df["cumulative_flops"].max())
    return results


if __name__ == "__main__":

    base_folder = "/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/"

    parser = argparse.ArgumentParser(description="Aggregate results from runs.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Experiment directory containing subfolders.")
    parser.add_argument(
        "--parse_rules",
        type=str,
        default="{}",
        help='Dictionary of parse rules as a string, e.g. \'{"nlayer": "lambda x: int(x.split("-")[-1])"}\'',
    )
    args = parser.parse_args()

    parse_rules_dict = ast.literal_eval(args.parse_rules)
    parse_rules = {}
    for k, v in parse_rules_dict.items():
        parse_rules[k] = eval(v)

    exp_folder = os.path.join(base_folder, args.exp_dir)
    print(f"Processing experiment folder: {exp_folder}")
    if not os.path.exists(exp_folder):
        print(f"Experiment folder {exp_folder} does not exist. Exiting.")
        exit(1)
    dataset_folders = [f.path for f in os.scandir(exp_folder) if f.is_dir()]
    warnings.simplefilter(action="ignore", category=FutureWarning)

    run_folders = []

    data = []

    for dataset_run in Monitor().tqdm(dataset_folders, desc="Searching for results"):
        Monitor().print(f"Processing dataset: {dataset_run}")
        ds_run_folders = [
            f.path for f in os.scandir(dataset_run) if f.is_dir() and os.path.exists(os.path.join(f.path, "donefile"))
        ]
        run_folders.extend(ds_run_folders)

    # Define global fields for parse_rules
    global_fields = list(parse_rules.keys())

    def process_run(args):
        run, dataset_run = args
        run_result = get_run_result(run)
        if len(run_result) == 0:
            return None
        parsed_run = {k: parse_rules[k](os.path.basename(run)) for k in global_fields}
        run_result = {**parsed_run, **run_result}
        run_result["dataset"] = os.path.basename(dataset_run)
        return run_result

    max_processes = min(24, multiprocessing.cpu_count())

    run_args = [(run, os.path.dirname(run)) for run in run_folders]
    random.shuffle(run_args)
    with multiprocessing.Pool(processes=max_processes) as pool:
        results = list(
            Monitor().tqdm(pool.imap_unordered(process_run, run_args), total=len(run_args), desc="Processing runs")
        )
    data.extend([r for r in results if r is not None])

    df = pd.DataFrame(data)
    if df.empty:
        print("No results found. Exiting.")
        exit(0)

    group_on = list(el for el in parse_rules.keys() if el != "repid") + ["dataset"]
    summary = df.groupby(group_on).count()["repid"].rename("count")
    print(summary[summary < 5])

    df = (
        df.groupby(group_on)
        .agg({col: ["mean", "sem"] for col in df.columns if col not in group_on and col != "repid"})
        .reset_index()
    )
    df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    # remove "_" at the end of column names
    df.columns = [col[:-1] if col.endswith("_") else col for col in df.columns]

    df.to_csv(os.path.join(exp_folder, "results.csv"), index=False)
