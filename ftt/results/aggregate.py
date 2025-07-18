import argparse
import ast
import io
import os
from typing import Dict
import warnings
import pandas as pd

from nnt.profiling.nvidia_profiler import NvidiaProfiler
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
    content = content.replace("[", '"[').replace("]", ']"')

    return content


def try_float(inp):
    """
    Attempts to convert the input to a float.
    If it fails, returns the input unchanged.
    """
    try:
        return float(inp)
    except ValueError:
        return None


def get_run_result(run_folder: str) -> Dict[str, float]:

    results = {}
    donefile_path = os.path.join(run_folder, "donefile")
    if not os.path.exists(donefile_path):
        return results

    validation_log_path = os.path.join(run_folder, "validation_log.csv")
    validation_df = pd.read_csv(io.StringIO(mask_brackets_in_csv(validation_log_path)), quotechar='"')
    for col in validation_df.columns:
        if col in ["timestamp", "learning_rate"]:
            continue
        min_val = validation_df[col].min()
        max_val = validation_df[col].max()
        results[f"{col}_max"] = try_float(max_val)
        results[f"{col}_min"] = try_float(min_val)

    energy_log_path = os.path.join(run_folder, "energy_log.csv")
    energy_prof = NvidiaProfiler.from_cache(energy_log_path)
    results["train_energy"] = float(energy_prof.get_total_energy(record_steps=["step_begin"]))
    results["total_energy"] = float(energy_prof.get_total_energy())
    results["train_time"] = float(energy_prof.get_total_time(record_steps=["step_begin"]))
    results["total_time"] = float(energy_prof.get_total_time())

    flops_budget_log_path = os.path.join(run_folder, "flops_budget_log.csv")
    if os.path.exists(flops_budget_log_path):
        flops_budget_df = pd.read_csv(flops_budget_log_path)
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

    df = pd.DataFrame()

    for dataset_run in Monitor().tqdm(dataset_folders, desc="Processing Results"):
        Monitor().print(f"Processing dataset: {dataset_run}")
        run_folders = [
            f.path for f in os.scandir(dataset_run) if f.is_dir() and os.path.exists(os.path.join(f.path, "donefile"))
        ]
        for run in Monitor().tqdm(run_folders, desc=f"Processing runs in {os.path.basename(dataset_run)}"):
            # print(f"Processing run: {run}")
            run_result = get_run_result(run)
            if not run_result:
                continue
            parsed_run = {k: v(os.path.basename(run)) for k, v in parse_rules.items()}
            run_result = {**parsed_run, **run_result}
            run_result["dataset"] = os.path.basename(dataset_run)
            warnings.simplefilter(action="ignore", category=FutureWarning)
            df = pd.concat([df, pd.DataFrame([run_result])], ignore_index=True)

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
