


import os
import streamlit as st
from dotenv import load_dotenv
import pandas as pd

BASE_DIR =  "/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/out/"

task_groups = {
    "NLU": [
        "glue_cola",
        "glue_sst2",
        "glue_mrpc",
        "glue_qqp",
        "glue_mnli_matched",
        "glue_mnli_mismatched",
        "glue_qnli",
        "glue_rte",
    ],
    "reasoning": [
        "arc_easy",
        "arc_challenge",
        "piqa",
        "boolq",
        "hellaswag",
    ],
    "NLG": [
        "rocstories_title_answer_generation",
        "gigaword_summarization",
        "alpaca_mmlu",
    ],
}

exp_names = [
    "glue_cola",
    "glue_sst2",
    "glue_mrpc",
    "glue_qqp",
    "glue_mnli_matched",
    "glue_mnli_mismatched",
    "glue_qnli",
    "glue_rte",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "boolq",
    "hellaswag",
    "rocstories_title_answer_generation",
    "gigaword_summarization",
    "alpaca_mmlu",
]

pretty_metrics = {
    "glue_cola": "MCC",
    "glue_sst2": "Accuracy",
    "glue_mrpc": "F1",
    "glue_qqp": "F1",
    "glue_mnli_matched": "Accuracy",
    "glue_mnli_mismatched": "Accuracy",
    "glue_qnli": "Accuracy",
    "glue_rte": "Accuracy",
    "arc_easy": "Accuracy",
    "arc_challenge": "Accuracy",
    "piqa": "Accuracy",
    "boolq": "Accuracy",
    "hellaswag": "Accuracy",
    "task219_rocstories_title_answer_generation": "ROUGE-L",
    "allenai_task288_gigaword_summarization": "ROUGE-L",
    "alpaca_mmlu": "ROUGE-L",
}

pretty_names = {
    "glue_cola": "CoLA",
    "glue_sst2": "SST2",
    "glue_mrpc": "MRPC",
    "glue_qqp": "QQP",
    "glue_mnli_matched": "MNLIm",
    "glue_mnli_mismatched": "MNLImm",
    "glue_qnli": "QNLI",
    "glue_rte": "RTE",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Chal.",
    "piqa": "PIQA",
    "boolq": "BoolQ",
    "hellaswag": "HellaSwag",
    "task219_rocstories_title_answer_generation": "ROCStories",
    "allenai_task288_gigaword_summarization": "GigaWord",
    "alpaca_mmlu": "Alpaca",
}

performance_metrics = {
    "glue_cola": "OneHotClassificationMetrics_mcc_max_mean",
    "glue_sst2": "OneHotClassificationMetrics_accuracy_max_mean",
    "glue_mrpc": "OneHotClassificationMetrics_f1_score_max_mean",
    "glue_qqp": "OneHotClassificationMetrics_f1_score_max_mean",
    "glue_mnli_matched": "OneHotClassificationMetrics_accuracy_max_mean",
    "glue_mnli_mismatched": "OneHotClassificationMetrics_accuracy_max_mean",
    "glue_qnli": "OneHotClassificationMetrics_accuracy_max_mean",
    "glue_rte": "OneHotClassificationMetrics_accuracy_max_mean",
    "arc_easy": "OneHotClassificationMetrics_accuracy_max_mean",
    "arc_challenge": "OneHotClassificationMetrics_accuracy_max_mean",
    "piqa": "OneHotClassificationMetrics_accuracy_max_mean",
    "boolq": "OneHotClassificationMetrics_accuracy_max_mean",
    "hellaswag": "OneHotClassificationMetrics_accuracy_max_mean",
    "task219_rocstories_title_answer_generation": "RougeScore_rougeL_max_mean",
    "allenai_task288_gigaword_summarization": "RougeScore_rougeL_max_mean",
    "alpaca_mmlu": "RougeScore_rougeL_max_mean",
}

def get_base_table(df, add_cols=[]):

    df["performance"] = df.apply(
        lambda row: row[performance_metrics[row["dataset"]]] if row["dataset"] in performance_metrics else None, axis=1
    ) * 100
    df["performance_sem"] = df.apply(
        lambda row: row[performance_metrics[row["dataset"]].replace("_mean", "_sem")] if row["dataset"] in performance_metrics else None, axis=1
    ) * 100
    cols_filter = ["train_energy_mean", "train_energy_sem", "total_energy_mean", "total_energy_sem", "train_time_mean", "train_time_sem", "total_time_mean", "total_time_sem", "dataset", "performance", "performance_sem", "total_flops_mean"]
    cols_filter += add_cols
    df = df[cols_filter]
    return df

def get_base_model_performance(df):
    """
    Get the base model performance from the dataframe.
    This is used to compare the performance of different models against the base model.
    """
    df["performance"] = df.apply(
        lambda row: row[performance_metrics[row["dataset"]].replace("max", "min")] if row["dataset"] in performance_metrics else None, axis=1
    ) * 100
    return df[["dataset", "performance"]].groupby("dataset").mean().reset_index()

def static_selection_table():

    df = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    base = get_base_model_performance(df)
    df = get_base_table(df, add_cols=["nlayer"])
    df["c"] = 17 - df["nlayer"]
    df = df[df["c"].isin([1, 4, 8, 12, 16])]
    base["c"] = "base"
    df = pd.concat([df, base], ignore_index=True)
    df = df.pivot(index="c", columns="dataset", values=["performance", "performance_sem"])
    
    values = df["performance"]
    values_sem = df["performance_sem"]
    values = values.applymap(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    values_sem = values_sem.applymap(lambda x: f" ({x:.2f})" if pd.notnull(x) else "")
    df = values + values_sem
    # add pretty names
    df.columns = [f"{pretty_names[col]}" for col in df.columns]
    st.write(df)

def static_optimal_table():

    df = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    df = get_base_table(df, add_cols=["nlayer"])
    df["c"] = 17 - df["nlayer"]

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3 
    
    acceptable_performance_degradation = 3
    for dataset in df["dataset"].unique():
        for metric in ["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]:
            base_value = df.loc[(df["dataset"] == dataset) & (df["c"] == 1), metric].values[0]
            df.loc[df["dataset"] == dataset, f"{metric}_reduction"] = base_value - df.loc[df["dataset"] == dataset, metric]
            if metric == "performance":
                df.loc[df["dataset"] == dataset, "acceptable"] = df.loc[df["dataset"] == dataset, "performance"] >= (base_value - acceptable_performance_degradation)


    for dataset in df["dataset"].unique():
        # get maximal c with acceptable performance
        max_c = df.loc[(df["dataset"] == dataset) & (df["acceptable"]), "c"].max()
        # drop all with x != max_c
        df = df[~((df["dataset"] == dataset) & (df["c"] != max_c))]
    

    # pivot the table, rows should be the metrics: train_energy_mean, train_time_mean, total_flops_mean, performance and reduction: train_energy_reduction, train_time_reduction, total_flops_reduction and c
    # columns should be the datasets
    df = df.melt(
        id_vars=["dataset"],
        value_vars=["train_energy_mean", "train_time_mean", "total_flops_mean", "performance", 
                    "train_energy_mean_reduction", "train_time_mean_reduction", "total_flops_mean_reduction", "c"],
        var_name="metric",
        value_name="value"
    )
    df = df.pivot_table(
        index=["metric"],
        columns="dataset",
        values="value",
        aggfunc='first'
    )
    st.write(df)


if __name__ == "__main__":

    # streamlit run ./ftt/results/plotting/results_tables.py --server.fileWatcherType=poll
    st.set_page_config(
        page_title="Result Tables",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    static_selection_table()
    st.write("---")
    static_optimal_table()
