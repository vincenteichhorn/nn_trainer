import os
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "ftt/out/approach_out/"

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
        "alpaca_mmlu",
        "allenai_task219_rocstories_title_answer_generation",
        "allenai_task288_gigaword_summarization",
    ],
}

colors = {
    "blueish": "#2574a9",
    "greenish": "#74a925",
    "redish": "#a92574",
    "cyanish": "#25a974",
}

order = list(task_groups["NLU"]) + list(task_groups["reasoning"]) + list(task_groups["NLG"])


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
    "alpaca_mmlu",
    "allenai_task219_rocstories_title_answer_generation",
    "allenai_task288_gigaword_summarization",
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
    "allenai_task219_rocstories_title_answer_generation": "ROUGE-L",
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
    "allenai_task219_rocstories_title_answer_generation": "ROCStories",
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
    "allenai_task219_rocstories_title_answer_generation": "RougeScore_rougeL_max_mean",
    "allenai_task288_gigaword_summarization": "RougeScore_rougeL_max_mean",
    "alpaca_mmlu": "RougeScore_rougeL_max_mean",
}


def get_base_table(df, add_cols=[]):

    df["performance"] = (
        df.apply(
            lambda row: row[performance_metrics[row["dataset"]]] if row["dataset"] in performance_metrics else None, axis=1
        )
        * 100
    )
    df["performance_sem"] = (
        df.apply(
            lambda row: (
                row[performance_metrics[row["dataset"]].replace("_mean", "_sem")]
                if row["dataset"] in performance_metrics
                else None
            ),
            axis=1,
        )
        * 100
    )
    cols_filter = [
        "train_energy_mean",
        "train_energy_sem",
        "total_energy_mean",
        "total_energy_sem",
        "train_time_mean",
        "train_time_sem",
        "total_time_mean",
        "total_time_sem",
        "dataset",
        "performance",
        "performance_sem",
        "total_flops_mean",
    ]
    cols_filter += add_cols
    df = df[cols_filter]
    return df


def get_base_model_performance(df):
    """
    Get the base model performance from the dataframe.
    This is used to compare the performance of different models against the base model.
    """
    df["performance"] = (
        df.apply(
            lambda row: (
                row[performance_metrics[row["dataset"]].replace("max", "min")]
                if row["dataset"] in performance_metrics
                else None
            ),
            axis=1,
        )
        * 100
    )
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
    # sort columbs by order
    df = df.reindex(columns=order, fill_value="N/A")
    df = df.rename(columns={col: pretty_names[col] for col in df.columns})

    st.write("### Static Top Layers (Selection)")
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

    acceptable_performance_degradation = 3  # percent
    for dataset in df["dataset"].unique():
        for metric in ["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]:
            base_value = df.loc[(df["dataset"] == dataset) & (df["c"] == 1), metric].values[0]
            # Compute relative reduction in percent
            df.loc[df["dataset"] == dataset, f"{metric}_reduction (%)"] = (
                (base_value - df.loc[df["dataset"] == dataset, metric]) / base_value * 100
            )
            if metric == "performance":
                df.loc[df["dataset"] == dataset, "acceptable"] = (
                    df.loc[df["dataset"] == dataset, "performance_reduction (%)"] <= acceptable_performance_degradation
                )

    for dataset in df["dataset"].unique():
        # get maximal c with acceptable performance
        max_c = df.loc[(df["dataset"] == dataset) & (df["acceptable"]), "c"].max()
        # drop all with x != max_c
        df = df[~((df["dataset"] == dataset) & (df["c"] != max_c))]

    # pivot the table, rows should be the metrics: train_energy_mean, train_time_mean, total_flops_mean, performance and reduction: train_energy_reduction, train_time_reduction, total_flops_reduction and c
    # columns should be the datasets
    df = df.melt(
        id_vars=["dataset"],
        value_vars=[
            "train_energy_mean",
            "train_time_mean",
            "total_flops_mean",
            "train_energy_sem",
            "train_time_sem",
            "performance",
            "performance_sem",
            "performance_reduction (%)",
            "train_energy_mean_reduction (%)",
            "train_time_mean_reduction (%)",
            "total_flops_mean_reduction (%)",
            "c",
        ],
        var_name="metric",
        value_name="value",
    )
    df = df.pivot_table(index=["metric"], columns="dataset", values="value", aggfunc="first")
    df = df.reindex(columns=order, fill_value="N/A")
    df = df.rename(columns={col: pretty_names[col] for col in df.columns})
    for i, row in df.iterrows():
        if "_sem" in row.name:
            sem_vals = row.values
            value_row_name = (
                row.name.replace("_sem", "_mean") if not "performance" in row.name else row.name.replace("_sem", "")
            )
            if value_row_name in df.index:
                value_vals = df.loc[df.index == value_row_name].values[0]
                df.loc[df.index == value_row_name] = [
                    f"{float(value):.2f} ({sem:.2f})" if pd.notnull(value) else "N/A"
                    for value, sem in zip(value_vals, sem_vals)
                ]
        else:
            df.loc[i] = [f"{value:.2f}" if pd.notnull(value) else "N/A" for value in row.values]
    df = df[~df.index.str.endswith("_sem")]
    st.write('### Static Top Layers ("Optimal" static top layers within 3% performance degradation)')
    st.write(df)


def stochastic_table(rho=0.75):

    df = pd.read_csv(os.path.join(BASE_DIR, "stochastic/results.csv"))
    base = get_base_model_performance(df)

    df = get_base_table(df, add_cols=["savings"])
    base["savings"] = "base"
    df = pd.concat([df, base], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["c"] = 17 - fl["nlayer"]
    fl = fl[fl["c"] == 1]
    fl["savings"] = "full LoRA"
    df = pd.concat([df, fl], ignore_index=True)

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3

    for dataset in df["dataset"].unique():
        for metric in ["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]:
            base_value = df.loc[(df["dataset"] == dataset) & (df["savings"] == "full LoRA"), metric].values[0]
            df.loc[df["dataset"] == dataset, f"{metric}_reduction (%)"] = (
                (base_value - df.loc[df["dataset"] == dataset, metric]) / base_value * 100
            )

    df = df[df["savings"].isin([rho])]

    df = df.melt(
        id_vars=["dataset"],
        value_vars=[
            "train_energy_mean",
            "train_time_mean",
            "total_flops_mean",
            "train_energy_sem",
            "train_time_sem",
            "performance",
            "performance_sem",
            "performance_reduction (%)",
            "train_energy_mean_reduction (%)",
            "train_time_mean_reduction (%)",
            "total_flops_mean_reduction (%)",
            "savings",
        ],
        var_name="metric",
        value_name="value",
    )
    df = df.pivot_table(index=["metric"], columns="dataset", values="value", aggfunc="first")
    df = df.reindex(order, axis=1)
    df = df.rename(columns={col: pretty_names[col] for col in df.columns})
    for i, row in df.iterrows():
        if "_sem" in row.name:
            sem_vals = row.values
            value_row_name = (
                row.name.replace("_sem", "_mean") if not "performance" in row.name else row.name.replace("_sem", "")
            )
            if value_row_name in df.index:
                value_vals = df.loc[df.index == value_row_name].values[0]
                df.loc[df.index == value_row_name] = [
                    f"{float(value):.2f} ({sem:.2f})" if pd.notnull(value) else "N/A"
                    for value, sem in zip(value_vals, sem_vals)
                ]
        else:
            df.loc[i] = [f"{value:.2f}" if pd.notnull(value) else "N/A" for value in row.values]
    df = df[~df.index.str.endswith("_sem")]
    df = df.rename(index={"savings": "rho"})
    df = df.reindex(
        [
            "rho",
            "performance",
            "performance_reduction (%)",
            "total_flops_mean",
            "total_flops_mean_reduction (%)",
            "train_energy_mean",
            "train_energy_mean_reduction (%)",
            "train_time_mean",
            "train_time_mean_reduction (%)",
        ]
    )
    st.write("### Stochastic Top Layers (rho=" + str(rho) + ")")
    st.write(df)


def green_trainer_table(rho=0.5):

    df = pd.read_csv(os.path.join(BASE_DIR, "green_trainer/results.csv"))
    base = get_base_model_performance(df)

    df = get_base_table(df, add_cols=["rho"])
    base["rho"] = "base"
    df = pd.concat([df, base], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["c"] = 17 - fl["nlayer"]
    fl = fl[fl["c"] == 1]
    fl = fl.drop(columns=["nlayer", "c"])
    fl["rho"] = "full LoRA"
    df = pd.concat([df, fl], ignore_index=True)

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3
    for dataset in df["dataset"].unique():
        for metric in ["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]:
            base_value = df.loc[(df["dataset"] == dataset) & (df["rho"] == "full LoRA"), metric].values[0]
            df.loc[df["dataset"] == dataset, f"{metric}_reduction (%)"] = (
                (base_value - df.loc[df["dataset"] == dataset, metric]) / base_value * 100
            )

    df = df[df["rho"].isin([rho])]

    df = df.melt(
        id_vars=["dataset"],
        value_vars=[
            "train_energy_mean",
            "train_time_mean",
            "total_flops_mean",
            "train_energy_sem",
            "train_time_sem",
            "performance",
            "performance_sem",
            "performance_reduction (%)",
            "train_energy_mean_reduction (%)",
            "train_time_mean_reduction (%)",
            "total_flops_mean_reduction (%)",
            "rho",
        ],
        var_name="metric",
        value_name="value",
    )
    df = df.pivot_table(index=["metric"], columns="dataset", values="value", aggfunc="first")
    df = df.reindex(order, axis=1)
    df = df.rename(columns={col: pretty_names[col] for col in df.columns})
    for i, row in df.iterrows():
        if "_sem" in row.name:
            sem_vals = row.values
            value_row_name = (
                row.name.replace("_sem", "_mean") if not "performance" in row.name else row.name.replace("_sem", "")
            )
            if value_row_name in df.index:
                value_vals = df.loc[df.index == value_row_name].values[0]
                df.loc[df.index == value_row_name] = [
                    f"{float(value):.2f} ({sem:.2f})" if pd.notnull(value) else "N/A"
                    for value, sem in zip(value_vals, sem_vals)
                ]
        else:
            df.loc[i] = [f"{value:.2f}" if pd.notnull(value) else "N/A" for value in row.values]
    df = df[~df.index.str.endswith("_sem")]
    df = df.rename(index={"savings": "rho"})
    df = df.reindex(
        [
            "rho",
            "performance",
            "performance_reduction (%)",
            "total_flops_mean",
            "total_flops_mean_reduction (%)",
            "train_energy_mean",
            "train_energy_mean_reduction (%)",
            "train_time_mean",
            "train_time_mean_reduction (%)",
        ]
    )
    st.write("### Green Trainer Top Layers (rho=" + str(rho) + ")")
    st.write(df)


def the_mother_pareto_front():

    df = pd.read_csv(os.path.join(BASE_DIR, "stochastic/results.csv"))
    base = get_base_model_performance(df)
    df = get_base_table(df, add_cols=["savings"])
    df.rename(columns={"savings": "param"}, inplace=True)
    df["approach"] = "stochastic"
    base["savings"] = "base"
    base["approach"] = "base"
    base.rename(columns={"savings": "param"}, inplace=True)
    df = pd.concat([df, base], ignore_index=True)

    gt = pd.read_csv(os.path.join(BASE_DIR, "green_trainer/results.csv"))
    gt = get_base_table(gt, add_cols=["rho"])
    gt = gt.rename(columns={"rho": "param"})
    gt["approach"] = "green_trainer"
    df = pd.concat([df, gt], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["param"] = 17 - fl["nlayer"]
    fl = fl.drop(columns=["nlayer"])
    fl["approach"] = "static"
    df = pd.concat([df, fl], ignore_index=True)

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3

    # create a matplotlib figure with a scatter plot of the pareto front
    # there are 16 datasets, therefore create a 4x4 grid of subplots
    num_rows = 6
    num_cols = 3
    grid = [st.columns(num_cols, vertical_alignment="bottom") for _ in range(num_rows)]
    approaches = df["approach"].unique()
    # sort so that base is first
    approaches = sorted(approaches, key=lambda x: x == "base", reverse=True)
    for ds in df["dataset"].unique():
        row = order.index(ds) // num_cols
        col = order.index(ds) % num_cols
        fig, ax = plt.subplots(figsize=(4, 3))
        for approach in approaches:
            data_subset = df[df["dataset"] == ds]
            subset = data_subset[data_subset["approach"] == approach]
            if approach == "base":
                # add a line of the base model performance
                ax.axhline(y=subset["performance"].values[0], color="black", linestyle="--")
                # add annotation for the base model performance
                ax.annotate(
                    "Base Model",
                    (data_subset["train_energy_mean"].min(), subset["performance"].values[0]),
                    textcoords="offset points",
                    xytext=(-5, 5),
                    ha="left",
                    fontsize=8,
                    color="black",
                )
                continue
            if approach == "static":
                # add v line for full LoRA energy
                full_lora_energy = subset[subset["param"] == 1]["train_energy_mean"].values[0]
                ax.axvline(x=full_lora_energy, color="orange", linestyle="--")
                ax.annotate(
                    "Full LoRA",
                    (full_lora_energy, (data_subset["performance"].min())),
                    textcoords="offset points",
                    xytext=(-7, 10),
                    ha="center",
                    fontsize=8,
                    color="orange",
                    rotation=90,
                )

            if subset.empty:
                continue
            param_dsc = {"green_trainer": "rho", "stochastic": "e", "static": "min. l"}
            approach_dsc = {
                "green_trainer": "Green Trainer",
                "stochastic": "Stochastic Top Layers",
                "static": "Static Top Layers",
            }
            color = (
                "green"
                if approach == "green_trainer"
                else "grey" if approach == "static" else "blue" if approach == "stochastic" else "black"
            )
            ax.scatter(
                subset["train_energy_mean"],
                subset["performance"],
                label=f"{approach_dsc[approach]} ({param_dsc[approach]})",
                s=50,
                color=color,
            )
            # Add annotation for each dot with its param value
            for idx, row_ in subset.iterrows():
                annot = str(row_["param"])
                ax.annotate(
                    annot,
                    (row_["train_energy_mean"], row_["performance"]),
                    textcoords="offset points",
                    xytext=(0, -15),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
        ax.set_title(f"{pretty_names[ds]}")
        ax.set_xlabel("Train Energy (kJ)")
        performance_dsc = {
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
            "allenai_task219_rocstories_title_answer_generation": "ROUGE-L",
            "allenai_task288_gigaword_summarization": "ROUGE-L",
            "alpaca_mmlu": "ROUGE-L",
        }
        ax.set_ylabel(performance_dsc[ds])
        # set x ticks to be in scienfic notation
        # ax.set_xticks(ax.get_xticks()[::2])  # Show every second xtick
        ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()])
        # ax.set_yticks(ax.get_yticks()[::2])  # Show every second ytick
        ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()])
        if row == 0 and col == 0:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=1)
        grid[row][col].pyplot(fig)


def average_pareto_front():

    df = pd.read_csv(os.path.join(BASE_DIR, "stochastic/results.csv"))
    base = get_base_model_performance(df)
    df = get_base_table(df, add_cols=["savings"])
    df.rename(columns={"savings": "param"}, inplace=True)
    df["approach"] = "stochastic"
    base["savings"] = "base"
    base["approach"] = "base"
    base.rename(columns={"savings": "param"}, inplace=True)
    df = pd.concat([df, base], ignore_index=True)

    gt = pd.read_csv(os.path.join(BASE_DIR, "green_trainer/results.csv"))
    gt = get_base_table(gt, add_cols=["rho"])
    gt = gt.rename(columns={"rho": "param"})
    gt["approach"] = "green_trainer"
    df = pd.concat([df, gt], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["param"] = 17 - fl["nlayer"]
    fl = fl.drop(columns=["nlayer"])
    fl["approach"] = "static"
    df = pd.concat([df, fl], ignore_index=True)

    approaches = df["approach"].unique()

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3

    df["group"] = df["dataset"].apply(
        lambda x: "NLU" if x in task_groups["NLU"] else ("reasoning" if x in task_groups["reasoning"] else "NLG")
    )
    gdf = (
        df.groupby(["group", "approach", "param"])
        .agg(
            train_energy_mean=("train_energy_mean", "mean"),
            performance_mean=("performance", "mean"),
            total_flops_mean=("total_flops_mean", "mean"),
            total_time_mean=("total_time_mean", "mean"),
            train_energy_sem=("train_energy_sem", "mean"),
            performance_sem=("performance_sem", "mean"),
            total_time_sem=("total_time_sem", "mean"),
        )
        .reset_index()
    )
    for metric in ["train_energy_mean", "total_flops_mean", "total_time_mean"]:
        cols = st.columns(3, vertical_alignment="bottom")
        for group in gdf["group"].unique():
            # create three pareto front plots for each group
            fig, ax = plt.subplots(figsize=(4, 3))
            subset = gdf[gdf["group"] == group]
            fl_energy = subset.loc[(subset["approach"] == "static") & (subset["param"] == 1), metric].values[0]
            subset[metric] = (fl_energy - subset[metric]) / fl_energy * 100
            col = list(task_groups.keys()).index(group)
            # st.write(subset)
            for approach in ["static", "green_trainer", "stochastic"]:
                subset_ = subset[subset["approach"] == approach]
                if subset_.empty or approach == "base":
                    continue
                if approach == "static":
                    # add v line for full LoRA energy
                    full_lora_energy = subset_[subset_["param"] == 1][metric].values[0]
                    ax.axvline(x=full_lora_energy, color="black", linestyle="--")
                    # ax.annotate(
                    #     "Full LoRA",
                    #     (full_lora_energy, (subset_["performance_mean"].min())),
                    #     textcoords="offset points",
                    #     xytext=(-7, 10),
                    #     ha="center",
                    #     fontsize=8,
                    #     color="black",
                    #     rotation=90,
                    # )
                param_dsc = {"green_trainer": "rho", "stochastic": "e", "static": "min. l"}
                approach_dsc = {
                    "green_trainer": "Green Trainer",
                    "stochastic": "Stochastic Top Layers",
                    "static": "Static Top Layers",
                }
                color = (
                    colors["greenish"]
                    if approach == "green_trainer"
                    else (
                        colors["blueish"]
                        if approach == "stochastic"
                        else colors["redish"] if approach == "static" else "black"
                    )
                )
                ax.errorbar(
                    subset_[metric],
                    subset_["performance_mean"],
                    # yerr=subset_["performance_sem"],
                    # xerr=subset_[f"{metric.replace('_mean', '_sem')}"] if not "flops" in metric else None,
                    fmt="o",
                    label=f"{approach_dsc[approach]} ({param_dsc[approach]})",
                    markersize=6,
                    color=color,
                    capsize=3,
                )
                # Add annotation for each dot with its param value
                for idx, row_ in subset_.iterrows():
                    ax.annotate(
                        str(row_["param"]),
                        (row_[metric], row_["performance_mean"]),
                        textcoords="offset points",
                        xytext=(0, -12) if approach == "static" else (0, 8),
                        ha="center",
                        fontsize=8,
                        color="black",
                    )
            ax.set_title("Reasoning" if group == "reasoning" else group)
            ax.set_xlabel(
                "Avg. Energy Savings (%)"
                if metric == "train_energy_mean"
                else "Avg. FLOPs Savings (%)" if metric == "total_flops_mean" else "Avg. Time Savings (%)"
            )
            performance_dsc = {
                "NLU": "Avg. Performance (Accuracy)",
                "reasoning": "Avg. Performance (Accuracy)",
                "NLG": "Avg. Performance (ROUGE-L)",
            }
            ax.set_ylabel(performance_dsc[group])
            # set x ticks to be in scienfic notation
            ax.set_xticks(ax.get_xticks()[::1])  # Show every second xtick
            ax.set_xticklabels([f"{tick:.0f}" for tick in ax.get_xticks()])
            ax.set_yticks(ax.get_yticks()[::1])  # Show every second ytick
            ax.set_yticklabels([f"{tick:.0f}" for tick in ax.get_yticks()])
            if group == "NLU" and metric == "train_energy_mean":
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.5), ncol=1)
            # ax.legend()
            cols[col].pyplot(fig)


def the_mother_table():

    df = pd.read_csv(os.path.join(BASE_DIR, "stochastic/results.csv"))
    base = get_base_model_performance(df)
    df = get_base_table(df, add_cols=["savings"])
    df.rename(columns={"savings": "param"}, inplace=True)
    df["approach"] = "stochastic"
    base["savings"] = "base"
    base["approach"] = "base"
    base.rename(columns={"savings": "param"}, inplace=True)
    df = pd.concat([df, base], ignore_index=True)

    gt = pd.read_csv(os.path.join(BASE_DIR, "green_trainer/results.csv"))
    gt = get_base_table(gt, add_cols=["rho"])
    gt = gt.rename(columns={"rho": "param"})
    gt["approach"] = "green_trainer"
    df = pd.concat([df, gt], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["param"] = 17 - fl["nlayer"]
    fl = fl.drop(columns=["nlayer"])
    fl["approach"] = "static"
    df = pd.concat([df, fl], ignore_index=True)

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3
        if "time" in col:
            # minutes
            df[col] /= 60

    for dataset in df["dataset"].unique():
        for metric in ["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]:
            base_value = df.loc[(df["dataset"] == dataset) & (df["param"] == 1), metric].values[0]
            df.loc[df["dataset"] == dataset, f"{metric}_reduction (%)"] = (
                (base_value - df.loc[df["dataset"] == dataset, metric]) / base_value * 100
            )

    df = df.melt(
        id_vars=["dataset", "approach", "param"],
        value_vars=[
            "train_energy_mean",
            "train_time_mean",
            "total_flops_mean",
            "train_energy_sem",
            "train_time_sem",
            "performance",
            "performance_sem",
            "performance_reduction (%)",
            "train_energy_mean_reduction (%)",
            "train_time_mean_reduction (%)",
            "total_flops_mean_reduction (%)",
        ],
        var_name="metric",
        value_name="value",
    )
    df = df.pivot_table(
        index=["approach", "param", "metric"], columns="dataset", values="value", aggfunc="first"
    ).reset_index()

    for i, row in df.iterrows():
        if "_sem" in row["metric"]:
            sem_vals = row.values[3:]
            value_row_name = (
                row["metric"].replace("_sem", "_mean")
                if not "performance" in row["metric"]
                else row["metric"].replace("_sem", "")
            )
            mask = (df["metric"] == value_row_name) & (df["approach"] == row["approach"]) & (df["param"] == row["param"])
            if any(mask):
                value_vals = df.loc[mask].values[0][3:]
                df.iloc[mask, 3:] = [
                    f"{float(value):.2f}" + "\\tiny{ " + f"({sem:.2f})" + "}" if pd.notnull(value) else "N/A"
                    for value, sem in zip(value_vals, sem_vals)
                ]
        else:
            df.loc[i, df.columns[3:]] = [f"{value:.2f}" if pd.notnull(value) else "N/A" for value in row.values[3:]]
    df = df[~df["metric"].str.endswith("_sem")]
    df["approach"] = df["approach"].replace(
        {
            "stochastic": "Top-LoRA Stochastic",
            "green_trainer": "GreenTrainer",
            "static": "Top-LoRA Static",
            "base": "Pre-Trained Model",
        }
    )

    copy_latex = ""
    copy_latex += "\\newcolumntype{Y}{>{\\raggedright\\arraybackslash}p{1.3cm}}\n"
    copy_latex += "\\newcolumntype{C}{>{\\columncolor{lightgreenish}}X}\n"
    copy_latex += "\\newcolumntype{L}{>{\\columncolor{lightgreenish}}l}\n"

    for i, approach in enumerate(df["approach"].unique()):
        subset = df[df["approach"] == approach]

        group_a = task_groups["NLU"]
        group_b = task_groups["reasoning"] + task_groups["NLG"]

        for param in subset["param"].unique():
            # st.write(f"#### Parameter: {param}")
            subset_param = subset[subset["param"] == param]
            copy_latex += "      \\begin{table}\n"
            copy_latex += "      \\centering\n"
            copy_latex += "      \\footnotesize\n"
            copy_latex += "      \\setlength{\\tabcolsep}{3pt}\n"
            copy_latex += "      \\renewcommand{\\arraystretch}{1.2}\n"
            for grp in [group_a, group_b]:
                subset_grp = subset_param[["param", "metric"] + grp]
                # st.write(subset_grp)
                copy_latex += latex_table(subset_grp, is_a="glue_rte" in grp)
                copy_latex += "    \\vspace{1em}\n\n"
            param_dsc = {
                "Pre-Trained Model": None,
                "GreenTrainer": "$\\rho$",
                "Top-LoRA Stochastic": "$e$",
                "Top-LoRA Static": "$s_{\\text{stat.}}(\\mathcal{B}_m)$",
            }
            desc = (
                " after fine-tuning using using \\textbf{"
                + approach
                + "}"
                + (" with " + param_dsc[approach] + " = " + str(param) + ". ")
                if param_dsc[approach]
                else " before fine-tuning (Pre-Trained Model)."
            )
            caption = (
                "Averaged validation performance, training time $t$, energy consumption $E$ (each with standard error in parentheses), and PFLOPs grouped by task type"
                + desc
                + "Relative reductions (performance degradation and energy savings) are shown in italics. A green column indicates that the fine-tuning performance is within 3 \% relative degradation compared to full LoRA."
            )
            copy_latex += "\\caption{" + caption + "}\n"
            copy_latex += "\\label{tbl:all_results}\n"
            copy_latex += "\\end{table}\n"

    st.text_area(
        "Copy LaTeX Table",
        value=copy_latex,
        height=300,
        help="Copy this LaTeX table to your document.",
    )


def latex_table(df, is_a=True):

    coloring = [
        float(df.loc[df["metric"] == "performance_reduction (%)", col].values[0]) < 3
        for col in df.columns
        if col not in ["metric", "param"]
    ]

    latex = ""
    if is_a:
        columns = "Y?" + "".join(["C" if color else "X" for color in coloring])
        latex += "    \\begin{tabularx}{\\textwidth}{" + columns + "}\n"
        latex += "        & \multicolumn{8}{c}{\\textit{NLU / GLUE}} \\\\\n"
        latex += "        & \\textbf{CoLA} & \\textbf{SST2} & \\textbf{MRPC} & \\textbf{QQP} & \\textbf{MNLIm} & \\textbf{MNLImm} & \\textbf{QNLI} & \\textbf{RTE} \\\\\n"
    else:
        columns = (
            "Y?"
            + "".join(["C" if color else "X" for color in coloring[0:4]])
            + ("L" if coloring[4] else "l")
            + "|"
            + "".join(["C" if color else "X" for color in coloring[5:]])
        )
        latex += "    \\begin{tabularx}{\\textwidth}{" + columns + "}\n"
        latex += "        & \\multicolumn{5}{c}{\\textit{Reasoning}} & \multicolumn{3}{c}{\\textit{NLG}} \\\\\n"
        latex += "        & \\textbf{ARC-E.} & \\textbf{ARC-C.} & \\textbf{PIQA} & \\textbf{BoolQ} & \\textbf{HellaSwag} & \\textbf{ROCStor.} & \\textbf{Gigaword} & \\textbf{Alpaca} \\\\\n"
    latex += "        \\specialrule{2pt}{1pt}{1pt}\n"

    # sort rows so that performance is first, then energy, then time, then flops
    df = df.sort_values(
        by=[
            "metric",
        ],
        key=lambda x: x.map(
            {
                "performance": 0,
                "performance_reduction (%)": 1,
                "train_energy_mean": 2,
                "train_energy_mean_reduction (%)": 3,
                "train_time_mean": 4,
                "train_time_mean_reduction (%)": 5,
                "total_flops_mean": 6,
                "total_flops_mean_reduction (%)": 7,
            }
        ),
    )
    # st.write(df)
    for i, row in df.iterrows():
        metric_dsc = {
            "performance": "Perform.",
            "train_energy_mean": "$E$ (kJ)",
            "train_time_mean": "$t$ (min)",
            "total_flops_mean": "PFLOPs",
        }
        if "reduction" in row["metric"]:
            continue
        latex += (
            "        "
            + metric_dsc[row["metric"]]
            + " & "
            + " & ".join([el for el in row.values[2:] if pd.notnull(el)])
            + " \\\\\n"
        )
        reduction_row = df.loc[df["metric"] == row["metric"] + "_reduction (%)"]
        if not reduction_row.empty:
            latex += (
                "       "
                + " & "
                + " & ".join(["\\textit{-" + el + " \%}" for el in reduction_row.values[0][2:] if pd.notnull(el)])
                + " \\\\\n"
            )
        latex += "       \\hline\n"
    # remove last hline
    latex = latex[:-9] + "\n"
    latex += "     \\end{tabularx}\n"
    return latex


def avg_table():
    df = pd.read_csv(os.path.join(BASE_DIR, "stochastic/results.csv"))
    base = get_base_model_performance(df)
    df = get_base_table(df, add_cols=["savings"])
    df.rename(columns={"savings": "param"}, inplace=True)
    df["approach"] = "stochastic"
    base["savings"] = "base"
    base["approach"] = "base"
    base.rename(columns={"savings": "param"}, inplace=True)
    df = pd.concat([df, base], ignore_index=True)

    gt = pd.read_csv(os.path.join(BASE_DIR, "green_trainer/results.csv"))
    gt = get_base_table(gt, add_cols=["rho"])
    gt = gt.rename(columns={"rho": "param"})
    gt["approach"] = "green_trainer"
    df = pd.concat([df, gt], ignore_index=True)

    fl = pd.read_csv(os.path.join(BASE_DIR, "static/results.csv"))
    fl = get_base_table(fl, add_cols=["nlayer"])
    fl["param"] = 17 - fl["nlayer"]
    fl = fl.drop(columns=["nlayer"])
    fl["approach"] = "static"
    df = pd.concat([df, fl], ignore_index=True)

    selection = {
        "stochastic": [0.25, 0.5, 0.75],
        "green_trainer": [0.25, 0.5, 0.75],
        "static": [1, 5, 9, 13, 16],
    }
    sel_df = pd.DataFrame(columns=df.columns)
    for key, val in selection.items():
        for v in val:
            sel_df = pd.concat([sel_df, df[(df["approach"] == key) & (df["param"] == v)]], ignore_index=True)
    grp_df = (
        sel_df.groupby(["approach", "param"])
        .agg(
            train_energy_mean=("train_energy_mean", "mean"),
            train_time_mean=("train_time_mean", "mean"),
            total_flops_mean=("total_flops_mean", "mean"),
            performance=("performance", "mean"),
            train_energy_sem=("train_energy_sem", lambda x: np.sqrt(np.sum(x**2))),
            train_time_sem=("train_time_sem", lambda x: np.sqrt(np.sum(x**2))),
            performance_sem=("performance_sem", lambda x: np.sqrt(np.sum(x**2))),
        )
        .reset_index()
    )
    grp_df["approach"] = grp_df["approach"].replace(
        {
            "stochastic": "Top-LoRA Stochastic",
            "green_trainer": "GreenTrainer",
            "static": "Top-LoRA Static",
        }
    )

    grp_df["approach"] = grp_df["approach"].astype(str) + " " + grp_df["param"].astype(str)
    grp_df = grp_df.drop(columns=["param"]).set_index("approach").T

    values = grp_df.loc[["train_energy_mean", "train_time_mean", "total_flops_mean", "performance"]]
    sems = grp_df.loc[["train_energy_sem", "train_time_sem", "performance_sem"]]

    for i, row in values.iterrows():

        row_values = row.values
        if "flops" in row.name:
            # convert to pFLOPs
            row_values = row.values / 1e15
        if "energy" in row.name:
            # convert to kJ
            row_values = row.values / 1e3
        if "time" in row.name:
            # convert to minutes
            row_values = row.values / 60

        red_row_name = (
            row.name.replace("_mean", "_reduction (%)") if not "performance" in row.name else row.name + "_reduction (%)"
        )
        baseline = values.loc[values.index == row.name, "Top-LoRA Static 1.0"].values[0]
        if "flops" in row.name:
            # convert to pFLOPs
            baseline /= 1e15
        if "energy" in row.name:
            # convert to kJ
            baseline /= 1e3
        if "time" in row.name:
            # convert to minutes
            baseline /= 60
        grp_df.loc[red_row_name] = [
            f"{-100 * (baseline - value) / baseline:.2f}%" if pd.notnull(value) else "N/A" for value in row_values
        ]

        sem_row_name = row.name.replace("_mean", "_sem") if not "performance" in row.name else row.name + "_sem"
        if not sem_row_name in sems.index:
            grp_df.loc[grp_df.index == row.name] = [
                f"{float(value):.2f}" if pd.notnull(value) else "N/A" for value in row_values
            ]
            continue
        sem_vals = sems.loc[sems.index == sem_row_name]
        if "flops" in row.name:
            # convert to pFLOPs
            sem_vals /= 1e15
        if "energy" in row.name:
            # convert to kJ
            sem_vals /= 1e3
        if "time" in row.name:
            # convert to minutes
            sem_vals /= 60
        grp_df.loc[grp_df.index == row.name] = [
            f"{float(value):.2f} ({sem:.2f})" if pd.notnull(value) else "N/A"
            for value, sem in zip(row_values, sem_vals.values[0])
        ]
    grp_df = grp_df[~grp_df.index.str.endswith("_sem")]
    grp_df = grp_df.reindex(
        [
            # "performance",
            "performance_reduction (%)",
            # "train_energy_mean",
            "train_energy_reduction (%)",
            # "train_time_mean",
            "train_time_reduction (%)",
            # "total_flops_mean",
            "total_flops_reduction (%)",
        ]
    )
    # set column order
    order = [
        "Top-LoRA Static 5.0",
        "Top-LoRA Static 9.0",
        "Top-LoRA Static 13.0",
        "Top-LoRA Static 16.0",
        "Top-LoRA Stochastic 0.25",
        "Top-LoRA Stochastic 0.5",
        "Top-LoRA Stochastic 0.75",
        "GreenTrainer 0.75",
        "GreenTrainer 0.5",
        "GreenTrainer 0.25",
    ]
    grp_df = grp_df[order]
    st.write(grp_df)


if __name__ == "__main__":

    # streamlit run ./ftt/results/plotting/results_tables.py --server.fileWatcherType=poll
    st.set_page_config(
        page_title="Result Tables",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    # static_selection_table()

    # st.write("---")
    # static_optimal_table()

    # st.write("---")
    # stochastic_table(0.75)

    # st.write("---")
    # stochastic_table(0.5)

    # st.write("---")
    # green_trainer_table(0.5)

    # st.write("---")
    # green_trainer_table(0.75)

    # st.write("---")
    # the_mother_pareto_front()

    st.write("---")
    average_pareto_front()

    # st.write("---")
    # the_mother_table()

    # st.write("---")
    # avg_table()
