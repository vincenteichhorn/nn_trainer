import os
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
    st.write(df)

    for col in df.columns:
        if "flops" in col:
            # pFLOPs
            df[col] /= 1e15
        if "energy" in col:
            # kJ
            df[col] /= 1e3

    # create a matplotlib figure with a scatter plot of the pareto front
    # there are 16 datasets, therefore create a 4x4 grid of subplots
    num_rows = 4
    num_cols = 4
    grid = [st.columns(num_cols) for _ in range(num_rows)]
    approaches = df["approach"].unique()
    # sort so that base is first
    approaches = sorted(approaches, key=lambda x: x == "base", reverse=True)
    for ds in df["dataset"].unique():
        row = order.index(ds) // num_cols
        col = order.index(ds) % num_cols
        fig, ax = plt.subplots(figsize=(4, 3))
        for approach in approaches:
            subset = df[(df["dataset"] == ds) & (df["approach"] == approach)]
            if approach == "base":
                # add a line of the base model performance
                ax.axhline(y=subset["performance"].values[0], color="black", linestyle="--", label="Base Model")
                continue
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
                ax.annotate(
                    str(row_["param"]),
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
            ax.legend()
        grid[row][col].pyplot(fig)

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
        )
        .reset_index()
    )
    st.write(gdf)
    for metric in ["train_energy_mean", "total_flops_mean", "total_time_mean"]:
        cols = st.columns(3)
        for group in gdf["group"].unique():
            # create three pareto front plots for each group
            fig, ax = plt.subplots(figsize=(4, 3))
            subset = gdf[gdf["group"] == group]
            col = list(task_groups.keys()).index(group)
            for approach in approaches:
                subset_ = subset[subset["approach"] == approach]
                if subset_.empty or approach == "base":
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
                    subset_[metric],
                    subset_["performance_mean"],
                    label=f"{approach_dsc[approach]} ({param_dsc[approach]})",
                    s=50,
                    color=color,
                )
                # Add annotation for each dot with its param value
                for idx, row_ in subset_.iterrows():
                    ax.annotate(
                        str(row_["param"]),
                        (row_[metric], row_["performance_mean"]),
                        textcoords="offset points",
                        xytext=(0, -15),
                        ha="center",
                        fontsize=8,
                        color="black",
                    )
            ax.set_title("Reasoning" if group == "reasoning" else group)
            ax.set_xlabel(
                "Energy (kJ)" if metric == "train_energy_mean" else "TFLOPs" if metric == "total_flops_mean" else "Time (s)"
            )
            performance_dsc = {
                "NLU": "Average Performance",
                "reasoning": "Average Performance",
                "NLG": "Average ROUGE-L",
            }
            ax.set_ylabel(performance_dsc[group])
            # set x ticks to be in scienfic notation
            # ax.set_xticks(ax.get_xticks()[::2])  # Show every second xtick
            ax.set_xticklabels([f"{tick:.1f}" for tick in ax.get_xticks()])
            # ax.set_yticks(ax.get_yticks()[::2])  # Show every second ytick
            ax.set_yticklabels([f"{tick:.1f}" for tick in ax.get_yticks()])
            if group == "NLU" and metric == "train_energy_mean":
                ax.legend()
            cols[col].pyplot(fig)


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
    static_optimal_table()

    st.write("---")
    stochastic_table(0.75)

    st.write("---")
    stochastic_table(0.5)

    st.write("---")
    green_trainer_table(0.5)

    # st.write("---")
    the_mother_pareto_front()
