import argparse
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.width", 2000)

pareto_front = pd.DataFrame(
    columns=["dataset", "x", "y", "x_sem", "y_sem", "annotation", "is_dominated", "legend_group", "color"]
)

exp_names = [
    "glue_cola",
    "glue_mrpc",
    "glue_qnli",
    # "glue_mnli_mismatched",
    "glue_qqp",
    "glue_rte",
    "glue_sst2",
    "glue_mnli_matched",
    "arc_easy",
    "arc_challenge",
    "piqa",
    "boolq",
    "hellaswag",
    "alpaca_mmlu",
    "allenai_task288_gigaword_summarization",
    "allenai_task219_rocstories_title_answer_generation",
]

selected_energy_metric = st.selectbox(
    "Select Energy Metric",
    ["flops_savings", "rel_flops_savings", "energy_savings", "rel_energy_savings", "time_savings", "rel_time_savings"],
    index=2,
)

options_list = [
    "show_error",
    "print_full_lora_baseline",
    "print_basemodel_performance",
    "show_basemodel_performance",
]
options = st.multiselect("Options", options_list, default=[])


def get_x_axis_metrics(exp_name):
    return selected_energy_metric
    # return "flops_savings"
    # return "rel_flops_savings"
    # return "energy_savings"
    # return "time_savings"
    # return "rel_energy_savings"


def get_x_axis_name(exp_name):
    dct = {
        "flops_savings": "Abs. FLOPs Savings (Compared to Full LoRA)",
        "rel_flops_savings": "Rel. FLOPs Savings (%)",
        "energy_savings": "Abs. Energy Savings (Compared to Full LoRA)",
        "rel_energy_savings": "Rel. Energy Savings (%)",
        "time_savings": "Abs. Time Savings (Compared to Full LoRA)",
        "rel_time_savings": "Rel. Time Savings (%)",
    }
    return dct[get_x_axis_metrics(exp_name)]


def get_baseline_energy_metric_name(exp_name):
    dct = {
        "flops_savings": "total_flops_mean",
        "rel_flops_savings": "total_flops_mean",
        "energy_savings": "train_energy_mean",
        "rel_energy_savings": "train_energy_mean",
        "time_savings": "train_time_mean",
        "rel_time_savings": "train_time_mean",
    }
    return dct[get_x_axis_metrics(exp_name)]


def get_baseline_energy_sem_name(exp_name):
    dct = {
        "flops_savings": "zero",
        "rel_flops_savings": "zero",
        "energy_savings": "train_energy_sem",
        "rel_energy_savings": "train_energy_sem",
        "time_savings": "train_time_sem",
        "rel_time_savings": "train_time_sem",
    }
    return dct[get_x_axis_metrics(exp_name)]


def get_energy_metric_name(exp_name):
    dct = {
        "flops_savings": "flops_savings",
        "rel_flops_savings": "rel_flops_savings",
        "energy_savings": "energy_savings",
        "rel_energy_savings": "rel_energy_savings",
        "time_savings": "time_savings",
        "rel_time_savings": "rel_time_savings",
    }
    return dct[get_x_axis_metrics(exp_name)]


def get_energy_sem_name(exp_name):
    dct = {
        "flops_savings": "zero",
        "rel_flops_savings": "zero",
        "energy_savings": "energy_savings_sem",
        "rel_energy_savings": "rel_energy_savings_sem",
        "time_savings": "time_savings_sem",
        "rel_time_savings": "rel_time_savings_sem",
    }
    return dct[get_x_axis_metrics(exp_name)]


def get_y_axis_metric(exp_name):
    dct = {
        "glue_cola": "OneHotClassificationMetrics_mcc",
        "glue_mrpc_backup": "OneHotClassificationMetrics_f1_score",
        "glue_mrpc": "OneHotClassificationMetrics_f1_score",
        "glue_qqp": "OneHotClassificationMetrics_f1_score",
        "alpaca_mmlu": "RougeScore_rougeL",
        "allenai_task288_gigaword_summarization": "RougeScore_rougeL",
        "allenai_task219_rocstories_title_answer_generation": "RougeScore_rougeL",
    }
    if exp_name in dct:
        return dct[exp_name]
    return "OneHotClassificationMetrics_accuracy"


def get_performance_metric_name(exp_name):
    return get_y_axis_metric(exp_name) + "_max_mean"


def get_pretrained_performance_metric_name(exp_name):
    return get_y_axis_metric(exp_name) + "_min_mean"


def get_performance_sem_name(exp_name):
    return get_y_axis_metric(exp_name) + "_max_sem"


def get_y_axis_name(exp_name):
    dct = {
        "glue_cola": "MCC",
        "glue_mrpc": "F1",
        "glue_qqp": "F1",
        "glue_mnli_mismatched": "Mismatched Accuracy",
        "glue_mnli_matched": "Matched Accuracy",
        "alpaca_mmlu": "ROUGE-L",
        "allenai_task288_gigaword_summarization": "ROUGE-L",
        "allenai_task219_rocstories_title_answer_generation": "ROUGE-L",
    }
    if exp_name in dct:
        return dct[exp_name]
    return "Accuracy"


def get_subplot_title(exp_name):
    dct = {
        "glue_cola": "CoLA",
        "glue_mrpc": "MRPC",
        "glue_mnli_mismatched": "MNLI",
        "glue_sst2": "SST-2",
        "glue_mnli_matched": "MNLI",
        "glue_qqp": "QQP",
        "glue_qnli": "QNLI",
        "glue_rte": "RTE",
        "arc_easy": "ARC Easy",
        "arc_challenge": "ARC Challenge",
        "boolq": "BoolQ",
        "hellaswag": "HellaSwag",
        "alpaca_mmlu": "Alpaca",
        "allenai_task288_gigaword_summarization": "AllanAI NI Gigaword Summarization",
        "allenai_task219_rocstories_title_answer_generation": "AllanAI NI RocStories Title Answer Generation",
    }
    if exp_name in dct:
        return dct[exp_name]
    return exp_name.replace("_", " ").upper()


def create_pareto_front(exp_names, x_axis_names, y_axis_names):
    fig = make_subplots(
        rows=len(exp_names) // 4 + 1,
        cols=4,
        subplot_titles=[get_subplot_title(el) for el in exp_names],
    )
    for i, exp in enumerate(exp_names):
        row = i // 4 + 1
        col = i % 4 + 1
        fig.update_xaxes(title_text=x_axis_names[exp], row=row, col=col)
        fig.update_yaxes(title_text=y_axis_names[exp], row=row, col=col)
    return fig


def add_to_pareto_front(
    fig, exp_names, df, x_cols, y_cols, x_sem_cols, y_sem_cols, annotation_cols, color="blue", legend_group=None
):
    fig.data = [el for el in fig.data if "data point" not in el.name]

    if legend_group is not None:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="lines",
                name=legend_group,
                showlegend=True,
                legendgroup=legend_group,
                line=dict(color=color, width=20),
            ),
            row=1,
            col=1,
        )
    for i, exp in enumerate(exp_names):
        row = i // 4 + 1
        col = i % 4 + 1
        exp_df = df[df["dataset"] == exp]
        if y_cols[exp] in exp_df.columns and x_cols[exp] in exp_df.columns:
            exp_pareto_front = exp_df[
                ["dataset", x_cols[exp], y_cols[exp], x_sem_cols[exp], y_sem_cols[exp], annotation_cols[exp]]
            ].copy()
            exp_pareto_front["is_dominated"] = False
            exp_pareto_front["legend_group"] = legend_group
            exp_pareto_front["color"] = color
            exp_pareto_front.rename(
                columns={
                    x_cols[exp]: "x",
                    y_cols[exp]: "y",
                    x_sem_cols[exp]: "x_sem",
                    y_sem_cols[exp]: "y_sem",
                    annotation_cols[exp]: "annotation",
                },
                inplace=True,
            )
            exp_pareto_front["y"] *= 100
            global pareto_front
            pareto_front = pd.concat([pareto_front, exp_pareto_front], ignore_index=True)
            pareto_front.loc[pareto_front["dataset"] == exp, "is_dominated"] = pareto_front[
                pareto_front["dataset"] == exp
            ].apply(
                lambda row: any(
                    (pareto_front.loc[pareto_front["dataset"] == exp, "x"] > row["x"])
                    & (pareto_front.loc[pareto_front["dataset"] == exp, "y"] > row["y"])
                    & (pareto_front.loc[pareto_front["dataset"] == exp, "is_dominated"] == False)
                ),
                axis=1,
            )
        for j, dot in pareto_front[pareto_front["dataset"] == exp].iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[dot["x"]],
                    y=[dot["y"]],
                    mode="markers+text",
                    text=[dot["annotation"]],
                    textposition="top center",
                    textfont=dict(size=10, color=dot["color"]),
                    name=f"{dot['dataset']} {dot['annotation']} data point",
                    marker=dict(
                        color=dot["color"],
                        size=6 if dot["is_dominated"] else 12,
                        symbol="x" if dot["is_dominated"] else "circle",
                    ),
                    showlegend=False,
                    legendgroup=dot["legend_group"],
                    error_x=dict(
                        type="data",
                        array=[dot["x_sem"]],
                        visible="show_error" in options,
                        color=dot["color"],
                    ),
                    error_y=dict(
                        type="data",
                        array=[dot["y_sem"]],
                        visible="show_error" in options,
                        color=dot["color"],
                    ),
                ),
                row=row,
                col=col,
            )

            min_x = min(pareto_front.loc[pareto_front["dataset"] == exp, "x"])
            max_x = max(pareto_front.loc[pareto_front["dataset"] == exp, "x"])
            min_y = min(pareto_front.loc[pareto_front["dataset"] == exp, "y"])
            max_y = max(pareto_front.loc[pareto_front["dataset"] == exp, "y"])
            # fig.update_xaxes(range=[min_x - 0.1 * (max_x - min_x), max_x + 0.1 * (max_x - min_x)], row=row, col=col)
            # fig.update_yaxes(range=[min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y)], row=row, col=col)
    return fig


def compute_energy_savings(df, baselines, baselines_sems):
    df["zero"] = 0.01
    df["energy_savings"] = np.nan
    for exp in exp_names:
        exp_df = df[df["dataset"] == exp]
        baseline = baselines[exp]
        baseline_sem = baselines_sems[exp]
        df.loc[df["dataset"] == exp, "energy_savings"] = baseline - exp_df["train_energy_mean"]
        df.loc[df["dataset"] == exp, "energy_savings_sem"] = baseline_sem + exp_df["train_energy_sem"]
        df.loc[df["dataset"] == exp, "rel_energy_savings"] = (baseline - exp_df["train_energy_mean"]) / baseline * 100
        df.loc[df["dataset"] == exp, "rel_energy_savings_sem"] = (baseline_sem - exp_df["train_energy_sem"]) / baseline * 100
        df.loc[df["dataset"] == exp, "flops_savings"] = baseline - exp_df["total_flops_mean"]
        df.loc[df["dataset"] == exp, "rel_flops_savings"] = (baseline - exp_df["total_flops_mean"]) / baseline * 100
        df.loc[df["dataset"] == exp, "time_savings"] = baseline - exp_df["train_time_mean"]
        df.loc[df["dataset"] == exp, "time_savings_sem"] = baseline_sem + exp_df["train_time_sem"]
        df.loc[df["dataset"] == exp, "rel_time_savings"] = (baseline - exp_df["train_time_mean"]) / baseline * 100
        df.loc[df["dataset"] == exp, "rel_time_savings_sem"] = (baseline_sem - exp_df["train_time_sem"]) / baseline * 100
    return df


def plt(st_obj: st = st):

    OUT_DIR = "/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/out/"

    x_axis_metrics = {exp: get_energy_metric_name(exp) for exp in exp_names}
    y_axis_metrics = {exp: get_performance_metric_name(exp) for exp in exp_names}
    x_axis_sem_metrics = {exp: get_energy_sem_name(exp) for exp in exp_names}
    y_axis_sem_metrics = {exp: get_performance_sem_name(exp) for exp in exp_names}
    x_axis_names = {exp: get_x_axis_name(exp) for exp in exp_names}
    y_axis_names = {exp: get_y_axis_name(exp) for exp in exp_names}

    fig = create_pareto_front(exp_names, x_axis_names, y_axis_names)

    DIR = os.path.join(OUT_DIR, "static/")
    top_df = pd.read_csv(os.path.join(DIR, "results.csv"))
    top_df["nlayer"] = top_df["nlayer"].astype(int)
    top_df["zero"] = 0.0
    baselines = {
        exp: top_df.loc[(top_df["dataset"] == exp) & (top_df["nlayer"] == 16), get_baseline_energy_metric_name(exp)].values[
            0
        ]
        for exp in exp_names
    }
    baselines_sems = {
        exp: top_df.loc[(top_df["dataset"] == exp) & (top_df["nlayer"] == 16), get_baseline_energy_sem_name(exp)].values[0]
        for exp in exp_names
    }
    baseline_performance = {
        exp: top_df.loc[(top_df["dataset"] == exp) & (top_df["nlayer"] == 16), y_axis_metrics[exp]].values[0] * 100
        for exp in exp_names
    }
    baseline_performance_sem = {
        exp: top_df.loc[(top_df["dataset"] == exp) & (top_df["nlayer"] == 16), y_axis_sem_metrics[exp]].values[0] * 100
        for exp in exp_names
    }
    baselines_df = pd.DataFrame(
        {
            "dataset": exp_names,
            "x": [baselines[exp] for exp in exp_names],
            "x_sem": [baselines_sems[exp] for exp in exp_names],
            "y": [baseline_performance[exp] for exp in exp_names],
            "y_sem": [baseline_performance_sem[exp] for exp in exp_names],
            "annotation": "Full LoRA",
        }
    )
    pretrained_model_perfornmance = {
        exp: top_df.loc[
            (top_df["dataset"] == exp) & (top_df["nlayer"] == 16), get_pretrained_performance_metric_name(exp)
        ].values[0]
        * 100
        for exp in exp_names
    }
    pretrained_model_perfornmance = pd.DataFrame(
        {
            "dataset": exp_names,
            "y": [pretrained_model_perfornmance[exp] for exp in exp_names],
        }
    )
    if "print_full_lora_baseline" in options:
        st.write(baselines_df)
    if "print_basemodel_performance" in options:
        st.write(pretrained_model_perfornmance)

    top_df = compute_energy_savings(top_df, baselines, baselines_sems)
    annotation_cols = {exp: "nlayer" for exp in exp_names}
    fig = add_to_pareto_front(
        fig,
        exp_names,
        top_df,
        x_axis_metrics,
        y_axis_metrics,
        x_axis_sem_metrics,
        y_axis_sem_metrics,
        annotation_cols,
        color="grey",
        legend_group="Static (nlayer)",
    )

    try:
        BETA_DIR = os.path.join(OUT_DIR, "stochastic/")
        beta_df = pd.read_csv(os.path.join(BETA_DIR, "results.csv"))
        beta_df = compute_energy_savings(beta_df, baselines, baselines_sems)
        annotation_cols = {exp: "savings" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            beta_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="greenyellow",
            legend_group="Stochastic (s)",
        )
    except FileNotFoundError:
        pass

    try:
        GT_DIR = os.path.join(OUT_DIR, "green_trainer/")
        gt_df = pd.read_csv(os.path.join(GT_DIR, "results.csv"))
        gt_df = compute_energy_savings(gt_df, baselines, baselines_sems)
        annotation_cols = {exp: "rho" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            gt_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="red",
            legend_group="Green Trainer (rho)",
        )
    except FileNotFoundError:
        pass

    try:
        ADPT_DIR = os.path.join(OUT_DIR, "adaptive/")
        adpt_df = pd.read_csv(os.path.join(ADPT_DIR, "results.csv"))
        adpt_df = compute_energy_savings(adpt_df, baselines, baselines_sems)
        det_df = adpt_df[adpt_df["approach"] == "deterministic"]
        annotation_cols = {exp: "rho" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            det_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="orange",
            legend_group="Adaptive (Deterministic) (rho)",
        )
        stoch_df = adpt_df[adpt_df["approach"] == "stochastic"]
        stoch_df["annotation"] = ""
        annotation_cols = {exp: "annotation" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            stoch_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="purple",
            legend_group="Adaptive (Stochastic)",
        )
    except FileNotFoundError:
        pass

    try:
        BAN_DIR = os.path.join(OUT_DIR, "bandits/")
        ban_df = pd.read_csv(os.path.join(BAN_DIR, "results.csv"))
        ban_df = compute_energy_savings(ban_df, baselines, baselines_sems)
        dlinucb_df = ban_df[ban_df["bandit"] == "dUCB"]
        dlinucb_df["annotation"] = (
            dlinucb_df["gamma"].astype(str)
            + "-"
            + dlinucb_df["lmda"].astype(str)
            + "-"
            + dlinucb_df["delta"].astype(str)
            + "-"
            + dlinucb_df["sigma"].astype(str)
        )
        annotation_cols = {exp: "annotation" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            dlinucb_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="magenta",
            legend_group="dLinUCB Bandit (gamma-lambda-delta-sigma)",
        )
        bayesian_df = ban_df[ban_df["bandit"] == "Bayesian"]
        bayesian_df["annotation"] = bayesian_df["alpha"].astype(str) + "-" + bayesian_df["beta"].astype(str)
        annotation_cols = {exp: "annotation" for exp in exp_names}
        fig = add_to_pareto_front(
            fig,
            exp_names,
            bayesian_df,
            x_axis_metrics,
            y_axis_metrics,
            x_axis_sem_metrics,
            y_axis_sem_metrics,
            annotation_cols,
            color="cyan",
            legend_group="Bayesian Bandit (alpha-beta)",
        )
    except FileNotFoundError:
        pass

    fig.update_layout(
        height=400 * (len(exp_names) // 4 + 1),
        width=1200,
    )

    if "show_basemodel_performance" in options:
        for i, exp in enumerate(exp_names):
            row = i // 4 + 1
            col = i % 4 + 1
            fig.add_hline(
                y=pretrained_model_perfornmance.loc[pretrained_model_perfornmance["dataset"] == exp, "y"].values[0],
                line_dash="dash",
                line_color="black",
                annotation_text=f"Pretrained Model",
                annotation_position="top left",
                annotation_font_color="black",
                annotation_font_size=10,
                row=row,
                col=col,
            )

    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":

    # streamlit run ./ftt/results/plotting/pareto_front.py --server.fileWatcherType=poll
    st.set_page_config(
        page_title="Pareto Front",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    plt()
