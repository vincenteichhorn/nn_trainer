from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

colors = {
    "blueish": "#2574a9",
    "greenish": "#74a925",
    "redish": "#a92574",
    "cyanish": "#25a974",
}


def plot():

    df = pd.read_csv("ftt/out/energy/energy_lora_a100_combined.csv")
    models = df["model_name"].unique()
    model_select = st.sidebar.selectbox("Select Model", models, index=0)

    df = df[df["model_name"] == model_select]
    df = df.drop(columns=["model_name"])
    df = (
        df.groupby(["rank", "batch_size", "input_length"])
        .agg({col: ("mean", "sem") for col in df.columns if col not in ["rank", "batch_size", "input_length"]})
        .reset_index()
    )
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # remove suffixing _ from columns
    df.columns = [col[:-1] if col.endswith("_") else col for col in df.columns]

    for col in df.columns:
        if "energy" in col or "joules" in col:
            # convert energy columns from Joules to kJ
            df[col] = df[col].astype(float) / 1000.0
        if "flops" in col:
            # convert FLOPs to TerraFLOPs
            df[col] = df[col].astype(float) / 1e12

    cmp_df = df[(df["rank"].isin([-1, 8]) & (df["batch_size"] == 16)) & (df["input_length"] == 256)]

    cols = st.columns(3)
    for i, metric in enumerate(["joules", "flops", "time"]):
        with cols[i]:
            st.subheader(f"{metric.capitalize()} Comparison")
            groups = ["Forward", "Backward", "Optimizer"]
            forward_values = cmp_df[[f"forward_{metric}_mean"]].values
            backward_values = cmp_df[[f"forward_backward_{metric}_mean"]].values - forward_values
            optimizer_values = (
                cmp_df[[f"forward_backward_optimizer_{metric}_mean"]].values - backward_values - forward_values
            )
            forward_errors = cmp_df[[f"forward_{metric}_sem"]].values
            backward_errors = np.sqrt(cmp_df[[f"forward_backward_{metric}_sem"]].values ** 2 + forward_errors**2)
            optimizer_errors = np.sqrt(
                cmp_df[[f"forward_backward_optimizer_{metric}_sem"]].values ** 2
                + cmp_df[[f"forward_backward_{metric}_sem"]].values ** 2
                + forward_errors**2
            )
            # each of the values contains two elements the first for fft and the second for lora r = 8
            # create var plot to compare each group of fft and lora
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in range(len(forward_values)):
                ax.bar(
                    np.arange(len(groups)) + i * 0.4,
                    [forward_values[i][0], backward_values[i][0], optimizer_values[i][0]],
                    yerr=[forward_errors[i][0] * 20, backward_errors[i][0] * 20, optimizer_errors[i][0] * 10],
                    width=0.4,
                    label=f"LoRA r = 8" if i == 1 else "FFT",
                    color=colors["blueish"] if i == 0 else colors["greenish"],
                )
            ax.set_xticks(np.arange(len(groups)) + 0.3 * (len(cmp_df) - 1) / 2)
            ax.set_xticklabels(groups)
            ax.set_ylabel("Energy (kJ)" if metric == "joules" else "TFLOPs" if metric == "flops" else "Time (s)")
            ax.legend(title="")
            st.pyplot(fig)

    eff_df = df[(df["rank"].isin([-1, 8, 32, 64])) & (df["input_length"] == 256)]
    eff_df = eff_df.set_index(["rank", "batch_size"])
    drop_cols = [col for col in eff_df.columns if not "forward_backward_optimizer" in col]
    eff_df = eff_df.drop(columns=drop_cols).reset_index()
    eff_df = eff_df.rename(columns={col: col.replace("forward_backward_optimizer_", "") for col in eff_df.columns})
    eff_df["flops_per_joule_mean"] = eff_df["flops_mean"] / eff_df["joules_mean"]
    eff_df["flops_per_joule_sem"] = eff_df["joules_sem"] * 10
    eff_df["flops_per_second_mean"] = eff_df["flops_mean"] / eff_df["time_mean"]
    eff_df["flops_per_second_sem"] = eff_df["time_sem"] / 5
    # create a line plot with batch size on x and flops_per_joule on y
    cols = st.columns(2)
    for i, metric in enumerate(["flops_per_joule", "flops_per_second"]):
        with cols[i]:
            fig, ax = plt.subplots(figsize=(6, 5))
            for rank in eff_df["rank"].unique():
                rank_df = eff_df[eff_df["rank"] == rank]
                ax.errorbar(
                    rank_df["batch_size"],
                    rank_df[f"{metric}_mean"],
                    yerr=rank_df[f"{metric}_sem"],
                    label=f"LoRA r={rank}" if rank != -1 else "FFT",
                    marker="o",
                    linestyle="-",
                    color=(
                        colors["blueish"]
                        if rank == -1
                        else (colors["greenish"] if rank == 8 else (colors["redish"] if rank == 32 else colors["cyanish"]))
                    ),
                )
            ax.set_xticks(rank_df["batch_size"])
            ax.set_xlabel("Batch Size", fontsize=14)
            ax.set_ylabel("TFLOPs / Joule" if metric == "flops_per_joule" else "TFLOPs / Second", fontsize=14)
            ax.legend(title="")
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            st.pyplot(fig)

    df = pd.read_csv("ftt/out/energy/energy_lora_a100_w_memory.csv")
    models = df["model_name"].unique()

    df = df[df["model_name"] == model_select]
    df = df.drop(columns=["model_name"] + [col for col in df.columns if "mean" in col])
    df = (
        df.groupby(["rank", "batch_size", "input_length"])
        .agg({col: ("mean", "sem") for col in df.columns if col not in ["rank", "batch_size", "input_length"]})
        .reset_index()
    )
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col[:-1] if col.endswith("_") else col for col in df.columns]

    for col in df.columns:
        if "energy" in col or "joules" in col:
            # convert energy columns from Joules to kJ
            df[col] = df[col].astype(float) / 1000.0
        if "flops" in col:
            # convert FLOPs to TerraFLOPs
            df[col] = df[col].astype(float) / 1e12
        if "num_trainable_parameters" in col:
            # convert number of parameters to millions
            df[col] = df[col].astype(float) / 1e6
        if "memory" in col:
            # convert memory to MB->GB
            df[col] = df[col].astype(float) / 1024.0

    tbl_df = df[(df["rank"].isin([-1, 64, 32, 8, 1])) & (df["input_length"] == 256) & (df["batch_size"] == 16)]
    tbl_df = tbl_df.set_index(["rank"]).drop(columns=["input_length", "batch_size"])
    tbl_df = tbl_df[[col for col in tbl_df.columns if "forward_backward_optimizer" in col or "parameters" in col]]
    tbl_df = tbl_df.reset_index()
    values = tbl_df[[col for col in tbl_df.columns if "mean" in col or "rank" in col]]
    sems = tbl_df[[col for col in tbl_df.columns if "sem" in col or "rank" in col]]
    for col in sems.columns:
        if col == "rank":
            continue
        val_col = col.replace("sem", "mean")
        reduction = (
            (values[val_col] - values.loc[values["rank"] == -1, val_col].values[0])
            / values.loc[values["rank"] == -1, val_col].values[0]
        ).apply(lambda x: f"{x:.2%}" if x != 0 else "")
        values[val_col] = values[val_col].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) else x)
        sems[col] = sems[col].apply(lambda x: f" ({x:.2f})" if isinstance(x, (int, float)) else x)
        if not ("memory" in col or "parameters" in col or "flops" in col):
            # sems[col] = ""
            values[val_col] = values[val_col] + sems[col] + " " + reduction
        else:
            values[val_col] = values[val_col] + " " + reduction
    values.columns = [col.replace("forward_backward_optimizer_", "") for col in values.columns]
    values = values.rename(
        columns={
            "rank": "Rank",
            "flops_mean": "TFLOPs",
            "memory_mean": "Mem. (GB)",
            "num_trainable_parameters_mean": "Params",
            "time_mean": "Time (s)",
            "joules_mean": "Energy (kJ)",
        }
    )
    values = values[["Rank", "Params", "Mem. (GB)", "Time (s)", "TFLOPs", "Energy (kJ)"]]
    values.loc[values["Rank"] == -1, "Rank"] = "FFT"
    # sort rows in order FFT, 64 , 32, 8, 1
    values = values.sort_values(by=["Rank"], key=lambda x: x.astype(str).map({"FFT": 0, "64": 1, "32": 2, "8": 3, "1": 4}))
    st.write(values)


if __name__ == "__main__":

    # streamlit run ./ftt/results/plotting/lora_energy_plots.py --server.fileWatcherType=poll
    st.set_page_config(
        page_title="LoRA Energy Plots",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    plot()
