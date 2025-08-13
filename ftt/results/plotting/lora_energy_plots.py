from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Define color palette for plots
colors = {
    "blueish": "#2574a9",
    "greenish": "#74a925",
    "redish": "#a92574",
    "cyanish": "#25a974",
}


def plot():
    # Load main energy results CSV
    df = pd.read_csv("ftt/out/energy/energy_lora_a100_combined.csv")
    models = df["model_name"].unique()
    # Sidebar model selection
    model_select = st.sidebar.selectbox("Select Model", models, index=0)

    # Filter for selected model
    df = df[df["model_name"] == model_select]
    df = df.drop(columns=["model_name"])
    # Group by rank, batch_size, input_length and aggregate mean and sem
    df = (
        df.groupby(["rank", "batch_size", "input_length"])
        .agg({col: ("mean", "sem") for col in df.columns if col not in ["rank", "batch_size", "input_length"]})
        .reset_index()
    )
    # Flatten multi-level columns
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # Remove trailing underscores from column names
    df.columns = [col[:-1] if col.endswith("_") else col for col in df.columns]

    # Unit conversions for energy and FLOPs
    for col in df.columns:
        if "energy" in col or "joules" in col:
            df[col] = df[col].astype(float) / 1000.0  # Joules to kJ
        if "flops" in col:
            # Convert FLOPs to PFLOPs
            df[col] = df[col].astype(float) / 1e15 * 100

    # Filter for comparison plot (FFT vs LoRA r=8)
    cmp_df = df[(df["rank"].isin([-1, 8]) & (df["batch_size"] == 16)) & (df["input_length"] == 256)]

    # Plot bar charts for joules, flops, and time
    cols = st.columns(3, vertical_alignment="bottom")
    for i, metric in enumerate(["joules", "flops", "time"]):
        with cols[i]:
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
            # Plot bars for FFT and LoRA r=8
            fig, ax = plt.subplots(figsize=(4, 3))
            for i in range(len(forward_values)):
                ax.bar(
                    np.arange(len(groups)) + i * 0.4,
                    [forward_values[i][0], backward_values[i][0], optimizer_values[i][0]],
                    yerr=[forward_errors[i][0], backward_errors[i][0], optimizer_errors[i][0]],
                    width=0.4,
                    label=f"LoRA r = 8" if i == 1 else "FFT",
                    color=colors["blueish"] if i == 0 else colors["greenish"],
                )
            ax.set_xticks(np.arange(len(groups)) + 0.3 * (len(cmp_df) - 1) / 2)
            ax.set_xticklabels(groups)
            ax.set_ylabel("Energy (kJ)" if metric == "joules" else "PFLOPs" if metric == "flops" else "Time (s)")
            ax.set_ylim(bottom=0)
            if metric == "flops":
                ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
            st.pyplot(fig)

    # Efficiency plots: GFLOPs/Joule and GFLOPs/Second
    eff_df = df[(df["rank"].isin([-1, 8, 32, 64])) & (df["input_length"] == 256)]
    eff_df = eff_df.set_index(["rank", "batch_size"])
    drop_cols = [col for col in eff_df.columns if not "forward_backward_optimizer" in col]
    eff_df = eff_df.drop(columns=drop_cols).reset_index()
    eff_df = eff_df.rename(columns={col: col.replace("forward_backward_optimizer_", "") for col in eff_df.columns})
    eff_df["flops_per_joule_mean"] = eff_df["flops_mean"] / eff_df["joules_mean"]
    eff_df["flops_per_joule_sem"] = eff_df["joules_sem"]
    eff_df["flops_per_second_mean"] = eff_df["flops_mean"] / (eff_df["time_mean"])
    eff_df["flops_per_second_sem"] = eff_df["time_sem"]
    # Line plots for efficiency metrics
    cols = st.columns(2)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for i, metric in enumerate(["flops_per_joule", "flops_per_second"]):
        with cols[i]:
            ax = axs[i]
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
            ax.set_ylabel("PFLOPs / kJ" if metric == "flops_per_joule" else "PFLOPs / Second", fontsize=14)
            ax.legend(title="", loc="upper left", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # save fig as svg
    fig.tight_layout()
    fig.savefig(f"ftt/out/figs/efficiency_{model_select.replace("/", "_")}.svg", format="svg", bbox_inches="tight")
    st.pyplot(fig)

    # Load memory results CSV
    df = pd.read_csv("ftt/out/energy/energy_lora_a100_w_memory.csv")
    models = df["model_name"].unique()

    # Filter for selected model and drop mean columns
    df = df[df["model_name"] == model_select]
    df = df.drop(columns=["model_name"] + [col for col in df.columns if "mean" in col])
    # Group and aggregate mean and sem
    df = (
        df.groupby(["rank", "batch_size", "input_length"])
        .agg({col: ("mean", "sem") for col in df.columns if col not in ["rank", "batch_size", "input_length"]})
        .reset_index()
    )
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    df.columns = [col[:-1] if col.endswith("_") else col for col in df.columns]

    # Unit conversions for memory, parameters, etc.
    for col in df.columns:
        if "energy" in col or "joules" in col:
            df[col] = df[col].astype(float) / 1000.0  # Joules to kJ
        if "flops" in col:
            df[col] = df[col].astype(float) / 1e12
        if "num_trainable_parameters" in col:
            df[col] = df[col].astype(float) / 1e6  # Params to millions
        if "memory" in col:
            df[col] = df[col].astype(float) / 1024.0  # MB to GB

    # Prepare summary table for selected ranks, batch size, and input length
    tbl_df = df[(df["rank"].isin([-1, 64, 32, 8, 1])) & (df["input_length"] == 256) & (df["batch_size"] == 16)]
    tbl_df = tbl_df.set_index(["rank"]).drop(columns=["input_length", "batch_size"])
    tbl_df = tbl_df[[col for col in tbl_df.columns if "forward_backward_optimizer" in col or "parameters" in col]]
    tbl_df = tbl_df.reset_index()
    values = tbl_df[[col for col in tbl_df.columns if "mean" in col or "rank" in col]]
    sems = tbl_df[[col for col in tbl_df.columns if "sem" in col or "rank" in col]]
    # Format values and calculate reduction percentages
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
            values[val_col] = values[val_col] + sems[col] + " " + reduction
        else:
            values[val_col] = values[val_col] + " " + reduction
    # Rename columns for display
    values.columns = [col.replace("forward_backward_optimizer_", "") for col in values.columns]
    values = values.rename(
        columns={
            "rank": "Rank",
            "flops_mean": "PFLOPs",
            "memory_mean": "Mem. (GB)",
            "num_trainable_parameters_mean": "Params",
            "time_mean": "Time (s)",
            "joules_mean": "Energy (kJ)",
        }
    )
    values = values[["Rank", "Params", "Mem. (GB)", "Time (s)", "PFLOPs", "Energy (kJ)"]]
    values.loc[values["Rank"] == -1, "Rank"] = "FFT"
    # Sort rows in order FFT, 64, 32, 8, 1
    values = values.sort_values(by=["Rank"], key=lambda x: x.astype(str).map({"FFT": 0, "64": 1, "32": 2, "8": 3, "1": 4}))
    # st.write(values)
    str_values = values.to_string(index=False, justify="left")
    st.write(str_values)


if __name__ == "__main__":
    # Streamlit entry point
    # streamlit run ./ftt/results/plotting/lora_energy_plots.py --server.fileWatcherType=poll
    st.set_page_config(
        page_title="LoRA Energy Plots",
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    load_dotenv()
    plot()
