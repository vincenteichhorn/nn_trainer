import os
import signal
import subprocess
import warnings
from datetime import datetime
from typing import List, Tuple, Any

import torch
import pandas as pd
from multiprocessing import Process, Value, Event

from nnt.profiling.profiler import Profiler
from nnt.profiling.multiprocessing_util import ResultHandler, MPQueueResultHandler, FileCacheResultHandler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class NvidiaProfiler(Profiler):
    """
    Profiler for GPU energy consumption as a context manager using nvidia-smi.
    Starts a separate process for profiling using nvidia-smi, collects power and memory usage, and supports caching results.

    Example:
        with NvidiaProfiler(interval=10) as prof:
            # ... run your GPU code ...
        df = prof.to_pandas()
        print(df.head())
    """

    interval: float
    data: list
    should_profiling_run: Value  # type: ignore
    profiling_started: Event  # type: ignore
    profiling_stopped: Event  # type: ignore
    result_handler: ResultHandler
    process: Process
    gpu_clock_speed: int | None

    def __init__(
        self,
        interval: int = 1,
        cache_file: str | None = None,
        force_cache: bool = False,
        gpu_clock_speed: int | None = None,
        read_only: bool = False,
    ):
        """
        Initialize the NvidiaProfiler.

        Args:
            interval (int): Interval in milliseconds for profiler steps.
            cache_file (str or None): File path to store profiling data in a CSV file.
            force_cache (bool): If True, overwrite the cache file if it exists.
            gpu_clock_speed (int or None): Set GPU clock speed in MHz (optional).
            read_only (bool): If True, only read data from cache file, do not start profiling process.
        """
        self.interval: float = interval
        self.data: List[Tuple[int, datetime, float]] = []
        self.should_profiling_run: Value = Value("i", 1)  # type: ignore
        self.profiling_started: Event = Event()  # type: ignore
        self.profiling_stopped: Event = Event()  # type: ignore
        self.result_handler: ResultHandler = (
            MPQueueResultHandler() if not cache_file else FileCacheResultHandler(cache_file, force_cache)
        )
        self.result_handler.set_columns(
            ("gpu_id", "timestamp", "power", "memory", "record_step"),
            (int, str, float, float, str),
        )
        self.process: Process = Process(
            target=NvidiaProfiler._nvidiasmi_profiling_process,
            args=(
                self.should_profiling_run,
                self.profiling_started,
                self.profiling_stopped,
                self.result_handler,
                self.interval,
            ),
        )
        self.gpu_clock_speed = gpu_clock_speed
        if gpu_clock_speed is not None:
            for gpu_id in range(torch.cuda.device_count()):
                self.set_gpu_clock_speed(gpu_id, gpu_clock_speed)
        if not read_only:
            super().__init__()

    def __del__(self):
        """
        Destructor to ensure the process is terminated and GPU clock speed is reset if the object is deleted.
        """
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
        if self.gpu_clock_speed is not None:
            for gpu_id in range(torch.cuda.device_count()):
                self.reset_gpu_clock_speed(gpu_id)

    def set_gpu_clock_speed(self, gpu_id: int, clock_speed: int) -> None:
        """
        Set GPU clock speed to a specific value.
        This doesn't guarantee a fixed value due to throttling, but can help reduce variance.

        Args:
            gpu_id (int): The GPU ID to set the clock speed for.
            clock_speed (int): The clock speed in MHz to set.
        """
        # check if sudo nvidia-smi is available
        if subprocess.getstatusoutput("sudo nvidia-smi")[0] != 0:
            # warnings.warn("could not set gpu clock speed, sudo nvidia-smi command not found. assure sudo rights.")
            return
        subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {gpu_id}", shell=True)
        subprocess.run(f"sudo nvidia-smi -lgc {clock_speed} -i {gpu_id}", shell=True)

    def reset_gpu_clock_speed(self, gpu_id: int) -> None:
        """
        Reset GPU clock speed to default values.

        Args:
            gpu_id (int): The GPU ID to reset.
        """
        if subprocess.getstatusoutput("sudo nvidia-smi")[0] != 0:
            # warnings.warn("could not reset gpu clock speed, sudo nvidia-smi command not found. assure sudo rights.")
            return
        subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {gpu_id}", shell=True)
        subprocess.run(f"sudo nvidia-smi -rgc -i {gpu_id}", shell=True)

    def record_step(self, name: str) -> None:
        """
        Record a named step for profiling and add a marker to the result handler.

        Args:
            name (str): The name of the step to record.
        Raises:
            ValueError: If name is '__unset__'.
        """
        if name == "__unset__":
            raise ValueError("Cannot record step with name '__unset__'. Use a different name.")
        super().record_step(name)
        self.result_handler.put((-1, datetime.now(), -1.0, -1.0, name))

    @staticmethod
    def _nvidiasmi_profiling_process(
        should_run: Value,  # type: ignore
        started: Event,  # type: ignore
        stopped: Event,  # type: ignore
        result_handler: ResultHandler,
        interval: int,
    ) -> None:
        """
        Static method for the separate profiling process. Opens a subprocess with nvidia-smi and saves GPU ID, timestamp, power, and memory for every interval.

        Args:
            should_run (multiprocessing.Value): Set to 1 to run, 0 to stop profiling.
            started (multiprocessing.Event): Notifies when profiling starts.
            stopped (multiprocessing.Event): Notifies when profiling ends.
            result_handler (ResultHandler): Handles shared memory for results.
            interval (int): Interval in milliseconds for nvidia-smi polling.
        """
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        def read_data(ln):
            try:
                vals: List[Any] = ln.strip().split(", ")
                gid: int = int(vals[0])
                ts: datetime = datetime.strptime(vals[1], "%Y/%m/%d %H:%M:%S.%f")
                pwr: float = float(vals[2].split(" ")[0])
                mem: float = float(vals[3].split(" ")[0])
                return (gid, ts, pwr, mem, "__unset__")
            except ValueError as e:
                warnings.warn(f"Error parsing line '{ln.strip()}':\n{e}")
                return None

        with (
            subprocess.Popen(
                ["nvidia-smi", "--query-gpu=index,timestamp,power.draw,memory.used", "--format=csv", f"-lms={interval}"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid,
            ) as nvidiasmi_process,
            result_handler as result,
        ):
            with nvidiasmi_process.stdout as out:
                _ = out.readline()
                if should_run.value:
                    started.set()
                while should_run.value:
                    data = read_data(out.readline())
                    if data is None:
                        continue
                    result.put(data)
        result.put(None)
        stopped.set()
        nvidiasmi_process.terminate()
        try:
            nvidiasmi_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(nvidiasmi_process.pid), signal.SIGKILL)

    def __enter__(self) -> "NvidiaProfiler":
        """
        Start the profiling process and wait until profiling has started.

        Returns:
            NvidiaProfiler: The profiler instance.
        """
        assert subprocess.getstatusoutput("nvidia-smi")[0] == 0, "Could not find nvidia-smi tool"
        self.process.start()
        self.profiling_started.wait()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        """
        Stop the profiling process, collect all data, and terminate the process.
        """
        try:
            self.should_profiling_run.value = 0
            self.profiling_stopped.wait(timeout=5)
            self.data = self.result_handler.get_all()
        finally:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=5)
            self.data = self._data_post_process(self.data)

    def start(self) -> None:
        """
        Start profiling (calls __enter__).
        """
        self.__enter__()

    def stop(self) -> None:
        """
        Stop profiling (calls __exit__).
        """
        self.__exit__()

    def _data_post_process(self, data: list) -> list:
        """
        Post-process the data to associate each measurement with the correct record step.

        Args:
            data (List[Tuple[int, datetime, float, float, str]]): Raw data to post-process.
        Returns:
            List[Tuple[int, datetime, float, float]]: Processed data with GPU ID, timestamp, power, memory, and record step.
        """
        processed_data: List[Tuple[int, datetime, float, float]] = []
        current_record_step: str = self.record_steps[0][1] if hasattr(self, "record_steps") else "__init__"
        for gpu_id, timestamp, power, memory, record_step in data:
            if record_step != "__unset__":
                current_record_step = record_step
                continue
            else:
                processed_data.append((gpu_id, timestamp, power, memory, current_record_step))
        if len(processed_data) == 0:
            return data
        return processed_data

    @staticmethod
    def from_cache(cache_file: str) -> "NvidiaProfiler":
        """
        Create a NvidiaProfiler object from a cache file. Only loads data; does not start profiling.

        Args:
            cache_file (str): Path to the cache file.
        Returns:
            NvidiaProfiler: Profiler object with loaded data.
        """
        prof = NvidiaProfiler(cache_file=cache_file, read_only=True)
        prof.data = prof.result_handler.get_all()
        prof.data = prof._data_post_process(prof.data)
        return prof

    def to_pandas(self) -> pd.DataFrame:
        """
        Generate a pandas DataFrame from the profiled data.

        Returns:
            pd.DataFrame: DataFrame with columns gpu_id, timestamp, power (W), memory (MiB), record_step.
        """
        df: pd.DataFrame = pd.DataFrame(self.data, columns=["gpu_id", "timestamp", "power", "memory", "record_step"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        return df

    def get_profiled_gpus(self) -> list:
        """
        Get a list of all GPU IDs that were profiled.

        Returns:
            List[int]: List of profiled GPU IDs.
        """
        df: pd.DataFrame = self.to_pandas()
        return df["gpu_id"].unique().tolist()

    def get_total_energy(
        self,
        gpu_ids: list | None = None,
        record_steps: list | None = None,
        return_data: bool = False,
    ) -> float:
        """
        Sum the power to get the total energy for specified GPUs and record steps.

        Args:
            gpu_ids (List[int], optional): GPU IDs to calculate energy for. Defaults to first GPU.
            record_steps (List[str], optional): Record steps to include. Defaults to all.
            return_data (bool, optional): If True, return raw data for each record step.
        Returns:
            float or list: Total energy in watt-seconds, or list of measurements if return_data is True.
        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_ids = gpu_ids or [df["gpu_id"].unique()[0]]
        df = df[df["gpu_id"].isin(gpu_ids)]
        df["time_interval"] = df["timestamp"].diff().dt.total_seconds().fillna(0)
        df["energy_interval"] = df["power"] * df["time_interval"]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["energy_interval"].sum())
        return df["energy_interval"].sum()

    def get_total_time(
        self,
        gpu_ids: list | None = None,
        record_steps: list | None = None,
        return_data: bool = False,
    ) -> float:
        """
        Calculate the total profiling time in seconds for specified GPUs and record steps.

        Args:
            gpu_ids (List[int], optional): GPU IDs to calculate time for. Defaults to first GPU.
            record_steps (List[str], optional): Record steps to include. Defaults to all.
            return_data (bool, optional): If True, return raw data for each record step.
        Returns:
            float or list: Total profiling time in seconds, or list of measurements if return_data is True.
        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_ids = gpu_ids or [df["gpu_id"].unique()[0]]
        df = df[df["gpu_id"].isin(gpu_ids)]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        time_spans = list(df.groupby("record_step_id")["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds()))
        if return_data:
            return time_spans
        return sum(time_spans)

    def get_max_memory(
        self, gpu_id: int | None = None, record_steps: list | None = None, return_data: bool = False
    ) -> float:
        """
        Get the maximum memory usage for a specified GPU and record steps.

        Args:
            gpu_id (int, optional): GPU ID to calculate max memory for. Defaults to first GPU.
            record_steps (List[str], optional): Record steps to include. Defaults to all.
            return_data (bool, optional): If True, return raw data for each record step.
        Returns:
            float or list: Maximum memory usage in MiB, or list of measurements if return_data is True.
        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_id = gpu_id or df["gpu_id"].unique()[0]
        df = df[df["gpu_id"] == gpu_id]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["memory"].max())
        return df["memory"].max()

    def get_mean_memory(
        self, gpu_id: int | None = None, record_steps: list | None = None, return_data: bool = False
    ) -> float:
        """
        Get the average memory usage for a specified GPU and record steps.

        Args:
            gpu_id (int, optional): GPU ID to calculate average memory for. Defaults to first GPU.
            record_steps (List[str], optional): Record steps to include. Defaults to all.
            return_data (bool, optional): If True, return raw data for each record step.
        Returns:
            float or list: Average memory usage in MiB, or list of measurements if return_data is True.
        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_id = gpu_id or df["gpu_id"].unique()[0]
        df = df[df["gpu_id"] == gpu_id]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["memory"].mean())
        return df["memory"].mean()

    def get_time_series_plot(self) -> go.Figure:
        """
        Create a Plotly figure with the recorded time series data for power and memory usage.

        Returns:
            plotly.graph_objects.Figure: The generated time series plot.
        """
        if not self.data:
            return None
        df: pd.DataFrame = self.to_pandas()
        fig: go.Figure = make_subplots(specs=[[{"secondary_y": True}]])

        profiled_gpus: List[int] = self.get_profiled_gpus()
        n_colors: int = max(len(profiled_gpus), 2)
        color_scale: List[str] = px.colors.sample_colorscale("Rainbow", [n / (n_colors - 1) for n in range(n_colors)])
        for i, gpu_id in enumerate(profiled_gpus):
            gpu_df = df[df["gpu_id"] == gpu_id]
            for trace, unit in [("power", "W"), ("memory", "MiB")]:
                fig.add_trace(
                    go.Scatter(
                        x=gpu_df["timestamp"],
                        y=gpu_df[trace],
                        name=f"{trace.capitalize()} ({unit})",
                        mode="lines+markers",
                        legendgroup=gpu_id,
                        legendgrouptitle_text=f"GPU #{gpu_id}",
                        line=dict(
                            color=color_scale[i],
                            width=4,
                            dash="dot" if trace == "memory" else "solid",
                        ),
                    ),
                    secondary_y=(trace == "memory"),
                )

        max_timestamp: datetime = df["timestamp"].max()
        plt_record_steps: List[Tuple[datetime, str]] = self.record_steps + [(max_timestamp, ".")]
        unique_record_step_names = set([name for _, name in plt_record_steps])
        n_colors: int = max(len(unique_record_step_names), 2)
        color_scale: List[str] = px.colors.sample_colorscale("viridis", [n / (n_colors - 1) for n in range(n_colors)])
        colors = {name: color_scale[i] for i, name in enumerate(unique_record_step_names)}
        (last_ts, last_name) = plt_record_steps[0]
        for ts, name in plt_record_steps[1:]:
            fig.add_vrect(
                x0=last_ts,
                x1=ts,
                annotation_text=last_name,
                annotation_position="top left",
                line_width=0,
                opacity=0.25,
                fillcolor=colors[last_name],
            )
            last_ts, last_name = ts, name

        fig.update_layout(title="GPU Memory and Power Usage", legend=dict(groupclick="toggleitem"))
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Power (W)", secondary_y=False)
        fig.update_yaxes(title_text="Memory (MiB)", secondary_y=True)

        return fig
