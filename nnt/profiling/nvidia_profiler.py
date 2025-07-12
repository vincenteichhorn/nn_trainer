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
    """ "
    Profiler for gpu energy consumption as context manager using nvidia-smi.
    The Profiler starts a seperate process for profiling using nvidia-smi.
    If one enters the context the parallel process gets started and it waits until the profiling got some data before continuing
    the execution inside the context. If the context closes all data is collected and the seperate process gets killed.

    Args:
        interval (int): milliseconds interval of profiler steps
        cache_file (str or None): file path to store the profiling data in a csv file. If None the data is stored in memory
            and collected when the profiler context exits otherwise a csv file is created and the data is written to it and read from it
            after the context exits
        force_cache (bool): if True the cache file will be overwritten if it already exists
    """

    def __init__(
        self,
        interval: int = 1,
        cache_file: str | None = None,
        force_cache: bool = False,
        gpu_clock_speed: int | None = None,  # in MHz, set to None to not change the clock speed
        read_only: bool = False,  # if True the profiler will not start the process and only read the data from the cache file
    ):
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
        Destructor to ensure the process is terminated if the object is deleted.
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
            gpu_id (int): the id of the gpu to set the clock speed for
            clock_speed (int): the clock speed in MHz to set
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
        """
        if subprocess.getstatusoutput("sudo nvidia-smi")[0] != 0:
            # warnings.warn("could not reset gpu clock speed, sudo nvidia-smi command not found. assure sudo rights.")
            return
        subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {gpu_id}", shell=True)
        subprocess.run(f"sudo nvidia-smi -rgc -i {gpu_id}", shell=True)

    def record_step(self, name):
        if name == "__unset__":
            raise ValueError("Cannot record step with name '__unset__'. Use a different name.")
        super().record_step(name)
        self.result_handler.put((-1, datetime.now(), -1.0, -1.0, name))

    @staticmethod
    def _nvidiasmi_profiling_process(
        should_run: Value, started: Event, stopped: Event, result_handler: ResultHandler, interval: int  # type: ignore
    ):
        """
        Static method for the seperate profiling process. Use this method in a multiprocessing process.
        Opens a subprocess with nvidia-smi and saves gpu_id, timestamp and power for every interval seconds and puts these values in the queue.

        Args:
            should_run (multiprocessing.Value (int)): should be 1 initially to run the subprocess, change it to 0 to stop the profiling
            started (multiprocessing.Event): notifies the process starter that profiling runs
            stopped (multiprocessing.Event): notifies the process starter that profiling ended
            queue (multiprocessing.Queue): handles the shared memory, call queue.get() to the firt put in (gpu_id, timestamp, power and used memory)
            interval (int): the interval in milliseconds the profiler should check nvidia-smi
        """

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
        nvidiasmi_process.wait()

    def __enter__(self):
        assert subprocess.getstatusoutput("nvidia-smi")[0] == 0, "Could not find nvidia-smi tool"
        self.process.start()
        self.profiling_started.wait()
        return self

    def __exit__(self, *args, **kwargs):
        self.should_profiling_run.value = 0
        self.profiling_stopped.wait()
        self.data = self.result_handler.get_all()
        self.process.join()
        self.process.terminate()
        self.data = self._data_post_process(self.data)

    def start(self):
        self.__enter__()

    def stop(self):
        self.__exit__()

    def _data_post_process(
        self, data: List[Tuple[int, datetime, float, float, str]]
    ) -> List[Tuple[int, datetime, float, float]]:
        """
        Post processes the data
        Args:
            data (List[Tuple[int, datetime, float, float, str]]): the data to post process
        Returns:
            List[Tuple[int, datetime, float, float]]: the post processed data with gpu_id, timestamp, power, memory and record_step
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
        creates a NvidiaProfiler object from a cache file.
        Note that only the data is loaded from the cache file, the profiler object is not
        started and the record_steps will be empty but
        available in the data directly.

        Args:
            cache_file (str): the path to the cache file

        Returns:
            NvidiaProfiler: the NvidiaProfiler object with the data from the cache file
        """
        prof = NvidiaProfiler(cache_file=cache_file, read_only=True)
        prof.data = prof.result_handler.get_all()
        prof.data = prof._data_post_process(prof.data)
        return prof

    def to_pandas(self) -> pd.DataFrame:
        """
        generates a pandas dataframe from the profiled data

        Returns:
            pandas DataFrame containing columns gpu_index, timestamp, power (in watts), memory (in MiB)
        """
        df: pd.DataFrame = pd.DataFrame(self.data, columns=["gpu_id", "timestamp", "power", "memory", "record_step"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        return df

    def get_profiled_gpus(self) -> List[int]:
        """
        Returns:
            a list of all gpu ids which got profiled
        """
        df: pd.DataFrame = self.to_pandas()
        return df["gpu_id"].unique().tolist()

    def get_total_energy(
        self,
        gpu_ids: List[int] | None = None,
        record_steps: List[str] | None = None,
        return_data: bool = False,
    ) -> float:
        """
        summes the power to get the total energy

        Args:
            gpu_id (List[int]): the ids of the gpu to calculate the total energy for (default = None, the first gpu id is used)
            record_steps (List[str]): records_steps to include in the summation (default = None, all record_steps will be included)
            return_data (bool): whether to return the raw data of the selection (default = False)
        Returns:
            the total energy recorded in watt seconds
            if return_data is True the list of measurements for each record_step in watt seconds is returned
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
        gpu_ids: List[int] | None = None,
        record_steps: List[str] | None = None,
        return_data: bool = False,
    ) -> float:
        """
        calculates the total profiling time in seconds for the given gpu_ids and record_steps

        Args:
            gpu_id (List[int]): the ids of the gpu to calculate the time for (default = None, the first gpu id is used)
            record_steps (List[str]): records_steps to include in the summation (default = None, all record_steps will be included)
            return_data (bool): whether to return the raw data of the selection (default = False)
        Returns:
            the total profiling time in seconds
        """
        if not self.data:
            return 0.0
        df: pd.DataFrame = self.to_pandas()
        df["record_step_id"] = df["record_step"].ne(df["record_step"].shift()).cumsum()
        gpu_ids = gpu_ids or [df["gpu_id"].unique()[0]]
        df = df[df["gpu_id"].isin(gpu_ids)]
        record_steps = record_steps or list(df["record_step"].unique())
        df = df[df["record_step"].isin(record_steps)]
        if return_data:
            return list(df.groupby("record_step_id")["timestamp"].apply(lambda x: (x.max() - x.min()).total_seconds()))
        return (df["timestamp"].max() - df["timestamp"].min()).total_seconds()

    def get_max_memory(
        self, gpu_id: int | None = None, record_steps: List[str] | None = None, return_data: bool = False
    ) -> float:
        """
        get average memory usage by gpu_id

        Args:
            gpu_id (int): the id of the gpu to calculate the avg memeory usage for (default = None, the first gpu id is used)

        Returns:
            the avgerage memory usage in MiB
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
        self, gpu_id: int | None = None, record_steps: List[str] | None = None, return_data: bool = False
    ) -> float:
        """
        get average memory usage by gpu_id

        Args:
            gpu_id (int): the id of the gpu to calculate the avg memeory usage for (default = None, the first gpu id is used)

        Returns:
            the avgerage memory usage in MiB
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
        Creates a plotly figure with the recorded time series data
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
