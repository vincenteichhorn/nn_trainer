from typing import Dict, List
import pandas as pd
from torch.profiler import profile
from torch.profiler import ProfilerActivity
from torch.autograd.profiler_util import FunctionEvent, EventList
from nnt.profiling.profiler import Profiler


class TorchProfiler(profile, Profiler):
    # pylint: disable=W0212:protected-access
    """subclass of (torch.profiler) profile"""

    def __init__(self, *args, **kwargs):
        defaults = {
            "with_flops": True,
            "profile_memory": True,
            "activities": [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        }
        kwargs = {**defaults, **kwargs}
        profile.__init__(self, *args, **kwargs)
        Profiler.__init__(self)

        self.numeric_columns = [
            "flops",
            "count",
            "self_device_time_total",
            "self_cpu_time_total",
            "device_time_total",
            "cpu_time_total",
            "self_device_memory_usage",
            "self_cpu_memory_usage",
            "device_memory_usage",
            "cpu_memory_usage",
        ]

    def _get_profiler_events(self) -> EventList[FunctionEvent]:
        """Ensures the profiler has function events and returns them."""
        assert self.profiler, "Profiling not stopped correctly"
        self.profiler._ensure_function_events()
        return self.profiler._function_events

    def to_pandas(self) -> pd.DataFrame:
        """converts profiler events to pandas dataframe

        Returns:
            pd.DataFrame() containing row-wise all events (calculations like aten::mm) with metrics
        """
        matched_events = self._get_profiler_events_by_record_step()

        rows = []
        for step, events in matched_events.items():
            for event in events:
                row = {col: getattr(event, col, None) for col in self.numeric_columns}
                row["name"] = getattr(event, "name", None)
                row["is_annotation"] = getattr(event, "is_user_annotation", None)
                row["device"] = getattr(event, "device_type", None).name
                row["record_step"] = step
                rows.append(row)

        df = pd.DataFrame(rows)
        df.loc[df["device"] == "CPU", "self_device_time_total"] = 0
        df.loc[df["device"] == "CPU", "device_time_total"] = 0
        df.loc[df["device"] == "CUDA", "self_cpu_time_total"] = 0
        df.loc[df["device"] == "CUDA", "cpu_time_total"] = 0
        df["self_cpu_time_total_percentage"] = df["self_cpu_time_total"] / df["self_cpu_time_total"].sum() * 100
        df["cpu_time_total_percentage"] = df["cpu_time_total"] / df["cpu_time_total"].sum() * 100
        return df

    def summary(self) -> pd.DataFrame:
        """generates a summary as a pandas dataframe

        Returns:
            pd.DataFrame() containing summed metrics for all event types (e.g. aten::mul, aten::mm)
        """
        df: pd.DataFrame = self.to_pandas()
        df = df[~df["is_annotation"]]
        df = df[["name"] + self.numeric_columns].groupby("name").sum()
        df = df.sort_values(by=["flops", "count"])
        return df

    def totals(self) -> pd.Series:
        """sums all metrics for the whole profiling

        Returns:
            pd.Series containing the total sums with keys
        """
        df: pd.DataFrame = self.to_pandas()
        df = df[~df["is_annotation"]]
        df = df[self.numeric_columns].sum(axis=0)
        return df

    def get_total_time(self, device: str = "CUDA") -> float:
        """computes the total device time from the profiler's events without converting to pandas df
        Args:
            device (str): the device to calculate the total time for (default = CUDA), options: CPU, CUDA
        Returns:
            float: the total device time in us
        """
        assert device in ["CPU", "CUDA"], "device must be either CPU or CUDA"
        events = self._get_profiler_events()
        time_field = "self_cpu_time_total" if device == "CPU" else "self_device_time_total"
        total_time = sum(
            getattr(event, time_field, 0.0)
            for event in events
            if event.device_type.name == device and not event.is_user_annotation
        )
        return total_time

    def get_total_flops(self) -> float:
        """computes the total FLOPs from the profiler's events without converting to pandas df

        Returns:
            float: The sum of FLOPs for all events.
        """
        events = self._get_profiler_events()
        total_flops = sum(getattr(event, "flops", 0.0) for event in events)
        return int(total_flops)

    def _get_profiler_events_by_record_step(self) -> Dict[str, List[FunctionEvent]]:
        """returns the profiler events grouped by record steps
        Returns:
            Dict[str, List[FunctionEvent]]: the profiler events grouped by record steps
        """
        events = self._get_profiler_events()
        matched_events = {step: [] for _, step in self.record_steps}
        base_timestamp = self.profiler.kineto_results.trace_start_ns() * 1e-3

        for event in events:
            event_start = event.time_range.start
            event_ts = base_timestamp + event_start

            diffs = [(event_ts - ts.timestamp() * 1e6, name) for ts, name in self.record_steps]
            pos_diffs = [(diff, name) for diff, name in diffs if diff >= 0]
            matched = min(pos_diffs, key=lambda x: x[0])
            matched_events[matched[1]].append(event)

        return matched_events

    def get_flops_by_step(self) -> pd.DataFrame:
        """computes the FLOPs for each step in the record_steps list based on time_range

        Returns:
            pd.DataFrame: A DataFrame with the FLOPs for each step.
        """
        matched_events = self._get_profiler_events_by_record_step()
        flops_by_step = {name: sum(event.flops for event in events) for name, events in matched_events.items()}
        df = pd.DataFrame.from_dict(flops_by_step, orient="index", columns=["flops"])

        return df

    def get_time_by_step(self) -> pd.DataFrame:
        """computes the time for each step in the record_steps list based on time_range
        Returns:
            pd.DataFrame: A DataFrame with the time is us for each step.
        """
        matched_events = self._get_profiler_events_by_record_step()
        time_by_step = {}
        for step, events in matched_events.items():
            gpu_time, cpu_time = 0.0, 0.0
            for event in events:
                if event.is_user_annotation:
                    continue
                if event.device_type.name == "CUDA":
                    gpu_time += getattr(event, "self_device_time_total", 0.0)
                elif event.device_type.name == "CPU":
                    cpu_time += getattr(event, "self_cpu_time_total", 0.0)
            time_by_step[step] = (cpu_time, gpu_time)
        df = pd.DataFrame.from_dict(time_by_step, orient="index", columns=["cpu_time", "gpu_time"])
        return df
