# `Profiler` Class

## Purpose

The `Profiler` class is a simple utility that lets you **mark specific points** in time during the execution of your code. These marks—called **steps**—can be used to measure and label sections of a program for later analysis.

You can use it to:

* Track the timing of specific operations (e.g., `"load_data"`, `"forward_pass"`)
* Annotate sections of code for profiling or debugging
* Align external profiling data (e.g., GPU usage) with specific program stages

---

## How It Works

You create a profiler instance, then call `record_step(name)` to log a moment in time with a label of your choice. Alternatively, you can use a context manager to automatically log entry and exit steps for a block of code.

All steps are stored as timestamped entries in `record_steps`.

---

## Attributes

### `record_steps: List[Tuple[datetime, str]]`

A list of all recorded steps. Each entry is a tuple:

```python
(timestamp, name)
```

* `timestamp`: When the step occurred (as a `datetime` object)
* `name`: The user-defined label for the step

---

## Methods

### `record_step(name: str) -> None`

Manually adds a labeled timestamp.

**Arguments**:

* `name` — A short string describing what’s happening in your program.

**Example**:

```python
profiler.record_step("load_data")
```

---

### `record_context(name: str)`

A context manager for automatically recording when a code block starts and ends.

When entering the block, it logs the step with the given name. When exiting, it logs another step with the name `"__other__"`.

**Arguments**:

* `name` — A label to describe what the code block is doing.

**Example**:

```python
with profiler.record_context("load_model"):
    model = load_my_model()
```

This will automatically record:

* `"load_model"` when the block starts
* `"__other__"` when the block ends

This can help pair resource usage or profiling data with specific code sections.

---

## Example

```python
from nnt.profiling.profiler import Profiler

profiler = Profiler()

profiler.record_step("start_training")

with profiler.record_context("forward_pass"):
    run_forward_pass()

profiler.record_step("end_training")

# Review steps
for timestamp, name in profiler.record_steps:
    print(f"{name} at {timestamp}")
```

---

## Notes

* All timestamps are based on `datetime.now()` and reflect wall-clock time.
* You can extend the class to:

  * Calculate durations between steps
  * Add custom labels on exit (instead of `"__other__"`)
  * Filter or export steps for further analysis

This class is often used as a foundation for higher-level profilers that gather hardware or system-level data, allowing events to be correlated with your program’s logical flow.

# `NvidiaProfiler` Class

## Purpose

`NvidiaProfiler` collects **GPU power usage** and **memory consumption** over time while your code runs. It’s useful when you want to:

* Understand the energy cost of a training run
* Compare memory usage across models
* Optimize resource consumption in GPU-intensive workloads

This class uses `nvidia-smi` under the hood to collect data at a fixed time interval and aligns it with your code’s logical steps (recorded using the base `Profiler`).

---

## How It Works

`NvidiaProfiler` runs in the background during your code and samples GPU metrics at regular intervals. You can start it manually or use it as a context manager. Steps can be recorded using `record_step()` to align GPU data with specific stages of your program.

You can optionally cache results to a CSV file for reuse or analysis later.

---

## Initialization

```python
NvidiaProfiler(
    interval=1000,
    cache_file=None,
    force_cache=False,
    gpu_clock_speed=None,
    read_only=False
)
```

### Parameters:

* `interval`: How often (in milliseconds) to collect GPU data.
* `cache_file`: If provided, saves profiling data to a CSV file.
* `force_cache`: If `True`, overwrites an existing cache file.
* `gpu_clock_speed`: Fixes the GPU clock speed to reduce measurement noise (requires sudo).
* `read_only`: If `True`, skips data collection and only loads from the cache file.

---

## Key Features

### Step Tracking

Inherits `record_step()` and `record_context()` from the base `Profiler`. Use these to mark points of interest in your program.

### GPU Sampling

Runs a separate process that:

* Periodically polls `nvidia-smi`
* Records GPU ID, timestamp, power usage (watts), and memory (MiB)

### Clock Speed Control

You can fix the GPU’s clock speed for consistent measurements:

```python
set_gpu_clock_speed(gpu_id, speed)
reset_gpu_clock_speed(gpu_id)
```

### Data Access

After profiling, use built-in methods to inspect or analyze the collected data.

---

## Data Utilities

```python
prof.get_total_energy()
prof.get_max_memory()
prof.get_mean_memory()
prof.to_pandas()
```

These help answer questions like:

* “How much energy did training use?”
* “What was peak GPU memory?”
* “Which step consumed the most power?”

---

## Visualization

You can generate a time series plot of power and memory over time:

```python
fig = prof.get_time_series_plot()
fig.show()
```

This highlights resource usage across steps and devices.

---

## Example

```python
with NvidiaProfiler(interval=500, cache_file="gpu_log.csv") as prof:
    prof.record_step("start_training")
    train_model()
    prof.record_step("end_training")

# Afterward
print(prof.get_total_energy())
df = prof.to_pandas()
```

---

## Notes

* Requires `nvidia-smi` to be available in your environment.
* Setting GPU clocks may require administrator access.
* Very short sampling intervals (<100ms) can lead to system delays or sampling issues.
* Reserved step name `"__unset__"` will raise an error if used.

---

# `TorchProfiler` Class

## Purpose

`TorchProfiler` wraps PyTorch’s native `torch.profiler.profile` and adds:

* Integration with logical steps (`record_step`)
* Easy export to `pandas`
* Summary statistics: FLOPs, memory, timing
* Step-level analysis

It’s ideal when you want to break down performance **by stage** (e.g., `forward`, `backward`, `optimizer`) while also using PyTorch's built-in profiling tools.

---

## How It Works

`TorchProfiler` behaves like PyTorch’s profiler but also uses `Profiler` to tag steps in your code. This makes it easier to group events like kernel calls, memory operations, and function times under human-readable labels.

Internally, it:

* Collects low-level event traces (CPU/GPU ops)
* Aligns events to your `record_step` timestamps
* Converts the data to `pandas` format for analysis

---

## Basic Use

```python
with TorchProfiler() as prof:
    prof.record_step("forward")
    run_forward()

    prof.record_step("backward")
    run_backward()

df = prof.to_pandas()
```

---

## Key Methods

### `record_step(name)`

Marks a point in the execution to later group profiling events by name.

---

### `to_pandas() -> pd.DataFrame`

Returns a DataFrame with all profiler events, including:

* `name`: Function or kernel name
* `device`: CPU or CUDA
* `self_cpu_time_total`, `cuda_time_total`
* `flops`, `memory`, `record_step`

You can filter or sort by device, step, or operation name.

---

### `summary()`

Groups all events by name and aggregates their metrics.

Useful to answer:

* Which operation consumed the most time?
* Where did most memory go?

---

### `totals()`

Provides a single-row summary with:

* Total FLOPs
* Total CPU/GPU time
* Total memory

---

### Per-Step Analysis

```python
prof.get_flops_by_step()
prof.get_time_by_step()
```

Helps you understand resource usage at each labeled step in your program.

---

## Step Alignment Details

Events are matched to steps based on time. Each event is assigned to the closest **previous** step you recorded. If no matching step is found, it’s skipped.

---

## Example

```python
with TorchProfiler() as prof:
    prof.record_step("start")
    train_one_batch()
    prof.record_step("end")

df = prof.to_pandas()
print(df.head())

summary = prof.summary()
totals = prof.totals()
```

---

## Notes

* Defaults to profiling both CPU and CUDA.
* Records FLOPs and memory usage by default.
* Assumes you use `record_step` to define step boundaries.
* Can be used alongside `NvidiaProfiler` for combined hardware + model analysis.