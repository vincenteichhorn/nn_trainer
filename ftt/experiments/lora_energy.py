import argparse
from itertools import product
import random

from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import pandas as pd
from multiprocessing import Queue

from ftt.lora import LoRAModel, load_model
from ftt.lora_strategies import LoRAUniformStrategy
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM
from nnt.datasets.dataset import DataSplit
from nnt.profiling.multiprocessing_util import start_seprate_process
from nnt.profiling.nvidia_profiler import NvidiaProfiler
from nnt.profiling.torch_profiler import TorchProfiler
from nnt.util.monitor import Monitor


def generate_random_dataset(num_samples, batch_size, input_length, tokenizer, verbose=False):

    tokenizer_vocab = list(tokenizer.get_vocab().keys())[:3]
    input_ids = []
    for _ in tqdm(
        range(num_samples * batch_size),
        desc="Generating random dataset",
        disable=not verbose,
    ):
        random_tokens = [random.choice(tokenizer_vocab) for _ in range(input_length)]
        input_ids_sample = tokenizer.convert_tokens_to_ids(random_tokens)
        input_ids_sample = torch.Tensor(input_ids_sample).to(torch.int64)
        input_ids.append(input_ids_sample)

    attention_mask = torch.ones_like(input_ids[0]).to(torch.int64)
    attention_masks = [attention_mask] * num_samples * batch_size
    dataset = DataSplit.from_iterable(
        [{"input_ids": inp, "labels": inp, "attention_mask": mask} for inp, mask in zip(input_ids, attention_masks)]
    )
    return dataset


def build_lora_model(model_name: str, rank: int):
    base_model, tokenizer = load_model(model_name)
    lora_model = LoRAModel(
        base_model,
        LoRAUniformStrategy(rank=rank, dropout=0.1, alpha=32),
        model_name,
    )
    return (model_name, model_name), lora_model, tokenizer


def build_fft_model(model_name):
    base_model, tokenizer = load_model(model_name)
    return (model_name, model_name), base_model, tokenizer


def energy_eval_lora(model_name, rank, input_length, batch_size, num_samples, num_warumup_samples):
    try:
        if rank == -1:
            _, model, tokenizer = build_fft_model(model_name)
        else:
            _, model, tokenizer = build_lora_model(model_name, rank)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        random_datasplit = generate_random_dataset(
            num_samples + num_warumup_samples,
            batch_size,
            input_length,
            tokenizer,
            verbose=False,
        )
        dataloader = DataLoader(
            random_datasplit,
            batch_size=batch_size,
            collate_fn=DataCollatorForCausalLM(tokenizer),
        )
        with NvidiaProfiler() as prof:
            for i, batch in Monitor().tqdm(enumerate(dataloader), desc="Evaluating energy metrics"):

                if i < num_warumup_samples:
                    continue
                batch = {k: v.to(model.device) for k, v in batch.items()}

                with prof.record_context("forward"):
                    outputs = model(**batch)
                with prof.record_context("backward"):
                    outputs.loss.backward()
                with prof.record_context("optimizer"):
                    optimizer.step()
        energy_metrics = dict()
        energy_metrics["forward_joules"] = prof.get_total_energy(record_steps=["forward"])
        energy_metrics["forward_backward_joules"] = prof.get_total_energy(record_steps=["forward", "backward"])
        energy_metrics["forward_backward_optimizer_joules"] = prof.get_total_energy(
            record_steps=["forward", "backward", "optimizer"]
        )
        energy_metrics["forward_time"] = prof.get_total_time(record_steps=["forward"])
        energy_metrics["forward_backward_time"] = prof.get_total_time(record_steps=["forward", "backward"])
        energy_metrics["forward_backward_optimizer_time"] = prof.get_total_time(
            record_steps=["forward", "backward", "optimizer"]
        )
        energy_metrics["forward_memory"] = prof.get_max_memory(record_steps=["forward"])
        energy_metrics["forward_backward_memory"] = prof.get_max_memory(record_steps=["forward", "backward"])
        energy_metrics["forward_backward_optimizer_memory"] = prof.get_max_memory(
            record_steps=["forward", "backward", "optimizer"]
        )
        single_batch = next(iter(dataloader))
        single_batch = {k: v.to(model.device) for k, v in single_batch.items()}
        with TorchProfiler() as profiler:
            with profiler.record_context("forward"):
                model(**single_batch)
            with profiler.record_context("backward"):
                outputs.loss.backward()
            with profiler.record_context("optimizer"):
                optimizer.step()
        energy_metrics["forward_flops"] = profiler.get_total_flops(record_steps=["forward"])
        energy_metrics["forward_backward_flops"] = profiler.get_total_flops(record_steps=["forward", "backward"])
        energy_metrics["forward_backward_optimizer_flops"] = profiler.get_total_flops(
            record_steps=["forward", "backward", "optimizer"]
        )

        energy_metrics["num_trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return energy_metrics
    except Exception as e:
        print(f"Error evaluating model {model_name} with rank {rank}: {e}")
        return None


def energy_eval_lora_process_wrapper(result_queue: Queue, *args, **kwargs):
    energy_metrics = energy_eval_lora(*args, **kwargs)
    result_queue.put(energy_metrics)


def energy_eval_lora_wrapped(model_name, rank, input_length, batch_size, num_samples, num_warumup_samples):
    energy_metrics = start_seprate_process(
        energy_eval_lora_process_wrapper,
        [model_name, rank, input_length, batch_size, num_samples, num_warumup_samples],
    )
    return energy_metrics


if __name__ == "__main__":

    out_file = "out/energy_lora_a100.csv"
    print(f"Output file: {out_file}")

    arg_parser = argparse.ArgumentParser(description="Energy evaluation for LoRA models")
    arg_parser.add_argument("--out_file", type=str, default=out_file, help="Output file for results")
    arg_parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B", help="Model name to evaluate")
    args = arg_parser.parse_args()
    out_file = args.out_file
    models = [args.model]
    input_lengths = [256]
    batch_sizes = [20, 16, 12, 8, 4]
    ranks = [-1, 1, 2, 4, 8, 16, 32] + list(range(64, 512 + 64, 64))
    repetitions = list(range(10))
    grid = list(product(repetitions, batch_sizes, models, input_lengths, ranks))

    num_samples = 100
    num_warumup_samples = 10

    df = pd.read_csv(out_file) if pd.io.common.file_exists(out_file) else pd.DataFrame()
    for i, (rep, batch_size, model_name, input_length, rank) in enumerate(grid):
        print(f"Processing {i + 1}/{len(grid)}: {model_name}, {input_length}, {batch_size}, {rank}, {rep}")
        if not df.empty:
            if (
                df[
                    (df["model_name"] == model_name)
                    & (df["input_length"] == input_length)
                    & (df["batch_size"] == batch_size)
                    & (df["rank"] == rank)
                    & (df["repetition"] == rep)
                ].shape[0]
                > 0
            ):
                print(f"Skipping already processed: {model_name}, {input_length}, {batch_size}, {rank}")
                continue
        metrics = energy_eval_lora_wrapped(
            model_name,
            rank,
            input_length,
            batch_size,
            num_samples,
            num_warumup_samples,
        )
        if metrics is None:
            print(f"Failed to evaluate model {model_name} with rank {rank}. Skipping...")
            continue
        sub_df = pd.DataFrame(metrics, index=[0])
        sub_df["model_name"] = model_name
        sub_df["input_length"] = input_length
        sub_df["batch_size"] = batch_size
        sub_df["repetition"] = rep
        sub_df["rank"] = rank
        df = pd.concat([df, sub_df], ignore_index=True)
        df.to_csv(out_file, index=False)
        torch.cuda.empty_cache()
