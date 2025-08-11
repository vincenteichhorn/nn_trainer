# Setup
The environment and dependencies are managed using Poetry. To set up the environment, run:
```bash
poetry install
```

To activate the environment, use:
```bash
source $(poetry env info --path)/bin/activate
```


# Experiment Documentation Thesis
## Experiments for Section "LoRA's Energy Saving Potential"
The experiment can be run with the following command:
```bash
python -m ftt.experiments.lora_energy --model_name <model_name> --out_file out/energy_lora.csv 
```
The resulting CSV file will contain the energy consumption metrics for LoRA and FFT in different configurations. The resulting CSV can be parsed with `ftt/results/plotting/lora_energy_plots.py` to generate plots using streamlit and matplotlib.
```bash
streamlit run ./ftt/results/plotting/lora_energy_plots.py --server.fileWatcherType=poll
```

## Experiments for Section "Towards Energy Efficient Fine-Tuning"
All approaches can be found in the `fft/approaches` directory. All approaches can be run with the following command:
```bash
bash run_all.sh --epochs 10 --learning_rate 5e-6 --train_batch_size 16 --eval_batch_size 16 --dataset_name "glue_mrpc" --validation "forward" --out_dir "results/"
```
All commands for all datasets can be found in `ftt/jobs/run_all`

The code will create the following folder structure:
```
results/
├── static
├── ├── glue_mrpc
├── ├── <other datasets>
├── stochastic
├── ├── glue_mrpc
├── ├── <other datasets>
├── green_trainer
├── ├── glue_mrpc
├── ├── <other datasets>
```
### Edge Case "glue_mnli_mismatched" and "glue_mnli_matched"
We run the experiments with the --dataset_name "glue_mnli_matched" and post-validate for the "glue_mnli_mismatched" dataset. 
The post-validation can be done with the following command after the results folders of each `glue_mnli_matched` is copied `glue_mnli_mismatched`:
```bash
poetry run python3 -m ftt.results.reeval \
    --exp_dir static/\
    --dataset glue_mnli_mismatched \
    --reeval_rule "lambda x: 'mnli_mismatched' in x" \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --base_model meta-llama/Llama-3.2-1B

poetry run python3 -m ftt.results.reeval \
    --exp_dir stochastic/\
    --dataset glue_mnli_mismatched \
    --reeval_rule "lambda x: 'mnli_mismatched' in x" \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --base_model meta-llama/Llama-3.2-1B

poetry run python3 -m ftt.results.reeval \
    --exp_dir green_trainer/\
    --dataset glue_mnli_mismatched \
    --reeval_rule "lambda x: 'mnli_mismatched' in x" \
    --tokenizer_name meta-llama/Llama-3.2-1B \
    --base_model meta-llama/Llama-3.2-1B
```

### Aggregation of Results
Aggregated results of the thesis are stored  in the `out` directory.
The results can be aggregated using the following commands:
```bash
python3 -m ftt.results.aggregate \
    --exp_dir out/static/ \
    --parse_rules '{
        "nlayer": "lambda x: int(x.split(\"-\")[-1])",
        "repid": "lambda x: int(x.split(\"-\")[0])"
    }'

python3 -m ftt.results.aggregate \
    --exp_dir out/stochastic \
    --parse_rules '{"savings": "lambda x: float(x.split(\"-\")[-3])", "repid": "lambda x: int(x.split(\"-\")[0])"}'

python3 -m ftt.results.aggregate \
    --exp_dir out/green_trainer \
    --parse_rules '{"rho": "lambda x: float(x.split(\"-\")[-3])", "repid": "lambda x: int(x.split(\"-\")[0])"}'

```
These scripts will create a `results.csv` file in each of the directories (`static/`, `stochastic/`). The results can be visualized using the following commands:
```bash
streamlit run ./ftt/results/plotting/pareto_front.py --server.fileWatcherType=poll
streamlit run ./ftt/results/plotting/results_tables.py --server.fileWatcherType=poll
```
---
Copyright (C) 2025  Vincent Eichhorn
