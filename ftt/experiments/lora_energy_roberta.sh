#!/bin/bash -eux
#SBATCH --job-name=lora_energy
#SBATCH --account sci-herbrich
#SBATCH --constraint GPU_MEM:40GB
#SBATCH --partition gpu
#SBATCH --gpus 1
#SBATCH --cpus-per-task 32
#SBATCH --mem 24000
#SBATCH --time 24:00:00
#SBATCH --container-image ghcr.io/hpi-artificial-intelligence-teaching/lora-bp-base-cuda-v2:cuda-12.2
#SBATCH --container-mounts /sc:/sc
#SBATCH --container-name lora-ba-dependencies
#SBATCH --container-writable
#SBATCH --container-remap-root
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user vincent.eichhorn@student.hpi.uni-potsdam.de
#SBATCH --output /sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/ba/jobs/job_lora_energy-%j.log

cd /sc/home/vincent.eichhorn/nn_trainer
which poetry
pwd
whoami

poetry run python3 -m ftt.experiments.lora_energy --out_file out/energy_lora_a100_roberta.csv --model_name FacebookAI/roberta-large


# bash /sc/home/vincent.eichhorn/nn_trainer/ftt/scripts/cluster_fuck.sh \
#     lora_b_rob \
#     /sc/home/vincent.eichhorn/nn_trainer/ftt/experiments/lora_energy_roberta.sh \
#     1-00:00:00 \
#     gx28 \
#     "" \
#     gpu

# bash /sc/home/vincent.eichhorn/nn_trainer/ftt/scripts/cluster_fuck.sh \
#     lora_b_llam \
#     /sc/home/vincent.eichhorn/nn_trainer/ftt/experiments/lora_energy_llama.sh \
#     1-00:00:00 \
#     gx28 \
#     "" \
#     gpu