#!/bin/bash -eux
#SBATCH --job-name=test
#SBATCH --account sci-herbrich
#SBATCH --nodelist=gx13
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=aisc
#SBATCH --cpus-per-task=8
#SBATCH --mem=24000
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=vincent.eichhorn@student.hpi.uni-potsdam.de
#SBATCH --output=/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/_jobs/job_test-%j.log

hostname
nvidia-smi