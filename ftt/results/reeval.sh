cd /sc/home/vincent.eichhorn/nn_trainer
nvidia-smi
which poetry
pwd
whoami

export TOKENIZERS_PARALLELISM=false

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


# bash /sc/home/vincent.eichhorn/nn_trainer/ftt/scripts/cluster_fuck.sh \
# reeval_mnli \
# /sc/home/vincent.eichhorn/nn_trainer/ftt/results/reeval.sh \
# 1-00:00:00 \
# gx01,gx03,gx04,gx05,gx25,gx27,gx28 \
# "" \
# gpu
