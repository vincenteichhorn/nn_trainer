cd /sc/home/vincent.eichhorn/nn_trainer
nvidia-smi
which poetry
pwd
whoami

export TOKENIZERS_PARALLELISM=false

BASE_OUTPUT_DIR="/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt"
EPOCHS=10
LEARNING_RATE=5e-6
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
DATASET_NAME="glue_mrpc"
BASE_MODEL_NAME="meta-llama/Llama-3.2-1B"
REPETITIONS=1
VALIDATION="forward"

# remove all "donefiles"
find "$BASE_OUTPUT_DIR/out/bandits/" -type f -name "donefile" -exec rm {} \;

poetry run python3 -m ftt.approaches.bandits \
    --output_dir "$BASE_OUTPUT_DIR/out/bandits/" \
    --num_repetitions $REPETITIONS \
    --training_args.num_epochs $EPOCHS \
    --training_args.batch_size $TRAIN_BATCH_SIZE \
    --training_args.learning_rate $LEARNING_RATE \
    --base_model_name $BASE_MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_validation $VALIDATION \
    --validation_batch_size $EVAL_BATCH_SIZE \

