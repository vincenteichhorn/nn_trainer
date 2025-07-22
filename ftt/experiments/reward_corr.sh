BASE_OUTPUT_DIR="/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/test"
EPOCHS=1
LEARNING_RATE=5e-6
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
DATASET_NAME="arc_easy"
BASE_MODEL_NAME="meta-llama/Llama-3.2-1B"
REPETITIONS=5
VALIDATION="forward"

poetry run python3 -m ftt.experiments.reward_corr \
    --output_dir "$BASE_OUTPUT_DIR/reward_corr/" \
    --num_repetitions $REPETITIONS \
    --training_args.num_epochs $EPOCHS \
    --training_args.batch_size $TRAIN_BATCH_SIZE \
    --training_args.learning_rate $LEARNING_RATE \
    --base_model_name $BASE_MODEL_NAME \
    --dataset_name $DATASET_NAME \
    --dataset_validation $VALIDATION \
    --validation_batch_size $EVAL_BATCH_SIZE \
    --savings 0.5