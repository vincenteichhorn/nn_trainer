cd /sc/home/vincent.eichhorn/nn_trainer
nvidia-smi
which poetry
pwd
whoami

export TOKENIZERS_PARALLELISM=false

BASE_OUTPUT_DIR="/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt"
EPOCHS=1
LEARNING_RATE=1e-5
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
DATASET_NAME="glue_sst2"
BASE_MODEL_NAME="meta-llama/Llama-3.2-1B"
REPETITIONS=5
VALIDATION="forward"

for NUM_TOP_LAYERS in {1..16}; do
    echo "Running LoRA with NUM_TOP_LAYERS=$NUM_TOP_LAYERS"
    poetry run python3 -m ftt.approaches.static \
        --output_dir "$BASE_OUTPUT_DIR/out/static/" \
        --num_repetitions $REPETITIONS \
        --num_top_layers $NUM_TOP_LAYERS \
        --training_args.num_epochs $EPOCHS \
        --training_args.batch_size $TRAIN_BATCH_SIZE \
        --training_args.learning_rate $LEARNING_RATE \
        --base_model_name $BASE_MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --dataset_validation $VALIDATION \
        --validation_batch_size $EVAL_BATCH_SIZE
    echo "Completed LoRA with NUM_TOP_LAYERS=$NUM_TOP_LAYERS"
done

for SAVINGS in 0.01 0.25 0.5 0.75 0.99; do
    echo "Running stochastic approach with SAVINGS=$SAVINGS"
    poetry run python3 -m ftt.approaches.stochastic \
        --output_dir "$BASE_OUTPUT_DIR/out/stochastic/" \
        --num_repetitions $REPETITIONS \
        --training_args.num_epochs $EPOCHS \
        --training_args.batch_size $TRAIN_BATCH_SIZE \
        --training_args.learning_rate $LEARNING_RATE \
        --base_model_name $BASE_MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --dataset_validation $VALIDATION \
        --validation_batch_size $EVAL_BATCH_SIZE \
        --savings $SAVINGS
    echo "Completed stochastic approach with SAVINGS=$SAVINGS"
done

for RHO in 0.25 0.5 0.75; do
    for SUB_APPROACH in "deterministic" "stochastic"; do
        echo "Running adaptive approach with RHO=$RHO and SUB_APPROACH=$SUB_APPROACH"
        poetry run python3 -m ftt.approaches.adaptive \
            --output_dir "$BASE_OUTPUT_DIR/out/adaptive/" \
            --num_repetitions $REPETITIONS \
            --training_args.num_epochs $EPOCHS \
            --training_args.batch_size $TRAIN_BATCH_SIZE \
            --training_args.learning_rate $LEARNING_RATE \
            --base_model_name $BASE_MODEL_NAME \
            --dataset_name $DATASET_NAME \
            --dataset_validation $VALIDATION \
            --validation_batch_size $EVAL_BATCH_SIZE \
            --rho $RHO \
            --sub_approach "$SUB_APPROACH"
        echo "Completed adaptive approach with RHO=$RHO and SUB_APPROACH=$SUB_APPROACH"
    done
done

for RHO in 0.25 0.5 0.75; do
    echo "Running green trainer with RHO=$RHO"
    poetry run python3 -m ftt.approaches.green_trainer \
        --output_dir "$BASE_OUTPUT_DIR/out/green_trainer/" \
        --num_repetitions $REPETITIONS \
        --training_args.num_epochs $EPOCHS \
        --training_args.batch_size $TRAIN_BATCH_SIZE \
        --training_args.learning_rate $LEARNING_RATE \
        --base_model_name $BASE_MODEL_NAME \
        --dataset_name $DATASET_NAME \
        --dataset_validation $VALIDATION \
        --validation_batch_size $EVAL_BATCH_SIZE \
        --rho $RHO
    echo "Completed green trainer with RHO=$RHO"
done
