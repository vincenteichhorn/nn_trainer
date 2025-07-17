cd /sc/home/vincent.eichhorn/nn_trainer
poetry install
nvidia-smi
which poetry
pwd
whoami
export $(grep -v '^#' .env | xargs)

# ps -u $USER | grep -iE 'python|cuda|torch' | awk '{print $1}' | xargs -r kill -9

# Default values
BASE_OUTPUT_DIR="/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt"
EPOCHS=10
LEARNING_RATE=5e-6
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=16
DATASET_NAME="glue_mrpc"
BASE_MODEL_NAME="meta-llama/Llama-3.2-1B"
REPETITIONS=5
VALIDATION="forward"

# Example command:
# bash run_all.sh --epochs 10 --learning_rate 5e-6 --train_batch_size 16 --eval_batch_size 16 --dataset_name "glue_mrpc" --validation "forward"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --output_dir) BASE_OUTPUT_DIR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
        --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --eval_batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --base_model_name) BASE_MODEL_NAME="$2"; shift 2 ;;
        --repetitions) REPETITIONS="$2"; shift 2 ;;
        --validation) VALIDATION="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# print the parsed arguments
echo "BASE_OUTPUT_DIR: $BASE_OUTPUT_DIR"
echo "EPOCHS: $EPOCHS"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "EVAL_BATCH_SIZE: $EVAL_BATCH_SIZE"
echo "DATASET_NAME: $DATASET_NAME"
echo "BASE_MODEL_NAME: $BASE_MODEL_NAME"
echo "REPETITIONS: $REPETITIONS"
echo "VALIDATION: $VALIDATION"

for NUM_TOP_LAYERS in 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1; do
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

for SAVINGS in 0.25 0.5 0.75; do
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

for RHO in 0.25 0.5 0.75; do
    echo "Running adaptive determinisitc approach with RHO=$RHO"
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
        --sub_approach "deterministic"
    echo "Completed adaptive deterministic approach with RHO=$RHO"
done

echo "Running adaptive stochastic approach with RHO=$RHO"
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
    --sub_approach "stochastic"
echo "Completed adaptive stochastic approach"

