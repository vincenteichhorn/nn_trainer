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


for DELTA in 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
    for GAMMA in 0.6 0.65 0.7 0.75 0.8 0.865 0.9 0.95 0.99; do
        for LAMBDA in 0.001 0.01 0.05 0.1 0.5 1.0 10.0; do
            for SIGMA in 0.01 0.05 0.1 0.2 0.5 1.0; do
                echo "Running Bandit with GAMMA: $GAMMA, DELTA: $DELTA, LAMBDA: $LAMBDA, SIGMA: $SIGMA"
                poetry run python3 -m ftt.approaches.bandits \
                    --output_dir "$BASE_OUTPUT_DIR/out/bandits/" \
                    --num_repetitions "$REPETITIONS" \
                    --training_args.num_epochs "$EPOCHS" \
                    --training_args.batch_size "$TRAIN_BATCH_SIZE" \
                    --training_args.learning_rate "$LEARNING_RATE" \
                    --base_model_name "$BASE_MODEL_NAME" \
                    --dataset_name "$DATASET_NAME" \
                    --dataset_validation "$VALIDATION" \
                    --validation_batch_size "$EVAL_BATCH_SIZE" \
                    --bandit dUCB \
                    --gamma $GAMMA \
                    --lmda $LAMBDA \
                    --delta $DELTA \
                    --sigma $SIGMA
            done
        done
    done
done
