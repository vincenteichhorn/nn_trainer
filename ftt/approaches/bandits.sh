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
