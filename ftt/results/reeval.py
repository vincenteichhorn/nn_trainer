import io
import os
import pandas as pd
import argparse
from transformers import AutoTokenizer

from ftt.datasets import get_dataset
from ftt.lora import load_model
from ftt.results.aggregate import mask_brackets_in_csv
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validation_metrics.generation_metrics import BleuScore, MeteorScore, NistScore, RougeScore
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.generation_validator import GenerationValidator
from nnt.validators.validator import ValidationArguments

basemodel_eval_cache = None


def eval_model(
    dataset,
    model_name="meta-llama/Llama-3.2-1B",
    tokenizer_name="meta-llama/Llama-3.2-1B",
    dataset_validation="forward",
):

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(model_name, tokenizer_name)
    model, tokenizer = load_model(model_name, tokenizer_name)

    validator = None
    validation_arguments = ValidationArguments(batch_size=16, data_collator=DataCollatorForCausalLM(tokenizer))
    if dataset_validation == "forward":
        validator = ForwardValidator(
            model=model,
            validation_args=validation_arguments,
            validation_data=dataset["validation"],
            metrics=[
                OneHotClassificationMetrics(
                    num_classes=len(dataset.get_task_classes()),
                    classes=tokenizer.convert_tokens_to_ids(dataset.get_task_classes()),
                    targets_key="labels",
                    logits_key="logits",
                )
            ],
        )
    elif dataset_validation == "generation":
        validator = GenerationValidator(
            model=model,
            tokenizer=tokenizer,
            validation_args=validation_arguments,
            validation_data=dataset["generation"],
            max_length=128,
            metrics=[
                BleuScore(target_key="output"),
                NistScore(target_key="output"),
                RougeScore(target_key="output"),
                MeteorScore(target_key="output"),
            ],
        )
    results = validator.validate()
    return results


def re_evaluate(
    training_dir: str,
    dataset,
    base_model="meta-llama/Llama-3.2-1B",
    tokenizer_name="meta-llama/Llama-3.2-1B",
    force: bool = False,
):

    re_validation_log_path = os.path.join(training_dir, "re_validation_log.csv")
    if not force and os.path.exists(re_validation_log_path):
        print(f"Re-evaluation log already exists in {training_dir}, skipping...")
        return

    validation_log_path = os.path.join(training_dir, "validation_log.csv")
    eval_log = pd.read_csv(io.StringIO(mask_brackets_in_csv(validation_log_path)), quotechar='"')

    global basemodel_eval_cache
    if basemodel_eval_cache is None:
        print("Evaluating base model...")
        basemodel_eval_cache = eval_model(dataset, model_name=base_model, tokenizer_name=tokenizer_name)

    eval_results = [basemodel_eval_cache]

    end_model_results = eval_model(
        dataset,
        model_name=os.path.join(training_dir, "model/"),
        tokenizer_name=tokenizer_name,
    )
    eval_results.append(end_model_results)

    flattened_results = []
    for result in eval_results:
        flat_result = {}
        for key, value in result.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_result[f"{key}_{sub_key}"] = sub_value
            else:
                flat_result[key] = value
        flattened_results.append(flat_result)

    reeval_log = pd.DataFrame(flattened_results)
    for col in reeval_log.columns:
        eval_log[col] = reeval_log[col]
    eval_log.to_csv(os.path.join(training_dir, "re_validation_log.csv"), index=False)


if __name__ == "__main__":

    # python3 -m ftt.results.reeval --exp_dir green_trainer/ --dataset glue_mnli_mismatched --reeval_rule "lambda x: 'mnli_mismatched' in x" --tokenizer_name meta-llama/Llama-3.2-1B --base_model meta-llama/Llama-3.2-1B

    base_dir = "/sc/projects/sci-herbrich/chair/lora-bp/vincent.eichhorn/nnt/out"
    parser = argparse.ArgumentParser(description="Re-evaluate checkpoints for a given experiment directory and rule.")
    parser.add_argument("--exp_dir", type=str, default="static/", help="Experiment directory")
    parser.add_argument("--dataset", type=str, default="glue_mrpc", help="GLUE task name (default: glue_mrpc)")
    parser.add_argument(
        "--reeval_rule",
        type=str,
        required=True,
        help="Lambda rule to filter training directories for re-evaluation (e.g., 'lambda x: \"mrpc\" in x')",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Tokenizer name to use (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Base model name to use for evaluation (default: meta-llama/Llama-3.2-1B)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if re_validation_log.csv already exists",
    )

    args = parser.parse_args()

    exp_dir = os.path.join(base_dir, args.exp_dir)
    reeval_rule = eval(args.reeval_rule)

    dataset = get_dataset(args.dataset)

    tmp_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print("Preparing dataset...")
    dataset.prepare(tmp_tokenizer)

    print(f"Searching for training directories in: {exp_dir}")
    training_dirs = [os.path.join(root, d) for root, dirs, _ in os.walk(exp_dir) for d in dirs]
    training_dirs = [d for d in training_dirs if reeval_rule(d) and os.path.exists(os.path.join(d, "donefile"))]
    print(f"Found {len(training_dirs)} training directories matching the rule.")
    print(f"Force: {args.force}")

    for i, training_dir in enumerate(training_dirs):
        print(f"Re-evaluating training directory: {training_dir}, {i + 1}/{len(training_dirs)}")
        re_evaluate(
            training_dir,
            dataset,
            base_model=args.base_model,
            tokenizer_name=args.tokenizer_name,
            force=args.force,
        )
