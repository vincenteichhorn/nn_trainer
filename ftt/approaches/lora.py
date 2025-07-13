from dataclasses import dataclass
import os
from typing import List, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.datasets import get_dataset
from ftt.lora import LoRAModel, load_model
from ftt.lora_strategies import LoRAPartialStrategy, LoRAUniformStrategy
from nnt.callbacks.energy_callback import EnergyCallback
from nnt.callbacks.flops_budget_callback import FLOPsBudgetControllCallback
from nnt.callbacks.logging_callback import LoggingCallback
from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.callbacks.validator_callback import ValidatorCallback
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM
from nnt.experiment import Experiment, ExperimentConfig, experiment_config_cli
from nnt.trainer import Trainer
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validation_metrics.generation_metrics import BleuScore, MeteorScore, NistScore, RougeScore
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.generation_validator import GenerationValidator
from nnt.validators.validator import ValidationArguments


class RepetitiveExperimentConfig(ExperimentConfig):

    num_repetitions: int = 5
    watch_done_file: bool = True


class LoRAExperimentConfig(RepetitiveExperimentConfig):
    """
    Configuration for the LoRA experiment.
    This class extends ExperimentConfig to include specific parameters for LoRA experiments.
    """

    base_model_name: str
    tokenizer_name: str
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    dataset_name: str = "glue_mrpc"
    dataset_validation: Literal["forward", "generation", ""] = ""
    generation_reference_column: str = "output"


class LoRAExperiment(Experiment):

    def prepare(self):
        """
        Prepare the static approach by loading the necessary components.
        This method should be implemented by subclasses.
        """
        model, tokenizer = self.load_model_and_tokenizer()
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = get_dataset(self.config.dataset_name)
        self.dataset.prepare(self.tokenizer)

        if self.config.training_args.data_collator is None:
            self.config.training_args.data_collator = DataCollatorForCausalLM(self.tokenizer)

        if self.config.training_args.model_save_function is None:
            self.config.training_args.model_save_function = lambda model, path: model.save_pretrained(path)

    def load_additional_callbacks(self) -> List[TrainerCallback]:
        """
        Load additional callbacks if needed.
        This method can be overridden by subclasses to add specific callbacks.
        """
        return []

    def load_model_and_tokenizer(self) -> Tuple[Module, PreTrainedTokenizer]:

        base_model, tokenizer = load_model(self.config.base_model_name, self.config.tokenizer_name)
        model = LoRAModel(
            base_model,
            LoRAUniformStrategy(
                rank=self.config.lora_rank,
                dropout=self.config.lora_dropout,
                alpha=self.config.lora_alpha,
            ),
            self.config.base_model_name,
        )
        return model, tokenizer

    def run(self):
        """
        Run the static approach using the provided configuration.
        This method should be implemented by subclasses.
        """

        for repid in range(self.config.num_repetitions):
            output_dir = f"{self.config.output_dir}/{self.config.dataset_name}/{repid}"
            done_file = f"{output_dir}/donefile"
            if self.config.watch_done_file and os.path.exists(done_file):
                print(f"Skipping repetition {repid} as done file exists: {done_file}")
                continue

            validator = None
            if self.config.dataset_validation == "forward":
                validator = ForwardValidator(
                    model=self.model,
                    validation_args=ValidationArguments(
                        batch_size=32, data_collator=DataCollatorForCausalLM(self.tokenizer)
                    ),
                    validation_data=self.dataset["validation"],
                    metrics=[
                        OneHotClassificationMetrics(
                            num_classes=len(self.dataset.get_task_classes()),
                            classes=self.tokenizer.convert_tokens_to_ids(self.dataset.get_task_classes()),
                            targets_key="labels",
                            logits_key="logits",
                        )
                    ],
                )
            elif self.config.dataset_validation == "generation":
                validator = GenerationValidator(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    validation_args=ValidationArguments(
                        batch_size=32, data_collator=DataCollatorForCausalLM(self.tokenizer)
                    ),
                    validation_data=self.dataset["generation"],
                    metrics=[
                        BleuScore(target_key=self.config.generation_reference_column),
                        NistScore(target_key=self.config.generation_reference_column),
                        RougeScore(target_key=self.config.generation_reference_column),
                        MeteorScore(target_key=self.config.generation_reference_column),
                    ],
                )

            trainer = Trainer(
                output_dir=output_dir,
                model=self.model,
                training_args=self.config.training_args,
                train_data=self.dataset["train"],
                callbacks=[
                    LoggingCallback(output_dir=output_dir),
                    ValidatorCallback(
                        output_dir=output_dir,
                        log_file="validation_log.csv",
                        validator=validator,
                    ),
                    FLOPsBudgetControllCallback(output_dir=output_dir, budget=1e9, should_stop_training=False),
                    *self.load_additional_callbacks(),
                    EnergyCallback(output_dir=output_dir, nvidia_query_interval=10),
                ],
            )
            trainer.run()
            if self.config.watch_done_file:
                with open(done_file, "w") as f:
                    f.write("done")
            print(f"Repetition {repid} completed. Output saved to {output_dir}.")


if __name__ == "__main__":

    config = experiment_config_cli(LoRAExperimentConfig)
    print(config)
    experiment = LoRAExperiment(config)
    experiment.run()

"""
options:
  -h, --help            show this help message and exit
  --name NAME
  --output_dir OUTPUT_DIR
  --training_args.num_epochs TRAINING_ARGS.NUM_EPOCHS
  --training_args.batch_size TRAINING_ARGS.BATCH_SIZE
  --training_args.data_collator TRAINING_ARGS.DATA_COLLATOR
  --training_args.learning_rate TRAINING_ARGS.LEARNING_RATE
  --training_args.weight_decay TRAINING_ARGS.WEIGHT_DECAY
  --training_args.monitor_strategy TRAINING_ARGS.MONITOR_STRATEGY
  --training_args.monitor_every TRAINING_ARGS.MONITOR_EVERY
  --training_args.checkpoint_strategy TRAINING_ARGS.CHECKPOINT_STRATEGY
  --training_args.checkpoint_every TRAINING_ARGS.CHECKPOINT_EVERY
  --training_args.model_save_function TRAINING_ARGS.MODEL_SAVE_FUNCTION
  --description DESCRIPTION
  --num_repetitions NUM_REPETITIONS
  --watch_done_file WATCH_DONE_FILE
  --base_model_name BASE_MODEL_NAME
  --tokenizer_name TOKENIZER_NAME
  --lora_rank LORA_RANK
  --lora_alpha LORA_ALPHA
  --lora_dropout LORA_DROPOUT
  --dataset_name DATASET_NAME
  --dataset_validation DATASET_VALIDATION
  --generation_reference_column GENERATION_REFERENCE_COLUMN


python3 -m ftt.approaches.lora \
    --name "LoRA Experiment" \
    --description "LoRA experiment" \
    --output_dir "./out/lora_experiment" \
    --training_args.num_epochs 5 \
    --training_args.batch_size 1 \
    --training_args.learning_rate 5e-6 \
    --num_repetitions 5 \
    --watch_done_file True \
    --base_model_name "meta-llama/Llama-3.2-1B" \
    --dataset_name "glue_mrpc" \
    --dataset_validation "forward" \
"""
