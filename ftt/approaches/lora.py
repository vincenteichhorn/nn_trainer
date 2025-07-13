from dataclasses import dataclass
import os
from typing import List, Literal, Tuple
import torch
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
    validate_strategy: Literal["steps", "epochs"] = "epochs"
    validate_every: int = 1
    validation_batch_size: int = 16
    generation_reference_column: str = "output"
    generation_max_length: int = 128


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

    def load_additional_callbacks(self, *args, **kwargs) -> List[TrainerCallback]:
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

    def get_repetition_output_dir(self, repid: int) -> str:
        """
        Get the output directory for a specific repetition.

        Args:
            repid (int): Repetition ID.
        Returns:
            str: Output directory path for the repetition.
        """
        return os.path.join(self.config.output_dir, self.config.dataset_name, str(repid))

    def run(self):
        """
        Run the static approach using the provided configuration.
        This method should be implemented by subclasses.
        """
        for repid in range(self.config.num_repetitions):
            model, tokenizer = self.load_model_and_tokenizer()
            output_dir = self.get_repetition_output_dir(repid)
            done_file = f"{output_dir}/donefile"
            if self.config.watch_done_file and os.path.exists(done_file):
                print(f"Skipping repetition {repid} as done file exists: {done_file}")
                continue
            print(f"Running repetition {repid} with output directory: {output_dir}")

            validator = None
            validation_arguments = ValidationArguments(
                batch_size=self.config.validation_batch_size, data_collator=DataCollatorForCausalLM(tokenizer)
            )
            if self.config.dataset_validation == "forward":
                validator = ForwardValidator(
                    model=model,
                    validation_args=validation_arguments,
                    validation_data=self.dataset["validation"],
                    metrics=[
                        OneHotClassificationMetrics(
                            num_classes=len(self.dataset.get_task_classes()),
                            classes=tokenizer.convert_tokens_to_ids(self.dataset.get_task_classes()),
                            targets_key="labels",
                            logits_key="logits",
                        )
                    ],
                )
            elif self.config.dataset_validation == "generation":
                validator = GenerationValidator(
                    model=model,
                    tokenizer=tokenizer,
                    validation_args=validation_arguments,
                    validation_data=self.dataset["generation"],
                    max_length=self.config.generation_max_length,
                    metrics=[
                        BleuScore(target_key=self.config.generation_reference_column),
                        NistScore(target_key=self.config.generation_reference_column),
                        RougeScore(target_key=self.config.generation_reference_column),
                        MeteorScore(target_key=self.config.generation_reference_column),
                    ],
                )

            trainer = Trainer(
                output_dir=output_dir,
                model=model,
                training_args=self.config.training_args,
                train_data=self.dataset["train"],
                callbacks=[
                    EnergyCallback(output_dir=output_dir, nvidia_query_interval=10),
                    LoggingCallback(output_dir=output_dir),
                    ValidatorCallback(
                        output_dir=output_dir,
                        log_file="validation_log.csv",
                        validator=validator,
                        validate_strategy=self.config.validate_strategy,
                        validate_every=self.config.validate_every,
                    ),
                    FLOPsBudgetControllCallback(output_dir=output_dir, budget=1e9, should_stop_training=False),
                    *self.load_additional_callbacks(rep_id=repid),
                ],
            )
            trainer.train()
            if self.config.watch_done_file:
                with open(done_file, "w") as f:
                    f.write("done")
            del model, tokenizer, trainer, validator
            torch.cuda.empty_cache()
            print(f"Repetition {repid} completed. Output saved to {output_dir}.")


if __name__ == "__main__":

    config = experiment_config_cli(LoRAExperimentConfig)
    experiment = LoRAExperiment(config)
    experiment.run()
    print("Experiment completed successfully.")
