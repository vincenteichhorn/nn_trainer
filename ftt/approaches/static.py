import os
import signal
import sys
from typing import List, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.approaches.lora import LoRAExperiment, LoRAExperimentConfig
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


class StaticApproachConfig(LoRAExperimentConfig):
    """
    Configuration for the static approach experiment.
    This class extends LoRAExperimentConfig to include parameters specific to static approaches.
    """

    num_top_layers: int = 1


class StaticApproach(LoRAExperiment):
    """
    Static approach experiment that uses LoRA with a static model.
    This class extends LoRAExperiment to implement the static approach.
    """

    def get_repetition_output_dir(self, repid: int) -> str:
        """
        Get the output directory for a specific repetition.

        Args:
            repid (int): The repetition ID.

        Returns:
            str: The output directory path for the specified repetition.
        """
        return f"{super().get_repetition_output_dir(repid)}-nlayer-{self.config.num_top_layers}"

    def load_model_and_tokenizer(self):
        base_model, tokenizer = load_model(self.config.base_model_name, self.config.tokenizer_name)
        layer_parse_rule = lambda name: (int(name.split(".")[2]) if len(name.split(".")) > 2 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in base_model.named_modules()) + 1
        model = LoRAModel(
            base_model,
            LoRAPartialStrategy(
                rank=self.config.lora_rank,
                dropout=self.config.lora_dropout,
                alpha=self.config.lora_alpha,
                num_total_layers=num_total_layers,
                num_layers=self.config.num_top_layers,
                begin_from="top",
                layer_id_parse_rule=layer_parse_rule,
            ),
            self.config.base_model_name,
        )
        return model, tokenizer


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(StaticApproachConfig, verbose=True)
    experiment = StaticApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
