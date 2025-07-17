import os
import signal
from typing import List, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.approaches.lora import LoRAExperiment, LoRAExperimentConfig
from ftt.datasets import get_dataset
from ftt.lora import LoRAModel, load_model
from ftt.lora_strategies import LoRAPartialStrategy, LoRAUniformStrategy
from ftt.model_impact_callbacks import AdaptiveLoRACallback
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


class AdaptiveApproachConfig(LoRAExperimentConfig):
    """
    Configuration for the Adaptive approach experiment.
    This class extends LoRAExperimentConfig to include parameters specific to Adaptive approaches.
    """

    rho: float = 0.5
    sub_approach: Literal["deterministic", "stochastic"] = "deterministic"
    importance_interval: int = 200


class AdaptiveApproach(LoRAExperiment):
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
        return f"{super().get_repetition_output_dir(repid)}-rho-{self.config.rho}-approach-{self.config.sub_approach}-interval-{self.config.importance_interval}"

    def load_additional_callbacks(self, *args, **kwargs) -> List[TrainerCallback]:
        """
        Load additional callbacks specific to the Adaptive approach.

        Returns:
            List[TrainerCallback]: A list of additional callbacks.
        """
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in self.model.named_modules()) + 1
        return [
            AdaptiveLoRACallback(
                num_total_layers=num_total_layers,
                layer_id_parse_rule=layer_parse_rule,
                approach=self.config.sub_approach,
                determinstic_rho=self.config.rho,
                interval=self.config.importance_interval,
            )
        ]


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(AdaptiveApproachConfig, verbose=True)
    experiment = AdaptiveApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
