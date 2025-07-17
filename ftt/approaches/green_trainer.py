import os
import signal
from typing import List, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.approaches.lora import LoRAExperiment, LoRAExperimentConfig
from ftt.datasets import get_dataset
from ftt.lora import LoRAModel, load_model
from ftt.lora_strategies import LoRAPartialStrategy, LoRAUniformStrategy
from ftt.model_impact_callbacks import GreenTrainerCallback
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


class GreenTrainerApproachConfig(LoRAExperimentConfig):
    """
    Configuration for the GreenTrainerA approach experiment.
    This class extends LoRAExperimentConfig to include parameters specific to GreenTrainerA approaches.
    """

    rho: float = 0.5
    importance_interval: int = 200


class GreenTrainerApproach(LoRAExperiment):
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
        return f"{super().get_repetition_output_dir(repid)}-rho-{self.config.rho}-interval-{self.config.importance_interval}"

    def load_model_and_tokenizer(self) -> Tuple[Module, PreTrainedTokenizer]:

        base_model, tokenizer = load_model(self.config.base_model_name, self.config.tokenizer_name)
        return base_model, tokenizer

    def load_additional_callbacks(self, *args, **kwargs) -> List[TrainerCallback]:
        """
        Load additional callbacks specific to the GreenTrainerA approach.

        Returns:
            List[TrainerCallback]: A list of additional callbacks.
        """
        return [
            GreenTrainerCallback(
                rho=self.config.rho,
                interval=self.config.importance_interval,
            )
        ]


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(GreenTrainerApproachConfig, verbose=True)
    experiment = GreenTrainerApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
