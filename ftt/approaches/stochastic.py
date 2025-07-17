import os
import signal
from typing import List, Literal, Tuple
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.approaches.lora import LoRAExperiment, LoRAExperimentConfig
from ftt.datasets import get_dataset
from ftt.lora import LoRAModel, load_model
from ftt.lora_strategies import LoRAPartialStrategy, LoRAUniformStrategy
from ftt.model_impact_callbacks import StochasticLoRACallback
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


class StochasticApproachConfig(LoRAExperimentConfig):
    """
    Configuration for the stochastic approach experiment.
    This class extends LoRAExperimentConfig to include parameters specific to stochastic approaches.
    """

    savings: float = 0.5
    concentration: float = 5


class StochasticApproach(LoRAExperiment):
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
        return f"{super().get_repetition_output_dir(repid)}-savings-{self.config.savings}-concentration-{self.config.concentration}"

    def load_additional_callbacks(self, *args, **kwargs) -> List[TrainerCallback]:
        """
        Load additional callbacks specific to the stochastic approach.

        Returns:
            List[TrainerCallback]: A list of additional callbacks.
        """
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in self.model.named_modules()) + 1
        rep_id = kwargs["rep_id"] if "rep_id" in kwargs else 0
        return [
            StochasticLoRACallback(
                layer_id_parse_rule=layer_parse_rule,
                num_total_layers=num_total_layers,
                savings=self.config.savings,
                random_seed=rep_id * int(100 * self.config.savings) + 42,
            )
        ]


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(StochasticApproachConfig, verbose=True)
    experiment = StochasticApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
