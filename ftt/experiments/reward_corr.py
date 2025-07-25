import math
import os
import random
from typing import Dict, List, Any, Callable

import numpy as np
import torch
from ftt.approaches.lora import LoRAExperiment
from ftt.approaches.stochastic import StochasticApproachConfig
from ftt.lora import LoRALayer
from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.experiment import experiment_config_cli
from nnt.trainer import Trainer
from nnt.util.fast_csv import FastCSV
from nnt.util.functions import save_json
from nnt.util.monitor import Monitor


class StochasticLoRATestCallback(TrainerCallback):
    """
    Callback for stochastic dropping of LoRA layers during training.
    Randomly disables gradients for layers below a sampled threshold.
    """

    def __init__(
        self,
        layer_id_parse_rule: Callable[[str], int],
        num_total_layers: int,
        random_seed: int = 42,
        savings: float = 0.5,
        concentration: float = 5,
        output_dir: str = None,
    ):
        """
        Args:
            layer_id_parse_rule (callable): Function to parse layer ID from module name.
            num_total_layers (int): Total number of layers.
            random_seed (int): Seed for random number generator.
            savings (float): Fraction of layers to drop (between 0 and 1).
        """
        self.savings = savings
        assert 0 < savings < 1, "savings must be in (0, 1)"
        self.k = concentration
        self.alpha = self.savings * self.k
        self.beta = (1 - self.savings) * self.k
        print(f"Using beta distribution with alpha={self.alpha}, beta={self.beta}")
        self.num_total_layers = num_total_layers
        self.layer_id_parse_rule = layer_id_parse_rule
        self.current_min_layer_id = 0
        self.output_dir = output_dir
        random.seed(random_seed)

        self.fast_log_writer = FastCSV(file_path=os.path.join(self.output_dir, "reward_log.csv"), force=True)

        self.init_train_loss = None
        self.absolute_consumed_budget = 0

        self.memorization = {
            "min_layer_id": self.num_total_layers - 1,
            "train_loss": None,
            "total_relative_loss_change": None,
            "total_relative_budget_change": None,
            "current_relative_loss_change": None,
            "current_relative_budget_change": None,
        }
        self.fast_log_writer.set_columns(["step"] + list(self.memorization.keys()))

    def _update_trainable_lora_layers(self, model, min_layer_id: int):
        """
        Updates the requires_grad flag for LoRA layers based on the selected minimum layer ID.

        Args:
            model: Model containing LoRA layers.
            min_layer_id (int): The minimum layer ID to consider for training.
        """
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                layer_id = self.layer_id_parse_rule(name)
                if layer_id < min_layer_id:
                    module.A.requires_grad_(False)
                    module.B.requires_grad_(False)
                else:
                    module.A.requires_grad_(True)
                    module.B.requires_grad_(True)

    def _model_gradient_norm(self, model):
        sum_squared_gradients = 0.0
        with torch.no_grad():
            for _, module in model.named_modules():
                if (
                    isinstance(module, LoRALayer)
                    and module.A.requires_grad
                    and module.B.requires_grad
                    and module.A.grad is not None
                    and module.B.grad is not None
                ):
                    a_norm = (module.A.grad.clone().detach() ** 2).sum().to(torch.float32).cpu().numpy()
                    b_norm = (module.B.grad.clone().detach() ** 2).sum().to(torch.float32).cpu().numpy()
                    sum_squared_gradients += a_norm + b_norm
        return math.sqrt(sum_squared_gradients)

    def on_step_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the beginning of each training step. Randomly disables gradients for LoRA layers.

        Args:
            info (dict): Step information.
            trainer (Trainer): Trainer instance.
        """
        model = trainer.model
        if info["train_loss"] is None and info["global_step"] < 25:
            return

        if self.init_train_loss is None:
            self.init_train_loss = info["train_loss"]

        self.memorization["total_relative_loss_change"] = (self.init_train_loss - info["train_loss"]) / self.init_train_loss

        total_budget = self.num_total_layers * info["num_train_steps"]
        self.absolute_consumed_budget += (self.memorization["min_layer_id"] + 1) / self.num_total_layers
        self.memorization["total_relative_budget_change"] = self.absolute_consumed_budget / total_budget

        self.memorization["current_relative_budget_change"] = (
            (self.memorization["min_layer_id"] + 1) / self.num_total_layers / total_budget
        )
        loss_change = (self.memorization["train_loss"] or info["train_loss"]) - info["train_loss"]
        self.memorization["current_relative_loss_change"] = loss_change / self.init_train_loss

        self.fast_log_writer.append(
            {
                "step": info["global_step"],
                **self.memorization,
            }
        )

        self.memorization["train_loss"] = info["train_loss"]
        self.memorization["min_layer_id"] = int(random.betavariate(self.alpha, self.beta) * self.num_total_layers)
        self._update_trainable_lora_layers(model, self.memorization["min_layer_id"])


class StochasticTestApproach(LoRAExperiment):
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
        model = kwargs["model"]
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in model.named_modules()) + 1
        rep_id = kwargs["rep_id"] if "rep_id" in kwargs else 0
        return [
            StochasticLoRATestCallback(
                layer_id_parse_rule=layer_parse_rule,
                num_total_layers=num_total_layers,
                savings=self.config.savings,
                concentration=self.config.concentration,
                random_seed=rep_id * int(100 * self.config.savings) + 42,
                output_dir=self.get_repetition_output_dir(repid=rep_id),
            )
        ]


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(StochasticApproachConfig, verbose=True)
    experiment = StochasticTestApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
