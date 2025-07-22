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
        self.fast_log_writer.set_columns(["step", "min_layer_id", "fisher_score"])

        self.marker_train_loss = None
        self.marker_gradient_sensitivity = None

    def _compute_fisher_scores(self, model):
        fisher_information = {}
        with torch.no_grad():
            for name, module in model.named_modules():
                layer_id = self.layer_id_parse_rule(name)
                if (
                    isinstance(module, LoRALayer)
                    and module.A.requires_grad
                    and module.B.requires_grad
                    and module.A.grad is not None
                    and module.B.grad is not None
                ):
                    a_fish = (module.A.grad.data.clone().detach() ** 2).mean().to(torch.float32).cpu().numpy()
                    b_fish = (module.B.grad.data.clone().detach() ** 2).mean().to(torch.float32).cpu().numpy()
                    if layer_id not in fisher_information:
                        fisher_information[layer_id] = {"A": a_fish, "B": b_fish}
                    else:
                        fisher_information[layer_id]["A"] += a_fish
                        fisher_information[layer_id]["B"] += b_fish
        return fisher_information

    def on_step_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the beginning of each training step. Randomly disables gradients for LoRA layers.

        Args:
            info (dict): Step information.
            trainer (Trainer): Trainer instance.
        """
        model = trainer.model
        global_step = info["global_step"]
        fisher_information = self._compute_fisher_scores(model)
        fisher_sum = sum(float(el["A"]) + float(el["B"]) for el in fisher_information.values())
        Monitor().print(f"Fisher information sum: {fisher_sum}")
        self.fast_log_writer.append(
            {
                "step": global_step,
                "min_layer_id": self.current_min_layer_id,
                "fisher_score": fisher_sum,
            }
        )

        min_layer_id = int(random.betavariate(self.alpha, self.beta) * self.num_total_layers)
        self.current_min_layer_id = min_layer_id
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                layer_id = self.layer_id_parse_rule(name)
                if layer_id < min_layer_id and layer_id != self.num_total_layers - 1:
                    module.A.requires_grad_(False)
                    module.B.requires_grad_(False)
                else:
                    module.A.requires_grad_(True)
                    module.B.requires_grad_(True)

    def on_step_end(self, info, trainer):
        global_step = info["global_step"]


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
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in self.model.named_modules()) + 1
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
