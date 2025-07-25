from abc import abstractmethod
import math
import os
import signal
from typing import List, Literal, Tuple
import numpy as np
import torch
from transformers import PreTrainedTokenizer
from torch.nn import Module
from ftt.approaches.lora import LoRAExperiment, LoRAExperimentConfig
from ftt.datasets import get_dataset
from ftt.lora import LoRALayer, LoRAModel, load_model
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
from nnt.util.fast_csv import FastCSV
from nnt.util.monitor import Monitor
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validation_metrics.generation_metrics import BleuScore, MeteorScore, NistScore, RougeScore
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.generation_validator import GenerationValidator
from nnt.validators.validator import ValidationArguments


class Bandit:
    """
    Base class for bandit approaches.
    This class defines the basic structure and methods for bandit approaches.
    """

    def __init__(self, num_actions: int, num_features: int):
        pass

    @abstractmethod
    def select_action(self, context_vector: np.ndarray) -> int:
        """
        Select an action based on the context vector.

        Args:
            context_vector (np.ndarray): The context vector for action selection.

        Returns:
            int: The index of the selected action.
        """
        pass

    @abstractmethod
    def update(self, chosen_action: int, context_vector: np.ndarray, reward: float):
        """
        Update the internal parameters based on the chosen action and observed reward.

        Args:
            chosen_action (int): The index of the chosen action.
            action_context (np.ndarray): The context vector associated with the chosen action.
            reward (float): The observed reward for the chosen action.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the internal state of the bandit while keeping hyperparameters.
        This method is called at the beginning of each new training run.
        """
        pass


class dLinUCBBandit(Bandit):
    def __init__(
        self,
        num_actions: int,
        gamma: float = 0.75,
        lmbd: float = 0.001,
        delta: float = 0.01,
        sigma: float = 0.5,
        S: float = 1.0,
        L: float = 1.0,
        verbose: bool = False,
    ):
        """

        Args:
            num_features (int): Feature dimension (d).
            gamma (float): Discount factor.
            lmbd (float): Regularization parameter.
            delta (float): Confidence level.
            sigma (float): Std deviation of noise.
            S (float): Bound on ||theta||.
            L (float): Bound on ||action||.
            verbose (bool): Enable debug logging.
        """
        self.num_actions = num_actions
        self.num_features = 2
        self.gamma = gamma
        self.lambda_ = lmbd
        self.delta = delta
        self.sigma = sigma
        self.S = S
        self.L = L
        self.verbose = verbose

        self.t = 0
        self.gamma2_t = 1.0

        self.V = self.lambda_ * np.eye(self.num_features)
        self.Ve = self.lambda_ * np.eye(self.num_features)
        self.b = np.zeros(self.num_features)

    def compute_beta(self):
        log_term = 2 * math.log(1.0 / self.delta) + self.num_features * math.log(
            1.0 + (1.0 - self.gamma2_t) * (self.L**2) / (self.lambda_ * self.num_features * (1.0 - self.gamma**2))
        )
        return np.sqrt(self.lambda_) * self.S + self.sigma * np.sqrt(log_term)

    def select_action(self) -> int:
        """
        Select an action using the D-LinUCB policy.

        Args:
            possible_actions (List[np.ndarray]): List of feature vectors.

        Returns:
            np.ndarray: Chosen action.
        """

        beta = self.compute_beta()
        self.t += 1
        self.gamma2_t *= self.gamma**2

        scores = []
        V_inv = np.linalg.inv(self.V)
        theta_hat = V_inv @ self.b

        for act in range(self.num_actions):
            action_vector = np.array([1.0, act + 1])
            expected_reward = action_vector @ theta_hat
            ucb_score = beta * np.sqrt(action_vector @ V_inv @ self.Ve @ V_inv @ action_vector)
            scores.append((expected_reward + ucb_score) / (act + 1))
        chosen_action = np.argmax(scores)
        return chosen_action

    def update(self, chosen_action: int, reward: float):
        """
        Update internal parameters.

        Args:
            action (np.ndarray): Action taken.
            reward (float): Observed reward.
        """
        action_vector = np.array([1.0, chosen_action + 1])
        outer = np.outer(action_vector, action_vector)
        self.V = self.gamma * self.V + outer + (1.0 - self.gamma) * self.lambda_ * np.eye(self.num_features)
        self.Ve = self.gamma**2 * self.Ve + outer + (1.0 - self.gamma**2) * self.lambda_ * np.eye(self.num_features)
        self.b = self.gamma * self.b + reward * action_vector

    def reset(self):
        """
        Reinitialize internal state while keeping hyperparameters.
        """
        self.t = 0
        self.gamma2_t = 1.0
        self.V = self.lambda_ * np.eye(self.num_features)
        self.Ve = self.lambda_ * np.eye(self.num_features)
        self.b = np.zeros(self.num_features)


class BanditCallback(TrainerCallback):
    """
    Callback for bandit approaches that manages the training process.
    This class extends TrainerCallback to implement the bandit approach logic.
    """

    def __init__(self, bandit: Bandit, num_total_layers: int, layer_id_parse_rule: callable, output_dir: str = None):
        """
        Initialize the BanditCallback with a Bandit instance.

        Args:
            bandit (Bandit): The bandit instance containing features, alpha, and gamma.
            num_total_layers (int): Total number of layers in the model.
            layer_id_parse_rule (callable): Function to parse layer IDs from module names.
        """
        super().__init__()
        self.output_dir = output_dir
        self.bandit = bandit
        self.num_total_layers = num_total_layers
        self.layer_id_parse_rule = layer_id_parse_rule
        self.current_action = None
        self.current_train_loss = None
        self.loss_history = []

        self.fast_csv_writer = FastCSV(os.path.join(output_dir, "bandit_log.csv"), force=True)
        self.fast_csv_writer.set_columns(
            [
                "global_step",
                "train_loss",
                "action",
                "reward",
            ]
        )

    def update_trainable_lora_layers(self, model, min_layer_id: int):
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

    def _compute_reward(self, model):
        fisher_score = 0.0
        with torch.no_grad():
            for _, module in model.named_modules():
                if (
                    isinstance(module, LoRALayer)
                    and module.A.requires_grad
                    and module.B.requires_grad
                    and module.A.grad is not None
                    and module.B.grad is not None
                ):
                    a_fish = (module.A.grad * module.A.data).clone().detach() ** 2
                    b_fish = (module.B.grad * module.B.data).clone().detach() ** 2
                    fisher_score += (
                        a_fish.mean().to(torch.float32).cpu().numpy() + b_fish.mean().to(torch.float32).cpu().numpy()
                    )
        return fisher_score

    def on_step_begin(self, info, trainer):
        """
        Called at the beginning of each training step to update the trainable LoRA layers.

        Args:
            info: Information about the current training step.
            trainer: The Trainer instance managing the training process.
        """
        global_step = info["global_step"]
        if global_step < 1:
            return
        self.current_train_loss = info["train_loss"]
        self.current_context_vector = np.array(self.loss_history)
        self.current_action = self.bandit.select_action(self.current_context_vector)
        min_layer_id = self.current_action
        self.update_trainable_lora_layers(trainer.model, min_layer_id)
        Monitor().print(f"Selected action with min layer ID: {min_layer_id}")

    def on_step_end(self, info, trainer):
        if self.current_action is None:
            return
        reward = self._compute_reward(trainer.model)
        self.bandit.update(self.current_action, self.current_context_vector, reward)

        self.fast_csv_writer.append(
            {
                "global_step": info["global_step"],
                "train_loss": math.exp(self.loss_history[-1]) - 1e-8 if self.loss_history else None,
                "action": self.current_action,
                "reward": reward,
            }
        )

    def __repr__(self):
        str = f"BanditCallback with {self.bandit.__class__.__name__}:\n" f"Number of Features: {self.bandit.num_features}\n"
        return str


class BanditApproachConfig(LoRAExperimentConfig):
    """
    Configuration for the Adaptive approach experiment.
    This class extends LoRAExperimentConfig to include parameters specific to Adaptive approaches.
    """

    gamma: float = 0.9
    lmda: float = 0.05
    delta: float = 0.05
    sigma: float = 0.5
    bandit: Literal["dUCB"] = "dUCB"


class BanditApproach(LoRAExperiment):
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
        return f"{super().get_repetition_output_dir(repid)}-" + "-".join(
            [
                f"bandit={self.config.bandit}",
                f"gamma={self.config.gamma}",
                f"lmda={self.config.lmda}",
                f"delta={self.config.delta}",
                f"sigma={self.config.sigma}",
            ]
        )

    def load_additional_callbacks(self, *args, **kwargs) -> List[TrainerCallback]:
        """
        Load additional callbacks specific to the Adaptive approach.

        Returns:
            List[TrainerCallback]: A list of additional callbacks.
        """
        model = kwargs.get("model")
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in model.named_modules()) + 1
        if self.config.bandit == "dLinUCB":
            bandit = dLinUCBBandit(
                num_actions=num_total_layers,
                num_features=10,
                gamma=self.config.gamma,
                lmbd=self.config.lmda,
                delta=self.config.delta,
                sigma=self.config.sigma,
            )
        else:
            raise ValueError(f"Unknown bandit type: {self.config.bandit}")
        return [
            BanditCallback(
                output_dir=self.get_repetition_output_dir(kwargs.get("repid", 0)),
                bandit=bandit,
                num_total_layers=num_total_layers,
                layer_id_parse_rule=layer_parse_rule,
            ),
        ]


if __name__ == "__main__":

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    config = experiment_config_cli(BanditApproachConfig, verbose=True)
    experiment = BanditApproach(config)
    experiment.run()
    print("Experiment completed successfully.")
