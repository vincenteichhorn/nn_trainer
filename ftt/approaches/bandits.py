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

    def __init__(self, num_features: int):
        """
        Initialize the bandit with the number of features.

        Args:
            num_features (int): Number of features (dimension of the action space).
        """
        self.num_features = num_features

    @abstractmethod
    def select_action(self, possible_actions: List[List[float]], budget_increase: List[float] = None) -> List[float]:
        """
        Select an action based on the current features and budget.

        Args:
            possible_actions (List[List[float]]): List of possible actions.
            budget_increase (List[float], optional): List of budget increases for each action.

        Returns:
            List[float]: Selected action.
        """
        pass

    @abstractmethod
    def update(self, action: List[float], reward: float):
        """
        Update the bandit state based on the selected action and received reward.

        Args:
            action (List[float]): The action that was taken.
            reward (float): The reward received for the action.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the bandit state.
        This method should be called to reset the bandit for a new episode or run.
        """
        pass


class dUCBBandit:
    def __init__(
        self,
        num_features: int,
        gamma: float = 0.75,
        lmbd: float = 0.001,
        delta: float = 0.01,
        sigma: float = 0.5,
        S: float = 1.0,
        L: float = 1.0,
        verbose: bool = False,
    ):
        """
        D-LinUCB Bandit implementation with general L.

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
        self.num_features = num_features
        self.gamma = gamma
        self.lmbd = lmbd
        self.delta = delta
        self.sigma = sigma
        self.S = S
        self.L = L
        self.verbose = verbose

        self.t = 0
        self.gamma2_t = 1.0

        self.V = self.lmbd * np.eye(self.num_features)
        self.Ve = self.lmbd * np.eye(self.num_features)
        self.b = np.zeros(self.num_features)
        self.theta_hat = np.zeros(self.num_features)

    def compute_beta(self):
        log_term = 2 * math.log(1.0 / self.delta) + self.num_features * math.log(
            1.0 + (1.0 - self.gamma2_t) * (self.L**2) / (self.lmbd * self.num_features * (1.0 - self.gamma**2))
        )
        return np.sqrt(self.lmbd) * self.S + self.sigma * np.sqrt(log_term)

    def select_action(self, possible_actions: List[np.ndarray], cost: List[float] = None) -> np.ndarray:
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

        for a in possible_actions:
            assert np.linalg.norm(a) <= self.L + 1e-8, f"Action norms must be ≤ {self.L}"

        V_inv = np.linalg.inv(self.V)
        self.theta_hat = V_inv @ self.b

        ucb_values = []
        if cost is None:
            cost = [1.0] * len(possible_actions)
        for a, c in zip(possible_actions, cost):
            score = self.theta_hat @ a
            v_inv_a = V_inv @ a
            uncertainty = beta * np.sqrt(v_inv_a @ self.Ve @ v_inv_a)
            ucb = (score + uncertainty) / c
            ucb_values.append(ucb)

        idx = int(np.argmax(ucb_values))

        if self.verbose:
            print(f"[t={self.t}] Selected action index: {idx}")
            print("Theta_hat:", self.theta_hat)
            print("Beta_t:", beta)
            print("UCB values:", ucb_values)

        return possible_actions[idx]

    def update(self, action: np.ndarray, reward: float):
        """
        Update internal parameters.

        Args:
            action (np.ndarray): Action taken.
            reward (float): Observed reward.
        """
        outer = np.outer(action, action)
        self.V = self.gamma * self.V + outer + (1.0 - self.gamma) * self.lmbd * np.eye(self.num_features)
        self.Ve = self.gamma**2 * self.Ve + outer + (1.0 - self.gamma**2) * self.lmbd * np.eye(self.num_features)
        self.b = self.gamma * self.b + reward * action

        if self.verbose:
            print("Updated V matrix:", self.V)
            print("Updated b vector:", self.b)
            print("Reward:", reward)

    def reset(self):
        """
        Reinitialize internal state while keeping hyperparameters.
        """
        self.t = 0
        self.gamma2_t = 1.0
        self.V = self.lmbd * np.eye(self.num_features)
        self.Ve = self.lmbd * np.eye(self.num_features)
        self.b = np.zeros(self.num_features)
        self.theta_hat = np.zeros(self.num_features)

        if self.verbose:
            print("Bandit state has been reset.")


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
        self.current_loss = np.inf
        self.current_action = None
        self.possible_actions = np.array([[0] * i + [1] * (self.num_total_layers - i) for i in range(self.num_total_layers)])
        self.budget_increase = np.array(
            [sum(action) for action in self.possible_actions]
        )  # [1] * len(self.possible_actions)

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

    def _compute_fisher_score(self, model):
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
        if global_step < 25:
            return
        self.current_action = self.bandit.select_action(self.possible_actions, self.budget_increase)
        min_layer_id = list(self.current_action).index(max(self.current_action))
        self.update_trainable_lora_layers(trainer.model, min_layer_id)
        train_loss = info["train_loss"]
        if train_loss:
            self.current_loss = train_loss
        Monitor().print(f"Selected action with min layer ID: {min_layer_id}")

    def on_step_end(self, info, trainer):
        # train_loss = info["train_loss"]
        # if train_loss and self.current_loss != np.inf:
        #     loss_change = self.current_loss - train_loss
        #     self.bandit.update(self.current_action, loss_change)
        if self.current_action is None:
            return
        fisher_score = self._compute_fisher_score(trainer.model)
        self.bandit.update(self.current_action, fisher_score)

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
        layer_parse_rule = lambda name: (int(name.split(".")[3]) if len(name.split(".")) > 3 else 0)  # noqa: E731
        num_total_layers = max(layer_parse_rule(name) for name, _ in self.model.named_modules()) + 1
        if self.config.bandit == "dLinUCB":
            bandit = dUCBBandit(
                num_features=num_total_layers,
                gamma=self.config.gamma,
                lmbd=self.config.lmda,
                delta=self.config.delta,
                sigma=self.config.sigma,
                L=num_total_layers,
            )
        else:
            raise ValueError(f"Unknown bandit type: {self.config.bandit}")
        return [
            BanditCallback(
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
