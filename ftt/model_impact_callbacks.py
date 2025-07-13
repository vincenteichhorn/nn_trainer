import random
from typing import TYPE_CHECKING, Callable, Dict, Any

import numpy as np
import torch
from ftt.lora import LoRALayer
from nnt.callbacks.trainer_callback import TrainerCallback

if TYPE_CHECKING:
    from nnt.trainer import Trainer


class StochasticLoRACallback(TrainerCallback):
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
        self.num_total_layers = num_total_layers
        self.layer_id_parse_rule = layer_id_parse_rule
        random.seed(random_seed)

    def on_step_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the beginning of each training step. Randomly disables gradients for LoRA layers.

        Args:
            info (dict): Step information.
            trainer (Trainer): Trainer instance.
        """
        model = trainer.model
        min_layer_id = int(random.betavariate(self.alpha, self.beta) * self.num_total_layers)
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                layer_id = self.layer_id_parse_rule(name)
                if layer_id < min_layer_id and layer_id != self.num_total_layers - 1:
                    module.A.requires_grad_(False)
                    module.B.requires_grad_(False)
                else:
                    module.A.requires_grad_(True)
                    module.B.requires_grad_(True)


class AdaptiveLoRACallback(TrainerCallback):
    """
    Callback for adaptively selecting trainable LoRA layers based on their importance.
    Supports stochastic and deterministic selection approaches.
    """

    def __init__(
        self,
        layer_id_parse_rule: Callable[[str], int],
        num_total_layers: int,
        interval: int = 100,
        approach: str = "stochastic",
        determinstic_rho: float = 0.5,
    ):
        """
        Args:
            layer_id_parse_rule (callable): Function to parse layer ID from module name.
            num_total_layers (int): Total number of layers.
            interval (int): Interval for updating layer importances.
            approach (str): Selection approach ("stochastic" or "deterministic").
            determinstic_rho (float): Threshold for deterministic selection.
        """
        self.approach = approach
        self.determinstic_rho = determinstic_rho
        self.interval = interval
        self.layer_id_parse_rule = layer_id_parse_rule
        self.num_total_layers = num_total_layers
        self.probabilities = np.zeros(num_total_layers, dtype=np.float32)
        self.probabilities[0] = 1.0

    def on_step_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the beginning of each training step. Updates trainable layers based on importance.

        Args:
            info (dict): Step information.
            trainer (Trainer): Trainer instance.
        """
        step = info["global_step"]
        model = trainer.model
        optimizer = trainer.optimizer

        if step % self.interval == 0:
            current_batch = info["current_batch"]
            layer_importances = self.score_importance(model, optimizer, current_batch)
            self.probabilities = layer_importances
            self.min_layer_deterministic = np.argmin(np.cumsum(self.probabilities[::-1])[::-1] >= self.determinstic_rho)
            print([f"{p:.4f}" for p in self.probabilities], self.min_layer_deterministic)

        self.update_trainable_layers(model)

    def get_min_layer_id(self) -> int:
        """
        Returns the minimum layer ID to keep trainable based on the selection approach.

        Returns:
            int: Minimum layer ID.
        """
        if self.approach == "stochastic":
            min_layer_id = np.random.choice(np.arange(self.num_total_layers), p=self.probabilities)
        elif self.approach == "deterministic":
            return self.min_layer_deterministic
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
        return min_layer_id

    def update_trainable_layers(self, model):
        """
        Updates the requires_grad flag for LoRA layers based on the selected minimum layer ID.

        Args:
            model: Model containing LoRA layers.
        """
        min_layer_id = self.get_min_layer_id()
        for name, module in model.named_modules():
            if isinstance(module, LoRALayer):
                layer_id = self.layer_id_parse_rule(name)
                if layer_id < min_layer_id:
                    module.A.requires_grad_(False)
                    module.B.requires_grad_(False)
                else:
                    module.A.requires_grad_(True)
                    module.B.requires_grad_(True)

    def score_importance(self, model, optimizer, batch) -> np.ndarray:
        """
        Scores the importance of each LoRA layer by measuring weight changes and gradients.

        Args:
            model: Model containing LoRA layers.
            optimizer: Optimizer instance.
            batch: Current training batch.

        Returns:
            np.ndarray: Importance scores for each layer.
        """
        lora_modules = {name: module for name, module in model.named_modules() if isinstance(module, LoRALayer)}

        # Set all LoRA layers to be trainable
        for module in lora_modules.values():
            module.A.requires_grad_(True)
            module.B.requires_grad_(True)

        # Cache original weights
        original_weights = {name: (module.A.data.clone(), module.B.data.clone()) for name, module in lora_modules.items()}

        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Cache updated weights
        updated_weights = {name: (module.A.data.clone(), module.B.data.clone()) for name, module in lora_modules.items()}

        # Compute weight differences
        weight_differences = {
            name: (
                updated_weights[name][0] - original_weights[name][0],
                updated_weights[name][1] - original_weights[name][1],
            )
            for name in lora_modules.keys()
        }

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Cache gradients
        gradients = {name: (module.A.grad.clone(), module.B.grad.clone()) for name, module in lora_modules.items()}

        # Compute importance scores (absolute sum of weight differences)
        importance_scores = {
            name: (
                (weight_diff[0]).abs().sum().item(),
                (weight_diff[1]).abs().sum().item(),
            )
            for name, (grad, weight_diff) in zip(lora_modules.keys(), zip(gradients.values(), weight_differences.values()))
        }

        # Aggregate importance scores per layer
        layer_importances = {}
        for name, (score_a, score_b) in importance_scores.items():
            layer_id = self.layer_id_parse_rule(name)
            if layer_id not in layer_importances:
                layer_importances[layer_id] = 0
            layer_importances[layer_id] += score_a + score_b

        importances = np.array(
            [el for _, el in sorted(layer_importances.items(), key=lambda item: item[0])], dtype=np.float32
        )
        importances = np.exp(importances)
        importances = importances / np.sum(importances)

        # Restore original weights
        for name, (a, b) in original_weights.items():
            lora_modules[name].A.data = a
            lora_modules[name].B.data = b
        return importances


class GreenTrainerCallback(TrainerCallback):
    """
    Callback for dynamic selection of trainable layers based on FLOPs and importance scores.
    Implements a dynamic programming approach for layer selection.
    """

    def __init__(self, interval: int = 200, rho: float = 0.5):
        """
        Args:
            interval (int): Interval for updating layer selection.
            rho (float): Fraction of total FLOPs allowed for selected layers.
        """
        self.interval = interval
        self.rho = rho
        self.model_params = {}
        self.downscale_factor = 1e9

    def on_training_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the start of training. Initializes model parameters.

        Args:
            info (dict): Training information.
            trainer (Trainer): Trainer instance.
        """
        model = trainer.model
        self.model_params = {name: param for name, param in model.named_parameters()}
        self.selectable_layers = np.array([param.requires_grad for _, param in self.model_params.items()])
        self.selectable_layers = np.flip(self.selectable_layers)

    def on_step_begin(self, info: Dict[str, Any], trainer: "Trainer"):
        """
        Called at the beginning of each training step. Selects trainable layers based on FLOPs and importance.

        Args:
            info (dict): Step information.
            trainer (Trainer): Trainer instance.
        """
        step = info["global_step"]
        if step % self.interval == 0:
            model = trainer.model
            current_batch = info["current_batch"]
            activation_tensor_flops, gradient_tensor_flops = self.get_model_tensor_flops(model, current_batch)
            activation_tensor_flops = np.array(activation_tensor_flops, dtype=np.int64)
            gradient_tensor_flops = np.array(gradient_tensor_flops, dtype=np.int64)
            activation_tensor_flops = np.flip(activation_tensor_flops)
            gradient_tensor_flops = np.flip(gradient_tensor_flops)

            optimizer = trainer.optimizer
            importances = self.score_importance(model, optimizer, current_batch)
            importances = np.array(importances, dtype=np.float32)
            importances = np.flip(importances)

            max_importance, mask = self.select_layers_dp(
                activation_tensor_flops,
                gradient_tensor_flops,
                importances,
                self.rho,
                selectable_mask=None,
                verbose=True,
            )
            selected_flops = np.sum(activation_tensor_flops[mask == 1]) + np.sum(gradient_tensor_flops[mask == 1])
            print(
                f"Selected {selected_flops:.2e} out of {(np.sum(activation_tensor_flops) + np.sum(gradient_tensor_flops)):.2e} FLOPs ({selected_flops / (np.sum(activation_tensor_flops) + np.sum(gradient_tensor_flops)) * 100:.2f}%)"
            )
            mask = np.flip(mask)
            if sum(mask) == 0:
                print("No layers selected, skipping")
                return
            for i, (_, param) in enumerate(self.model_params.items()):
                req_grad = bool(mask[i] == 1)
                param.requires_grad_(req_grad)

    def score_importance(self, model, optimizer, batch) -> np.ndarray:
        """
        Scores the importance of each parameter by measuring weight changes and gradients.

        Args:
            model: Model instance.
            optimizer: Optimizer instance.
            batch: Current training batch.

        Returns:
            np.ndarray: Importance scores for each parameter.
        """
        self.reset_trainable_layers(model)

        def get_weights(param):
            return param.data.clone()

        original_weights = {name: get_weights(param) for name, param in self.model_params.items()}

        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        updated_weights = {name: get_weights(param) for name, param in self.model_params.items()}

        def get_weight_differences(original, updated):
            return original - updated

        weight_differences = {
            name: get_weight_differences(original, updated)
            for name, (original, updated) in zip(
                self.model_params.keys(),
                zip(original_weights.values(), updated_weights.values()),
            )
        }

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        def get_gradients(param):
            return param.grad.clone() if param.grad is not None else torch.zeros_like(param.data)

        gradients = {name: get_gradients(param) for name, param in self.model_params.items()}

        def compute_importance_scores(grad, weight_diff):
            return torch.sum(grad * weight_diff).item()

        importances = {
            name: compute_importance_scores(grad, weight_diff)
            for name, (grad, weight_diff) in zip(
                self.model_params.keys(),
                zip(gradients.values(), weight_differences.values()),
            )
        }
        importances = np.array(list(importances.values()), dtype=np.float32)
        importances = importances / np.max(np.abs(importances))
        for name, original in original_weights.items():
            self.model_params[name].data = original

        return importances

    def reset_trainable_layers(self, model):
        """
        Sets all parameters to be trainable.

        Args:
            model: Model instance.
        """
        if not self.model_params:
            return
        for name, param in self.model_params.items():
            param.requires_grad_(True)

    def get_model_tensor_flops(self, model, current_batch):
        """
        Estimates FLOPs for activations and gradients for each parameter.

        Args:
            model: Model instance.
            current_batch: Current training batch.

        Returns:
            Tuple[List[int], List[int]]: Activation and gradient FLOPs per parameter.
        """
        input_length = current_batch["input_ids"].shape[1]
        batch_size = current_batch["input_ids"].shape[0]

        self.reset_trainable_layers(model)

        activation_tensor_flops = []
        gradient_tensor_flops = []

        for idx, (name, param) in enumerate(self.model_params.items()):
            if "embed_tokens" in name:
                activation_tensor_flops.append(0)
                gradient_tensor_flops.append(0)
            elif "proj" in name:
                if "weight" in name or "lora":
                    flops = 2 * input_length * param.shape[0] * param.shape[1] * batch_size
                    activation_tensor_flops.append(flops)
                    gradient_tensor_flops.append(flops)
                elif "bias" in name:
                    flops = input_length * param.shape[0] * batch_size
                    activation_tensor_flops.append(0)
                    gradient_tensor_flops.append(flops)
                else:
                    print("Unknown projection name:", name)
            elif "norm" in name:
                if "weight" in name:
                    flops = input_length * param.shape[0] * batch_size
                    activation_tensor_flops.append(flops)
                    gradient_tensor_flops.append(flops)
                elif "bias" in name:
                    activation_tensor_flops.append(0)
                    gradient_tensor_flops.append(0)
                else:
                    print("Unknown norm name:", name)
            else:
                print("Unknown layer name:", name)
        return activation_tensor_flops, gradient_tensor_flops

    def select_layers_dp(
        self,
        t_dy,
        t_dw,
        importances,
        rho=0.3,
        selectable_mask=None,
        verbose=False,
    ):
        """
        Solves the layer selection problem using dynamic programming.

        Args:
            t_dy (np.ndarray): Time cost before each layer during backpropagation [N]
            t_dw (np.ndarray): Time cost within each layer during backpropagation [N]
            importances (np.ndarray): Importance score of selecting each layer [N]
            rho (float): Maximum allowed fraction of total backward pass time
            selectable_mask (np.ndarray, optional): Binary mask of whether each layer is selectable [N]
            verbose (bool): If True, prints DP progress

        Returns:
            Tuple[float, np.ndarray]: Maximum importance achievable and binary mask of selected layers [N]
        """

        if selectable_mask is not None:
            MAX_TIME_BUCKETS = 1e3
            scale = MAX_TIME_BUCKETS / np.max(t_dw + t_dy)
            t_dw = np.maximum((t_dw * scale).astype(np.int64), 1)
            t_dy = np.maximum((t_dy * scale).astype(np.int64), 1)
        else:
            t_dw = (t_dw.astype(np.int64) / self.downscale_factor).astype(np.int64)
            t_dy = (t_dy.astype(np.int64) / self.downscale_factor).astype(np.int64)

        num_layers = len(t_dw)
        total_bp_time = np.sum(t_dw + t_dy)
        time_limit = int(rho * total_bp_time)

        if selectable_mask is None:
            selectable_mask = np.ones(num_layers, dtype=np.uint8)

        # Determine the max number of layers we can consider before exceeding the time limit
        cumulative_dy = 0
        for limit in range(num_layers):
            cumulative_dy += t_dy[limit]
            if cumulative_dy > time_limit:
                break
        layer_limit = limit

        # DP tables
        dp_table = np.zeros((layer_limit + 1, time_limit + 1), dtype=np.float32)  # importance
        feasible = np.zeros((layer_limit + 1, time_limit + 1), dtype=np.uint8)  # state feasibility
        selection_table = np.zeros((layer_limit + 1, time_limit + 1, num_layers), dtype=np.uint8)  # selection mask

        feasible[0, 0] = 1
        feasible[1:, 0] = 1
        feasible[0, 1:] = 1

        max_total_importance = -np.inf
        best_k, best_t = 0, 0

        for k in range(1, layer_limit + 1):
            if verbose:
                print(f"DP: processing layer {k}/{layer_limit}", end="\r")
            for time_used in range(time_limit + 1):
                # Case 1: Don't select current layer (k - 1)
                best_importance = dp_table[k - 1, time_used]
                prev_k_opt, prev_t_opt = -1, -1

                # Case 2: Try selecting current layer (k - 1) if allowed
                if selectable_mask[k - 1]:
                    remaining_time = time_used - t_dw[k - 1]
                    for prev_k in range(k - 1, -1, -1):
                        remaining_time -= t_dy[prev_k]
                        if remaining_time < 0:
                            break
                        if feasible[prev_k, remaining_time]:
                            candidate_importance = dp_table[prev_k, remaining_time] + importances[k - 1]
                            if candidate_importance > best_importance:
                                best_importance = candidate_importance
                                prev_k_opt, prev_t_opt = prev_k, remaining_time

                if prev_k_opt >= 0:
                    # Found a better option by selecting layer k - 1
                    dp_table[k, time_used] = best_importance
                    selection_table[k, time_used, : k - 1] = selection_table[prev_k_opt, prev_t_opt, : k - 1]
                    selection_table[k, time_used, k - 1] = 1
                    feasible[k, time_used] = 1
                else:
                    # Stick with not selecting it
                    dp_table[k, time_used] = dp_table[k - 1, time_used]
                    selection_table[k, time_used, : k - 1] = selection_table[k - 1, time_used, : k - 1]
                    selection_table[k, time_used, k - 1] = 0
                    feasible[k, time_used] = 0

                # Track best global importance
                if dp_table[k, time_used] > max_total_importance:
                    max_total_importance = dp_table[k, time_used]
                    best_k, best_t = k, time_used

        selected_layers_mask = selection_table[best_k, best_t, :]
        return max_total_importance, selected_layers_mask
