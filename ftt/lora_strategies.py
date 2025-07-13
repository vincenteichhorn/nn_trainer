from math import floor
import torch.nn as nn


class LoRALayerStrategy:
    """
    Base class for LoRA layer selection and configuration strategies.
    """

    def should_layer_apply(self, layer: nn.Module, name: str) -> bool:
        """
        Determines if LoRA should be applied to the given layer.

        Args:
            layer (nn.Module): The layer to check.
            name (str): The name of the layer.

        Returns:
            bool: True if LoRA should be applied, False otherwise.
        """
        raise NotImplementedError

    def get_layer_rank(self, layer: nn.Module, name: str) -> int:
        """
        Returns the rank for the given layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            int: The rank value.
        """
        raise NotImplementedError

    def get_layer_dropout(self, layer: nn.Module, name: str) -> float:
        """
        Returns the dropout rate for the given layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The dropout rate.
        """
        raise NotImplementedError

    def get_layer_alpha(self, layer: nn.Module, name: str) -> float:
        """
        Returns the alpha scaling factor for the given layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The alpha value.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Returns a string representation of the strategy.

        Returns:
            str: The class name.
        """
        return f"{self.__class__.__name__}()"


class LoRAConfigLayerStrategy(LoRALayerStrategy):
    """
    LoRA strategy based on a configuration dictionary specifying per-layer settings.
    """

    def __init__(self, config: dict):
        """
        Args:
            config (dict): Configuration dictionary with layer settings.
        """
        self.config = config
        super().__init__()

    def should_layer_apply(self, layer: nn.Module, name: str) -> bool:
        """
        Checks if LoRA should be applied to the layer based on config.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            bool: True if LoRA should be applied, False otherwise.
        """
        return name in self.config["layers"]

    def get_layer_rank(self, layer: nn.Module, name: str) -> int:
        """
        Gets the rank for the layer from config.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            int: The rank value.
        """
        return self.config["layers"][name]["rank"]

    def get_layer_dropout(self, layer: nn.Module, name: str) -> float:
        """
        Gets the dropout rate for the layer from config.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The dropout rate.
        """
        return self.config["layers"][name]["dropout"]

    def get_layer_alpha(self, layer: nn.Module, name: str) -> float:
        """
        Gets the alpha scaling factor for the layer from config.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The alpha value.
        """
        return self.config["layers"][name]["alpha"]

    def __repr__(self) -> str:
        """
        Returns a string representation of the strategy.

        Returns:
            str: The class name.
        """
        return f"{self.__class__.__name__}()"


class LoRAUniformStrategy(LoRALayerStrategy):
    """
    A strategy for applying LoRA (Low-Rank Adaptation) uniformly across layers in a neural network.

    This strategy applies the same rank, dropout, and alpha parameters to all layers whose names contain
    "att" (attention) or "mlp" (multi-layer perceptron).

    Args:
        rank (int): The rank parameter for LoRA layers.
        dropout (float): The dropout rate to apply to LoRA layers.
        alpha (float): The scaling factor (alpha) for LoRA layers.
    """

    def __init__(self, rank: int, dropout: float, alpha: float):
        self.rank = rank
        self.dropout = dropout
        self.alpha = alpha
        super().__init__()

    def should_layer_apply(self, layer: nn.Module, name: str):
        return "att" in name or "mlp" in name

    def get_layer_rank(self, layer: nn.Module, name: str):
        return self.rank

    def get_layer_dropout(self, layer: nn.Module, name: str):
        return self.dropout

    def get_layer_alpha(self, layer: nn.Module, name: str):
        return self.alpha

    def __repr__(self):
        return f"{self.__class__.__name__}(rank={self.rank}, dropout={self.dropout}, alpha={self.alpha})"


class LoRAPartialStrategy(LoRALayerStrategy):
    """
    LoRA strategy for applying LoRA to a subset of layers, either from the top or bottom.
    """

    def __init__(
        self,
        rank: int,
        dropout: float,
        alpha: float,
        num_total_layers: int,
        num_layers: int,
        begin_from: str = "top",
        layer_id_parse_rule=lambda name: int(name.split(".")[2]),
    ):
        """
        Args:
            rank (int): The rank of the LoRA layers.
            dropout (float): The dropout rate for the LoRA layers.
            alpha (float): The scaling factor for the LoRA layers.
            num_total_layers (int): The total number of layers in the model.
            num_layers (int): The number of layers to apply LoRA to.
            begin_from (str): The layer to start from ("top" or "bottom").
            layer_id_parse_rule (callable): Function to parse layer ID from name.
        """
        super().__init__()
        self.rank = rank
        self.dropout = dropout
        self.alpha = alpha
        self.num_total_layers = num_total_layers
        self.num_layers = num_layers
        self.begin_from = begin_from
        self.parse_layer_id = layer_id_parse_rule

    def should_layer_apply(self, layer: nn.Module, name: str) -> bool:
        """
        Determines if LoRA should be applied to the layer based on position.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            bool: True if LoRA should be applied, False otherwise.
        """
        if "self_attn" not in name and "mlp" not in name:
            return False
        layer_id = self.parse_layer_id(name)
        if self.begin_from == "top" and layer_id >= self.num_total_layers - self.num_layers:
            return True
        elif self.begin_from == "bottom" and layer_id < self.num_layers:
            return True
        return False

    def get_layer_rank(self, layer: nn.Module, name: str) -> int:
        """
        Returns the rank for the layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            int: The rank value.
        """
        return self.rank

    def get_layer_dropout(self, layer: nn.Module, name: str) -> float:
        """
        Returns the dropout rate for the layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The dropout rate.
        """
        return self.dropout

    def get_layer_alpha(self, layer: nn.Module, name: str) -> float:
        """
        Returns the alpha scaling factor for the layer.

        Args:
            layer (nn.Module): The layer.
            name (str): The name of the layer.

        Returns:
            float: The alpha value.
        """
        return self.alpha

    def __repr__(self) -> str:
        """
        Returns a string representation of the strategy.

        Returns:
            str: The class name and parameters.
        """
        return f"{self.__class__.__name__}(rank={self.rank}, dropout={self.dropout}, alpha={self.alpha}, num_total_layers={self.num_total_layers}, num_layers={self.num_layers}, begin_from={self.begin_from})"
