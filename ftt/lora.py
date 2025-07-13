import numpy as np
import torch
import torch.nn as nn
import os
from typing import Tuple
import warnings
from huggingface_hub import ModelHubMixin
from transformers import AutoModelForCausalLM, AutoTokenizer

from ftt.lora_strategies import LoRAConfigLayerStrategy, LoRALayerStrategy
from nnt.util.functions import load_json, save_json


class LoRALayer(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout

        stdev = 1.0 / torch.sqrt(torch.tensor(rank, dtype=dtype))
        self.A = nn.Parameter(
            torch.randn(in_features, rank, dtype=dtype, device=device) * stdev,
            requires_grad=True,
        )
        self.B = nn.Parameter(
            torch.zeros(rank, out_features, dtype=dtype, device=device),
            requires_grad=True,
        )
        self.dropout_layer = nn.Dropout(dropout)
        self.phi = self.alpha / np.sqrt(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.dropout_layer(x) @ self.A) @ self.B * self.phi


class LinearLoRA(nn.Linear):

    def __init__(self, rank: int, alpha: float, dropout: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            device=self.weight.device,
            dtype=self.weight.dtype,
        )
        self.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + self.lora(x)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int, alpha: float, dropout: float, *args, **kwargs) -> "LinearLoRA":
        linear_lora = cls(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            *args,
            **kwargs,
        )
        linear_lora.weight = linear.weight
        linear_lora.weight.requires_grad = False
        if linear.bias is not None:
            linear_lora.bias = linear.bias
            linear_lora.bias.requires_grad = False
        return linear_lora


class LoRAModelAdapterSaveMixin(ModelHubMixin):

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id,
        revision,
        cache_dir,
        force_download,
        proxies,
        resume_download,
        local_files_only,
        token,
        **model_kwargs,
    ):
        adapter_config = load_json(os.path.join(model_id, "adapter_config.json"))
        base_model_name = adapter_config["base_model_name"]
        model_kwargs["base_model_name"] = base_model_name
        model_kwargs["layer_strategy"] = LoRAConfigLayerStrategy(adapter_config)
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            adapter = torch.load(os.path.join(model_id, "adapter.pth"))
            model.load_adapter(adapter)
        else:
            raise ValueError(f"Model {model_id} not available locally.")
        return model

    def save_pretrained(self, save_directory: str):
        adapter = self.get_adapter()
        os.makedirs(save_directory, exist_ok=True)
        torch.save(adapter, os.path.join(save_directory, "adapter.pth"))
        save_json(
            self.get_adapter_config(),
            os.path.join(save_directory, "adapter_config.json"),
        )


class LoRAModel(nn.DataParallel, LoRAModelAdapterSaveMixin):
    def __init__(
        self,
        module: nn.Module = None,
        layer_strategy: LoRALayerStrategy = None,
        base_model_name: str = "",
        **kwargs,
    ):
        if module is None:
            module = AutoModelForCausalLM.from_pretrained(base_model_name, **kwargs)
        super().__init__(module)
        self.base_model_name = base_model_name
        self.layer_strategy = layer_strategy

        if layer_strategy is not None:
            self.configure_adapter()
        self.forward = self.module.forward

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise AttributeError
            return getattr(self.module, name)

    def __repr__(self):
        return f"LoRAModel({'base_model_name=' + self.base_model_name if self.base_model_name else ''}, layer_strategy={self.layer_strategy}, model={self.module})"

    def configure_adapter(self):
        for name, module in self.module.named_modules():
            module.requires_grad_(False)
            parent_name, child_name = name.rsplit(".", 1) if "." in name else (None, name)
            parent_module = self.module.get_submodule(parent_name) if parent_name else self.module
            if self.layer_strategy.should_layer_apply(module, name) and isinstance(module, nn.Linear):
                rank = self.layer_strategy.get_layer_rank(module, name)
                if rank <= 0:
                    warnings.warn(f"Rank for layer {name} is {rank}, skipping adaptation.")
                    continue
                alpha = self.layer_strategy.get_layer_alpha(module, name)
                dropout = self.layer_strategy.get_layer_dropout(module, name)
                adapted_module = LinearLoRA.from_linear(
                    module,
                    rank,
                    alpha,
                    dropout,
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )
                setattr(parent_module, child_name, adapted_module)

    def get_adapter(self):
        return {name: mod for name, mod in self.module.state_dict().items() if "lora" in name}

    def get_adapter_config(self):
        return {
            "base_model_name": self.base_model_name,
            "layers": {
                name: {
                    "rank": self.layer_strategy.get_layer_rank(module, name),
                    "alpha": self.layer_strategy.get_layer_alpha(module, name),
                    "dropout": self.layer_strategy.get_layer_dropout(module, name),
                }
                for name, module in self.module.named_modules()
                if isinstance(module, LinearLoRA)
            },
        }

    def load_adapter(self, adapter):
        self.module.load_state_dict(adapter, strict=False)


def load_model(
    model_name: str, tokenizer_name: str = None, device: str = "cuda"
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model from Hugging Face's model hub.
    Args:
        model_name (str): The name of the model to load.
        tokenizer_name (str): The name of the tokenizer to load.
        device (str): The device to load the model on (default: "cuda").
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.
    """
    if os.path.isdir(model_name) and os.path.isfile(os.path.join(model_name, "adapter.pth")):
        model = LoRAModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer
