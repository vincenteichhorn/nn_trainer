from typing import Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_model(
    model_name: str, tokenizer_name: str = None, device: str | None = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a base model and tokenizer from Hugging Face's model hub and move the model to the appropriate device.

    Args:
        model_name (str): The name of the model to load.
        tokenizer_name (str, optional): The name of the tokenizer to load. If None, uses model_name.
        device (str, optional): The device to load the model on (default: "cuda" if available, else "cpu").
    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer.

    Example:
        model, tokenizer = load_model("gpt2")
        print(model)
        print(tokenizer)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
    tokenizer.add_special_tokens({"pad_token": "<|reserved_special_token_0|>"})
    model.config.pad_token_id = tokenizer.pad_token_id
    # print(f"EOS token id: {tokenizer.eos_token_id}, PAD token id: {tokenizer.pad_token_id}")
    return model, tokenizer
