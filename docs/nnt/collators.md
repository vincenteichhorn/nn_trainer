Use `DataCollatorForCausalLM` for language modeling tasks needing padding and masked loss; use `PlainDataCollator` when inputs are already uniform in shape or padding is handled upstream.

# `PlainDataCollator`

This class handles simpler cases where inputs are already aligned or don’t require padding:

* **Input Variable Selection**: Filters input keys based on a predefined set.
* **Tensor Conversion and Stacking**: Converts values to tensors and stacks them without padding.

This is suitable for inputs like pre-padded embeddings or other fixed-size tensors.

---

# `DataCollatorForCausalLM`

This class prepares batches for training causal language models. It performs:

* **Tensor Conversion**: Converts `input_ids`, `attention_mask`, and optionally `labels` into PyTorch tensors.
* **Padding and Stacking**: Pads each tensor to the maximum sequence length in the batch (optionally rounded up to a multiple of a given value), then stacks them into a single batch tensor.
* **Label Handling**: If `labels` are not provided, it defaults them to `input_ids`. Padding tokens in `labels` are replaced with `-100` to be ignored in loss computation.

Key method:
`_stack_and_pad()` handles padding direction (left/right), padding values, and ensures uniform tensor sizes.

---
