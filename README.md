# NN Trainer

A modular framework for training neural networks with PyTorch, supporting advanced profiling, validation, metrics, and flexible dataset handling.  
Includes utilities for training large language models (LLMs) such as Meta-Llama-3.2-1B with custom datasets.

---

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.

1. **Install Poetry (if not already installed):**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone the repository:**
   ```bash
   git clone https://github.com/vincenteichhorn/nn_trainer.git
   cd nn_trainer
   ```

3. **Install dependencies:**
   ```bash
   poetry install
   ```

4. **Activate the virtual environment:**
   ```bash
   poetry shell
   ```

---

## Getting Started

### Example: Training Meta-Llama-3.2-1B on FLAN-T5 Instruction Data

#### 1. Load the Model and Tokenizer

```python
from nnt.models.load_models import load_model

model_name = "meta-llama/Meta-Llama-3-2-1b"
model, tokenizer = load_model(model_name)
```

#### 2. Create a Custom Dataset

Subclass `CausalLMDataset` to load FLAN-T5 instruction data from HuggingFace:

```python
from nnt.datasets.causal_lm_dataset import CausalLMDataset, LMConversation
from datasets import load_dataset

class FlanT5InstructionDataset(CausalLMDataset):
    def load(self):
        ds = load_dataset("SirNeural/flan_v2")
        self["train"] = ds["train"]

    def build_chat(self, sample, split_name):
        # Example: treat 'inputs' as user prompt, 'targets' as assistant response
        conversation = LMConversation()
        conversation.add_turn("user", sample["inputs"])
        conversation.add_turn("assistant", sample["targets"])
        return conversation
```

#### 3. Prepare the Dataset

```python
dataset = FlanT5InstructionDataset()
dataset.prepare(tokenizer)
```

#### 4. Train the Model

You can now use the provided `Trainer` class (or your own training loop) to train the model:

```python
from nnt.trainer import Trainer

trainer = Trainer(
    model=model,
    train_data=dataset["train"],
    validation_data=dataset["validation"],
    tokenizer=tokenizer,
    # ...other arguments...
)
trainer.train()
```

---

## Examples

You can find practical usage and advanced scenarios in the [`examples/`](examples/) folder.

### Example Scripts

- **`examples/toy_language_model.py`**  
  Train a toy transformer-based language model on a GLUE task (e.g., MRPC) using the Meta-Llama tokenizer and advanced validation, profiling, and logging callbacks.

- **`examples/toy_classification.py`**  
  Train a simple feedforward classification model on a synthetic dataset with validation, FLOPs/energy tracking, and logging.

#### Running an Example

Activate your Poetry environment and run an example script:

```bash
source $(poetry env info --path)/bin/activate
python examples/toy_language_model.py
```

or

```bash
source $(poetry env info --path)/bin/activate
python examples/toy_classification.py
```

Explore the folder for more scripts demonstrating various features of NN Trainer.

---

## Features

- **Flexible Dataset API:** Easily create custom datasets by subclassing `CausalLMDataset`.
- **Profiling:** Integrated GPU/CPU profiling and FLOPs/energy tracking.
- **Validation & Metrics:** Modular validation and metric computation for classification and generation tasks.
- **Callbacks:** Logging, validation, FLOPs budget, and energy tracking callbacks.
- **Collators:** Custom collators for batching and padding data.

---

## Advanced Usage

- **Profiling:** Use `TorchProfiler` and `NvidiaProfiler` for detailed resource analysis.
- **Custom Metrics:** Extend `ValidationMetric` for your own evaluation logic.
- **Custom Callbacks:** Implement `TrainerCallback` for custom training hooks.

---

## References

- [Meta-Llama-3-2-1b](https://huggingface.co/meta-llama/Meta-Llama-3-2-1b)
- [FLAN-T5 Instruction Data](https://huggingface.co/datasets/SirNeural/flan_v2)
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/index)
- [Datasets](https://huggingface.co/docs/datasets/index)
- [Evaluation Metrics](https://huggingface.co/docs/evaluate/index)
