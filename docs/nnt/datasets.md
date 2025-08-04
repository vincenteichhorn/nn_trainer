# Datasets

This guide explains the internal design of the dataset framework and provides a step-by-step tutorial on how to create your own dataset by subclassing the base classes.
---

## Overview: Dataset Structure

The codebase is built on two core classes:

* `DataSplit`: Represents a list of dictionary samples (i.e., rows).
* `Dataset`: Holds multiple named `DataSplit` instances such as `train`, `validation`, `test`.

For Causal Language Modeling:

* `CausalLMDataset` is a specialized `Dataset` class with:

  * Tokenization logic.
  * Support for "chat-style" dialogue using `LMConversation`.

---

## 📦 Base Classes

### `DataSplit`

Handles operations on a single split (`train`, `validation`, etc.).

```python
DataSplit([
    {"input": "Hello", "output": "Hi"},
    {"input": "What's your name?", "output": "I'm GPT."}
])
```

**Key Features:**

* Acts like a list of dictionaries.
* Supports unifying column structure.
* Loadable from JSON, pandas, dict, or iterable.

---

### `Dataset`

Holds named `DataSplit` instances.

```python
Dataset(train=[...], validation=[...])
```

**Abstract Method:**

```python
@abstractmethod
def load(self):
    pass  # You implement this in subclasses to populate splits
```

---

### `CausalLMDataset`

Extends `Dataset` for language modeling tasks.

* Adds `prepare(tokenizer)` to tokenize data using a `chat_template`.
* Requires `build_chat()` method to convert raw examples to an `LMConversation`.

---

## 🧱 LMConversation Class

Used to structure chat-like text into model input/label pairs.

```python
conv = LMConversation()
conv.add_turn("user", "Hello").add_turn("assistant", "Hi!")
```

---

## Example: Alpaca Dataset

```python
class AlpacaDataset(CausalLMDataset):

    def load(self):
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        self["train"] = DataSplit.from_iterable(...)
        self["validation"] = DataSplit.from_iterable(...)

    def build_chat(self, sample, split_name):
        return LMConversation() \
            .add_turn("user", f"{sample['instruction']} {sample['input']}") \
            .add_turn("assistant", sample["output"])
```

---

## How to Create a Custom Causal LM Dataset

### Subclass `CausalLMDataset`

```python
class MyDataset(CausalLMDataset):
    def load(self):
        raw_data = load_dataset("my_dataset_name")
        self["train"] = DataSplit.from_iterable(raw_data["train"])
        self["validation"] = DataSplit.from_iterable(raw_data["validation"])

    def build_chat(self, sample, split_name):
        conversation = LMConversation()
        conversation.add_turn("user", sample["question"])
        conversation.add_turn("assistant", sample["answer"])
        return conversation
```

### Tokenize Dataset

```python
dataset = MyDataset()
dataset.prepare(tokenizer)
```

This will:

* Build structured `chat` turns.
* Use `tokenizer.apply_chat_template()` if available.
* Add `input_ids` and `labels` for training.

---

## Tokenization Logic

The `prepare()` method works as follows:

1. For each example, it builds a chat (`LMConversation`) using `build_chat`.
2. Applies a chat template (if supported by the tokenizer).
3. Masks non-assistant tokens using `loss_mask_token = -100` if `assistant_labels_only=True`.

---

## Example: Toy Classification Dataset (non-LLM)

```python
class ToyClassificationDataset(Dataset):
    def load(self):
        def get_sample():
            return {"x": [...], "y": [...]}  # random data
        self["train"] = DataSplit([get_sample() for _ in range(1000)])
        self["validation"] = DataSplit([get_sample() for _ in range(100)])
```

Use this pattern for synthetic or structured input/output tasks not involving text.
