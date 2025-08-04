Here's a detailed tutorial explaining how to use the provided code for training and validating a toy language model on the MRPC task from GLUE with both classification and generation validations, including resource monitoring:

---

# Tutorial: Training a Toy Language Model with Classification and Generation Validation on GLUE MRPC

This tutorial demonstrates how to train a toy language model using the GLUE MRPC dataset with both **classification** and **generation** validation metrics, employing a flexible `Trainer` and several useful callbacks.

---

## Step 1: Import Necessary Modules

```python
from nnt.callbacks.energy_callback import EnergyCallback
from nnt.callbacks.flops_budget_callback import FLOPsBudgetControllCallback
from nnt.callbacks.logging_callback import LoggingCallback
from nnt.callbacks.validator_callback import ValidatorCallback
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM
from nnt.datasets.causal_lm_dataset import AlpacaSmallDatasetTruncated, GlueDatasets
from nnt.models.toy_models import ToyLanguageModel
from nnt.trainer import Trainer, TrainingArguments
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validation_metrics.generation_metrics import BleuScore, MeteorScore, NistScore, RougeScore
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.generation_validator import GenerationValidator
from nnt.validators.validator import ValidationArguments
from transformers import AutoTokenizer
```

---

## Step 2: Prepare Dataset and Tokenizer

* Load the MRPC dataset from GLUE with limited training size for faster experimentation.
* Initialize a tokenizer from a pretrained model checkpoint.
* Prepare dataset using the tokenizer to tokenize input sentences.

```python
dataset = GlueDatasets(verbose=True, task_name="mrpc", train_set_size=1000)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
dataset.prepare(tokenizer)
```

---

## Step 3: Initialize the Toy Language Model

* Define a small transformer-based language model.
* Parameters like vocabulary size, embedding dimension, number of layers, etc., are set.

```python
model = ToyLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embed_dim=4,
    max_seq_len=512,
    num_layers=1,
    num_heads=1,
    hidden_dim=4,
)
```

---

## Step 4: Setup Validators with Metrics

### Generation Validator

* Uses the `GenerationValidator` to generate text from the model and evaluate using generation metrics like BLEU, METEOR, NIST, and ROUGE.
* Uses a causal language modeling data collator for batching.
* The target is choosen arbitrarily as "sentence1" from the dataset. in the case of MRPC, this does not make sense, but it is used here for demonstration purposes.

```python
generation_validator = GenerationValidator(
    model=model,
    tokenizer=tokenizer,
    validation_args=ValidationArguments(batch_size=32, data_collator=DataCollatorForCausalLM(tokenizer)),
    validation_data=dataset["generation"],
    metrics=[
        BleuScore(target_key="sentence1"),
        NistScore(target_key="sentence1"),
        RougeScore(target_key="sentence1"),
        MeteorScore(target_key="sentence1"),
    ],
)
```

### Classification Validator

* Uses `ForwardValidator` to run a forward pass on validation data.
* Classification metric `OneHotClassificationMetrics` calculates accuracy, etc.
* Uses the same data collator to batch the data consistently.

```python
classes = tokenizer.convert_tokens_to_ids(dataset.get_task_classes())

forward_validator = ForwardValidator(
    model=model,
    validation_args=ValidationArguments(batch_size=32, data_collator=DataCollatorForCausalLM(tokenizer)),
    validation_data=dataset["validation"],
    metrics=[
        OneHotClassificationMetrics(num_classes=len(classes), classes=classes, targets_key="labels", logits_key="logits")
    ],
)
```

---

## Step 5: Define Training Arguments

* Training for 5 epochs with batch size 1 for simplicity.
* Learning rate, weight decay, monitoring and checkpoint intervals are configured.
* The data collator used during training matches the validators.

```python
training_args = TrainingArguments(
    num_epochs=5,
    batch_size=1,
    learning_rate=0.001,
    weight_decay=0.01,
    monitor_strategy="steps",
    monitor_every=1000,
    checkpoint_strategy="steps",
    checkpoint_every=1000,
    data_collator=DataCollatorForCausalLM(tokenizer),
)
```

---

## Step 6: Setup Callbacks for Training

* Logging progress to disk.
* Running validation for both classification and generation every 500 steps.
* Monitoring FLOPs budget to avoid excessive computation.
* Tracking energy consumption (requires NVIDIA GPU).

```python
output_dir = "./out/toy_language_model"

callbacks = [
    LoggingCallback(output_dir=output_dir),
    ValidatorCallback(
        output_dir=output_dir,
        log_file="validation_log.csv",
        validator=forward_validator,
        validate_strategy="steps",
        validate_every=500,
    ),
    ValidatorCallback(
        output_dir=output_dir,
        log_file="generation_validation_log.csv",
        validator=generation_validator,
        validate_strategy="steps",
        validate_every=500,
    ),
    FLOPsBudgetControllCallback(output_dir=output_dir, budget=1e9, should_stop_training=False),
    EnergyCallback(output_dir=output_dir, nvidia_query_interval=10),
]
```

---

## Step 7: Initialize Trainer and Start Training

* Create a `Trainer` instance with all components.
* Call `.train()` to begin training and periodically validate and log results.

```python
trainer = Trainer(
    output_dir=output_dir,
    model=model,
    training_args=training_args,
    train_data=dataset["train"],
    callbacks=callbacks,
)

trainer.train()
```

---

# Summary

* This setup demonstrates how to **train a toy language model** on GLUE's MRPC dataset.
* It performs **classification validation** and **generation validation** concurrently using dedicated validators.
* Training progress, checkpoints, FLOPs usage, and energy consumption are **monitored using callbacks**.
* The code uses a **flexible modular design**, allowing easy swapping of datasets, models, validators, and callbacks.

---

If you want me to help you customize the training pipeline, add your own dataset, or create custom callbacks, just ask!
