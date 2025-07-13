from nnt.callbacks.energy_callback import EnergyCallback
from nnt.callbacks.flops_budget_callback import FLOPsBudgetControllCallback
from nnt.callbacks.logging_callback import LoggingCallback
from nnt.callbacks.validator_callback import ValidatorCallback
from nnt.collators.causal_lm_data_collators import DataCollatorForCausalLM
from ftt.datasets import GlueDatasets
from nnt.models.toy_models import ToyLanguageModel
from nnt.trainer import Trainer, TrainingArguments
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validation_metrics.generation_metrics import BleuScore, MeteorScore, NistScore, RougeScore
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.generation_validator import GenerationValidator
from nnt.validators.validator import ValidationArguments
from transformers import AutoTokenizer

if __name__ == "__main__":

    dataset = GlueDatasets(verbose=True, task_name="mrpc", train_set_size=1000)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    dataset.prepare(tokenizer)

    model = ToyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embed_dim=4,
        max_seq_len=512,
        num_layers=1,
        num_heads=1,
        hidden_dim=4,
    )

    generation_validator = GenerationValidator(
        model=model,
        tokenizer=tokenizer,
        validation_args=ValidationArguments(batch_size=32, data_collator=DataCollatorForCausalLM(tokenizer)),
        validation_data=dataset["validation"],
        metrics=[
            BleuScore(target_key="sentence1"),
            NistScore(target_key="sentence1"),
            RougeScore(target_key="sentence1"),
            MeteorScore(target_key="sentence1"),
        ],
    )

    classes = tokenizer.convert_tokens_to_ids(dataset.get_task_classes())
    forward_validator = ForwardValidator(
        model=model,
        validation_args=ValidationArguments(batch_size=32, data_collator=DataCollatorForCausalLM(tokenizer)),
        validation_data=dataset["validation"],
        metrics=[
            OneHotClassificationMetrics(num_classes=len(classes), classes=classes, targets_key="labels", logits_key="logits")
        ],
    )

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

    output_dir = "./out/toy_language_model"
    trainer = Trainer(
        output_dir=output_dir,
        model=model,
        training_args=training_args,
        train_data=dataset["train"],
        callbacks=[
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
        ],
    )
    trainer.train()
