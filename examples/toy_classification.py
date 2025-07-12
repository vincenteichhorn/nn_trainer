from nnt.callbacks.logging_callback import LoggingCallback
from nnt.callbacks.validator_callback import ValidatorCallback
from nnt.datasets.toy_dataset import ToyClassificationDataset
from nnt.models.toy_models import ToyClassificationModel
from nnt.trainer import Trainer, TrainingArguments
from nnt.validation_metrics.classification_metrics import OneHotClassificationMetrics
from nnt.validators.forward_validator import ForwardValidator
from nnt.validators.validator import ValidationArguments, Validator

if __name__ == "__main__":

    num_classes = 2

    dataset = ToyClassificationDataset(input_size=10, output_size=num_classes, num_samples=1000)
    model = ToyClassificationModel(
        input_size=10,
        hidden_size=20,
        output_size=num_classes,
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
    )

    metrics = [OneHotClassificationMetrics(num_classes=num_classes, targets_key="y", logits_key="logits")]
    validator = ForwardValidator(
        model=model,
        validation_args=ValidationArguments(batch_size=32),
        validation_data=dataset["validation"],
        metrics=metrics,
    )

    output_dir = "./out/toy_classification_model"
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
                validator=validator,
                validate_strategy="steps",
                validate_every=500,
            ),
        ],
    )
    trainer.train()
