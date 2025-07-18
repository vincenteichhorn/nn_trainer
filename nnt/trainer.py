from dataclasses import dataclass
import inspect
import os
from typing import List, Literal
import numpy as np
import torch
from torch.utils.data import DataLoader

from nnt.callbacks.trainer_callback import TrainerCallback
from nnt.collators.data_collators import PlainDataCollator
from nnt.datasets.dataset import DataSplit
from nnt.util.functions import get_current_time
from nnt.util.monitor import Monitor


@dataclass
class TrainingArguments:
    """
    Data class for storing training configuration arguments.

    Attributes:
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size for training.
        data_collator (callable, optional): Function to collate data batches.
        learning_rate (float): Learning rate for optimizer.
        weight_decay (float): Weight decay for optimizer.
        monitor_strategy (Literal["steps", "epochs"]): Strategy for monitoring training progress.
        monitor_every (int): Frequency of monitoring.
        checkpoint_strategy (Literal["steps", "epochs"]): Strategy for saving checkpoints.
        checkpoint_every (int): Frequency of checkpointing.
        model_save_function (callable, optional): Custom function to save the model.
    """

    num_epochs: int
    batch_size: int
    data_collator: callable = None
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    monitor_strategy: Literal["steps", "epochs"] = "steps"
    monitor_every: int = 1000
    checkpoint_strategy: Literal["steps", "epochs"] = "steps"
    checkpoint_every: int = 1000
    model_save_function: callable = None


class Trainer:
    """
    Trainer class for managing the training loop, callbacks, and model saving.

    Args:
        output_dir (str): Directory to save outputs and checkpoints.
        model: PyTorch model to be trained.
        training_args (TrainingArguments): Training configuration arguments.
        train_data (DataSplit): Training dataset split.
        optimizer (torch.optim.Optimizer, optional): Optimizer for training.
        callbacks (List[TrainerCallback], optional): List of callback objects.
    """

    def __init__(
        self,
        output_dir: str,
        model,
        training_args: TrainingArguments,
        train_data: DataSplit,
        optimizer: torch.optim.Optimizer = None,
        callbacks: List[TrainerCallback] = [],
    ):
        """
        Initialize the Trainer object and set up directories, optimizer, and device.
        """
        self.output_dir = output_dir
        self.model = model
        self.training_args = training_args
        self.train_data = train_data
        self.callbacks = callbacks
        self.device = model.device if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optimizer or torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_args.learning_rate,
            weight_decay=self.training_args.weight_decay,
        )
        self.epoch_seed = 0
        self._should_stop = False
        if self.output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)

        self.model_input_parameter_names = list(inspect.signature(self.model.forward).parameters.keys())

    def _call_callbacks(self, name: str, info: dict):
        """
        Call the specified callback method for all registered callbacks.

        Args:
            name (str): Name of the callback method to call.
            info (dict): Information dictionary to pass to the callback.
        """
        for callback in self.callbacks:
            getattr(callback, name)(info, self)

    def _batch_to_device(self, batch):
        """
        Move batch tensors to the appropriate device (CPU or GPU).

        Args:
            batch (dict): Batch of data.
        Returns:
            dict: Batch with tensors moved to device.
        """
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    def _prepare_data(self):
        """
        Prepare the training data loader and set up the data collator.
        """

        if self.training_args.data_collator is None:
            self.training_args.data_collator = PlainDataCollator(input_variable_names=self.model_input_parameter_names)
        self.train_batches = DataLoader(
            self.train_data,
            batch_size=self.training_args.batch_size,
            collate_fn=self.training_args.data_collator,
            worker_init_fn=lambda id: np.random.seed(id + self.epoch_seed),
        )

    def _save_model(self, global_step: int, checkpiont=False):
        """
        Save the model state to disk, either as a checkpoint or final model.

        Args:
            global_step (int): Current global training step.
            checkpiont (bool): Whether to save as a checkpoint.
        """
        save_folder = (
            os.path.join(self.output_dir, "checkpoints", "checkpoint-" + str(global_step))
            if checkpiont
            else os.path.join(self.output_dir, "model")
        )
        os.makedirs(save_folder, exist_ok=True)
        if self.training_args.model_save_function is not None:
            self.training_args.model_save_function(self.model, save_folder)
        else:
            torch.save(self.model.state_dict(), os.path.join(save_folder, "model.pth"))

    def stop(self):
        """
        Signal the trainer to stop training early.
        """
        self._should_stop = True

    def train(self):
        """
        Run the main training loop, handling epochs, batches, callbacks, and model saving.

        Returns:
            model: The trained model.
        """

        self._prepare_data()

        self.model.train()

        num_train_steps = len(self.train_batches) * self.training_args.num_epochs
        global_step = 0
        current_epoch_floating = 0.0
        current_batch = None
        train_loss = None
        monitor_every = (
            self.training_args.monitor_every
            if self.training_args.monitor_strategy == "steps"
            else (len(self.train_batches) * self.training_args.monitor_every)
        )
        checkpoint_every = (
            self.training_args.checkpoint_every
            if self.training_args.checkpoint_strategy == "steps"
            else (len(self.train_batches) * self.training_args.checkpoint_every)
        )

        def _get_info():
            return {
                "epoch": current_epoch_floating,
                "global_step": global_step,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "timestamp": get_current_time(),
                "current_batch": current_batch,
                "train_loss": train_loss,
            }

        def _get_info_str():
            fixed_value_width = 8
            return " ".join(
                f"{k}: {str(v)[:fixed_value_width]:<{fixed_value_width}}"
                for k, v in _get_info().items()
                if k not in ["current_batch", "timestamp"]
            ).lstrip()

        self._call_callbacks("on_training_begin", _get_info())
        with Monitor().tqdm(total=num_train_steps, desc="Training") as pbar:
            for epoch_id in range(self.training_args.num_epochs):
                self.epoch_seed = epoch_id
                self._call_callbacks("on_epoch_begin", _get_info())
                for batch in Monitor().tqdm(
                    self.train_batches,
                    desc=f"Batches (Epoch {epoch_id + 1}/{self.training_args.num_epochs})",
                    total=len(self.train_batches),
                ):
                    self.model.train()
                    if self._should_stop:
                        self._call_callbacks("on_training_end", _get_info())
                        self._save_model(global_step)
                        return self.model
                    batch = self._batch_to_device(batch)
                    current_batch = batch
                    self._call_callbacks("on_step_begin", _get_info())
                    self.model.zero_grad()
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    train_loss = loss.item()
                    self.optimizer.step()
                    self._call_callbacks("on_step_end", _get_info())
                    pbar.update(1)
                    global_step += 1
                    current_epoch_floating += 1 / len(self.train_batches)
                    if global_step % monitor_every == 0 or global_step == 0:
                        Monitor().print(_get_info_str())
                    if global_step % checkpoint_every == 0 or global_step == 0:
                        self._save_model(global_step, checkpiont=True)
                        self._call_callbacks("on_checkpoint", _get_info())
                self._call_callbacks("on_epoch_end", _get_info())
        self._call_callbacks("on_training_end", _get_info())
        self._save_model(global_step)
        return self.model
