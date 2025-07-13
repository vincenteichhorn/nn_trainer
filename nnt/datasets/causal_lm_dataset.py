from abc import abstractmethod
import copy
from typing import Any, Dict, List, Union
from nnt.datasets.dataset import DataSplit, Dataset
from nnt.util.monitor import Monitor
from transformers import PreTrainedTokenizer


class LMConversation:
    """
    Represents a chat-style conversation for language modeling tasks.
    Stores turns, input IDs, and labels, and provides methods for formatting and tokenization.

    Example:
        conv = LMConversation()
        conv.add_turn("user", "Hello").add_turn("assistant", "Hi!")
        print(conv.chat)
    """

    chat: list
    input_ids: list
    labels: list

    def __init__(self):
        """
        Initialize the LMConversation with empty chat, input_ids, and labels.
        """
        self.chat = []
        self.input_ids = []
        self.labels = []

    def to_dict(self) -> Dict[str, str]:
        """
        Convert the conversation to a dictionary format.

        Returns:
            dict: The chat as a dictionary.
        """
        return self.chat

    def to_sample(self) -> Dict[str, Union[List[int], List[str]]]:
        """
        Convert the conversation to a sample format with input_ids and labels.

        Returns:
            dict: Dictionary containing input_ids, labels, and chat.
        """
        return {"input_ids": self.input_ids, "labels": self.labels, "chat": self.chat}

    def add_turn(self, role: str, content: str) -> "LMConversation":
        """
        Add a turn to the conversation.

        Args:
            role (str): The role of the speaker (e.g., "user", "assistant").
            content (str): The content of the turn.
        Returns:
            LMConversation: The updated conversation.
        """
        self.chat.append({"role": role, "content": content})
        return self

    def apply_chat_template_and_tokenize(
        self,
        tokenizer: PreTrainedTokenizer,
        assistant_labels_only: bool = True,
        loss_mask_token: int = -100,
    ) -> None:
        """
        Apply a chat template and tokenize the conversation using the provided tokenizer.
        Optionally mask non-assistant tokens for loss computation.

        Args:
            tokenizer: Tokenizer object with chat_template support.
            assistant_labels_only (bool): If True, mask non-assistant tokens.
            loss_mask_token (int): Token to use for masking.
        """
        assert all("role" in el and "content" in el for el in self.chat), "Chat must have 'role' and 'content' keys."

        def maybe_apply_chat_template(chat, add_generation_prompt, continue_final_message):
            if tokenizer.chat_template is not None:
                text = tokenizer.apply_chat_template(
                    chat,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                )
            else:
                text = "\n".join([f"{el['role']}:\n{el['content']}" for el in chat])
                text = text + "\nassistant:\n" if add_generation_prompt else text

            tokens = tokenizer([text])["input_ids"][0]
            return tokens

        done_input_ids = []
        done_chat = []
        assistant_mask = []
        for i, turn in enumerate(self.chat):
            is_last = i == len(self.chat) - 1
            is_user = turn["role"] == "user"
            input_ids_w_turn = maybe_apply_chat_template(
                done_chat + [turn],
                add_generation_prompt=is_user and is_last,
                continue_final_message=not is_last,
            )
            num_added_tokens = len(input_ids_w_turn) - len(done_input_ids)
            assistant_mask.extend([0] * num_added_tokens if is_user else [1] * num_added_tokens)
            done_input_ids = input_ids_w_turn
            done_chat.append(turn)

        input_ids = done_input_ids
        labels = copy.deepcopy(input_ids)
        if assistant_labels_only:
            labels = [loss_mask_token if mask == 0 else label for mask, label in zip(assistant_mask, labels)]

        self.input_ids = input_ids
        self.labels = labels

    def __repr__(self) -> str:
        """
        String representation of the LMConversation.
        """
        return f"LMConversation({self.chat})"


class CausalLMDataset(Dataset):
    """
    Dataset class for causal language modeling tasks with chat-style formatting and tokenization.

    Args:
        assisten_labels_only (bool): If True, mask non-assistant tokens for loss.
        verbose (bool): If True, print formatting progress.

    Example:
        ds = CausalLMDataset()
        ds.prepare(tokenizer)
        print(ds['train'][0])
    """

    assistant_labels_only: bool
    loss_mask_token: int
    verbose: bool

    def __init__(self, assisten_labels_only: bool = True, verbose: bool = True, *args, **kwargs):
        """
        Initialize the CausalLMDataset.
        """
        self.assistant_labels_only = assisten_labels_only
        self.loss_mask_token = -100
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def prepare(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Format and tokenize the provided examples for all splits.

        Args:
            tokenizer: Tokenizer object for formatting and tokenization.
        Returns:
            None
        """
        self.load_if_not_loaded()
        monitor = Monitor()
        with monitor.tqdm(
            desc="Formatting", disable=not self.verbose, total=sum(len(split) for split in self.splits().values())
        ) as pbar:
            for split in self.splits().keys():
                for i, sample in monitor.tqdm(
                    enumerate(self[split]), desc=f"{split.title()} split", total=len(self[split]), disable=not self.verbose
                ):
                    conversation = self.build_chat(sample, split_name=split)
                    conversation.apply_chat_template_and_tokenize(
                        tokenizer=tokenizer,
                        assistant_labels_only=self.assistant_labels_only,
                        loss_mask_token=self.loss_mask_token,
                    )
                    self[split][i] = {**sample, **conversation.to_sample()}
                    pbar.update(1)

                if self.verbose:
                    monitor.print(f"{split} sample:")
                    input_ids = self[split][0]["input_ids"]
                    # monitor.print(f"input_ids:\n| {input_ids}")
                    # single_tokens = [tokenizer.decode(x) for x in input_ids]
                    # monitor.print(f"single tokens:\n| {single_tokens}")
                    text = tokenizer.decode(input_ids)
                    # monitor.print("text:")
                    for ln in text.split("\n"):
                        monitor.print(f"| {ln}")

    @abstractmethod
    def build_chat(sample: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from an example for a given split.

        Args:
            sample (dict): The example to build the chat from.
            split_name (str): The name of the split.
        Returns:
            LMConversation: The chat built from the example.
        Example:
            return LMConversation().add_turn("user", "Hello").add_turn("assistant", "Hi there!")
        """
        raise NotImplementedError
