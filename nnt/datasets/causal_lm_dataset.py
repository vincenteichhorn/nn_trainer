from abc import abstractmethod
import copy
from typing import Dict, List, Union
from nnt.datasets.dataset import DataSplit, Dataset
from nnt.util.monitor import Monitor


class LMConversation:

    def __init__(self):
        self.chat = []
        self.input_ids = []
        self.labels = []

    def to_dict(self) -> Dict[str, str]:
        return self.chat

    def to_sample(self) -> Dict[str, Union[List[int], List[str]]]:
        """
        Convert the conversation to a sample format.
        Returns:
            Dict: A dictionary containing input_ids and labels.
        """
        return {"input_ids": self.input_ids, "labels": self.labels, "chat": self.chat}

    def add_turn(self, role: str, content: str):
        """
        Add a turn to the conversation.
        Args:
            role (str): The role of the speaker (e.g., "user", "assistant").
            content (str): The content of the turn.
        """
        self.chat.append({"role": role, "content": content})
        return self

    def apply_chat_template_and_tokenize(self, tokenizer, assistant_labels_only: bool = True, loss_mask_token: int = -100):

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

    def __repr__(self):
        return f"LMConversation({self.chat})"


class CausalLMDataset(Dataset):

    def __init__(self, assisten_labels_only: bool = True, verbose: bool = False, *args, **kwargs):
        self.assistant_labels_only = assisten_labels_only
        self.loss_mask_token = -100
        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def prepare(self, tokenizer):
        """
        Format and tokenize the provided examples.
        Args:
            examples (datasets.Dataset): The examples to format and tokenize.
            split (str): The split where the examples are from.
        Returns:
            DataSplit with two new columns: "input_ids" and "labels".
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
    def build_chat(sample, split_name) -> LMConversation:
        """
        build chat from example
        Args:
            example (Dict): The example to build the chat from.
            split_name (str): The name of the split.
        Returns:
            LMConversation: The chat built from the example.
        Example:
            return LMConversation().add_turn("user", "Hello").add_turn("assistant", "Hi there!")
        """
        raise NotImplementedError


from datasets import load_dataset


class AlpacaDataset(CausalLMDataset):

    def load(self):
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        validation_size = len(alpaca_ds["train"]) // 10
        self["train"] = DataSplit.from_iterable(alpaca_ds["train"].select(range(len(alpaca_ds["train"]) - validation_size)))
        self["validation"] = DataSplit.from_iterable(
            alpaca_ds["train"].select(range(len(alpaca_ds["train"]) - validation_size, len(alpaca_ds["train"])))
        )

    def build_chat(self, sample, split_name) -> LMConversation:
        """
        Build chat from example.
        Args:
            sample (Dict): The example to build the chat from.
            split_name (str): The name of the split.
        Returns:
            LMConversation: The chat built from the example.
        """
        conversation = LMConversation()
        conversation.add_turn("user", f"{sample['instruction']} {sample['input']}")
        conversation.add_turn("assistant", sample["output"])
        return conversation


class AlpacaSmallDatasetTruncated(AlpacaDataset):

    def __init__(self, num_samples=1000, max_len=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.max_len = max_len

    def load(self):
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        validation_size = self.num_samples // 10
        self["train"] = DataSplit.from_iterable(alpaca_ds["train"].select(range(self.num_samples - validation_size)))
        self["validation"] = DataSplit.from_iterable(
            alpaca_ds["train"].select(range(self.num_samples - validation_size, self.num_samples))
        )

    def prepare(self, tokenizer):
        super().prepare(tokenizer)

        # iterate all splits and samples and trucate to 10 tokens
        for split in self.splits().keys():
            for i, sample in enumerate(self[split]):
                if len(sample["input_ids"]) > 10:
                    self[split][i]["input_ids"] = sample["input_ids"][-self.max_len :]
                    self[split][i]["labels"] = sample["labels"][-self.max_len :]


class GlueDatasets(CausalLMDataset):

    def __init__(self, task_name: str, train_set_size: Union[int, None] = None, *args, **kwargs):
        self.task_name = task_name
        self.train_set_size = train_set_size
        super().__init__(*args, **kwargs)
        assert task_name in self.available_tasks(), f"Task {task_name} not available."

    def available_tasks(self):
        return [
            "cola",
            "sst2",
            "mrpc",
            "qqp",
            "mnli_matched",
            "mnli_mismatched",
            "qnli",
            "rte",
            "wnli",
        ]

    def get_task_classes(self, task=None):
        if task is None:
            task = self.task_name

        if task == "cola":
            return ["False", "True"]
        elif task == "sst2":
            return ["Negative", "Positive"]
        elif task == "mrpc":
            return ["False", "True"]
        elif task == "mnli_matched":
            return ["False", "True", "Neither"]
        elif task == "mnli_mismatched":
            return ["False", "True", "Neither"]
        elif task == "qnli":
            return ["Yes", "No"]
        elif task == "rte":
            return ["True", "False"]
        elif task == "wnli":
            return ["False", "True"]
        elif task == "qqp":
            return ["No", "Yes"]

    def load(self):
        ds = load_dataset("nyu-mll/glue", self.task_name)
        if "mnli" in self.task_name:
            train_ds = load_dataset("nyu-mll/glue", "mnli")
            ds["train"] = train_ds["train"]
        self.train_set_size = self.train_set_size or len(ds["train"])
        self["train"] = DataSplit.from_iterable(ds["train"].select(range(self.train_set_size)))
        self["validation"] = DataSplit.from_iterable(ds["validation"])
        self["generation"] = DataSplit.from_iterable(ds["validation"])
        return ds

    def build_chat(self, example, split_name):
        joined_classes = ", ".join(self.get_task_classes())
        assistant_content = self.get_task_classes()[example["label"]]
        if self.task_name == "cola":
            user_content = f"{example['sentence']}\nQuestion: Does this sentence make sense? {joined_classes}?"
        elif self.task_name == "sst2":
            user_content = f"{example['sentence']}\nQuestion: Is this sentence positive or negative? {joined_classes}?"
        elif self.task_name == "mrpc":
            user_content = f"{example['sentence1']}\n{example['sentence2']}\nQuestion: Do both sentences mean the same thing? {joined_classes}?"
        elif self.task_name == "mnli_matched":
            user_content = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nQuestion: Does the premise entail the hypothesis? {joined_classes}?"
        elif self.task_name == "mnli_mismatched":
            user_content = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nQuestion: Does the premise entail the hypothesis? {joined_classes}?"
        elif self.task_name == "qnli":
            user_content = f"Question: {example['question']}\nResponse: {example['sentence']}\nQuestion: Does this response answer the question? {joined_classes}?"
        elif self.task_name == "rte":
            user_content = f"{example['sentence1']}\n{example['sentence2']}\nQuestion: Do both sentences mean the same thing? {joined_classes}?"
        elif self.task_name == "wnli":
            user_content = f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}\nQuestion: Does the premise entail the hypothesis? {joined_classes}?"
        elif self.task_name == "qqp":
            user_content = f"{example['question1']}\n{example['question2']}\nQuestion: Do both questions mean the same thing? {joined_classes}?"

        conversation = LMConversation()
        conversation.add_turn("user", user_content)
        if split_name in ["train", "validation"]:
            conversation.add_turn("assistant", assistant_content)
        return conversation
