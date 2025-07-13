import os
from typing import Any, Dict, Union, List, Optional
from datasets import load_dataset
import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizer
import urllib

from nnt.datasets.causal_lm_dataset import CausalLMDataset, LMConversation
from nnt.datasets.dataset import DataSplit
from nnt.util.functions import load_json


def get_dataset(name: str, *args, **kwargs) -> CausalLMDataset:
    """
    Returns a dataset object based on the provided name.

    Args:
        name (str): Name of the dataset.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Available datasets:
        - "alpaca_mmlu": Alpaca MMLU dataset.
        - "glue_<task_name>": GLUE benchmark tasks
            - "glue_cola": CoLA dataset.
            - "glue_sst2": SST-2 dataset.
            - "glue_mrpc": MRPC dataset.
            - "glue_qqp": QQP dataset.
            - "glue_mnli_matched": MNLI matched dataset.
            - "glue_mnli_mismatched": MNLI mismatched dataset.
            - "glue_qnli": QNLI dataset.
            - "glue_rte": RTE dataset.
        - "arc_<subset>": ARC dataset subsets
            - "arc_easy": ARC easy subset.
            - "arc_challenge": ARC challenge subset.
        - "boolq": BoolQ dataset.
        - "piqa": PIQA dataset.
        - "hellaswag": HellaSwag dataset.
        - "alanai_<name>": AllenAI Natural Instructions datasets (e.g., alanai_math).
            - name should be the file name without the .json extension from https://github.com/allenai/natural-instructions/tree/master/tasks

    Returns:
        CausalLMDataset: The dataset object.

    Raises:
        ValueError: If the dataset name is not recognized.
    """

    if name == "alpaca_mmlu":
        return AlpacaMMLUDataset(*args, **kwargs)
    elif name.startswith("glue_"):
        task_name = name.split("glue_")[1]
        return GlueDatasets(task_name, *args, **kwargs)
    elif name.startswith("arc_"):
        subset = name.split("arc_")[1]
        return ArcDataset(subset, *args, **kwargs)
    elif name == "boolq":
        return BoolQDataset(*args, **kwargs)
    elif name == "piqa":
        return PIQADataset(*args, **kwargs)
    elif name == "hellaswag":
        return HellaSwagDataset(*args, **kwargs)
    elif name.startswith("alanai_"):
        name = name.split("alanai_")[1]
        return AlanAIDataset(name, *args, **kwargs)
    else:
        raise ValueError(
            f"Dataset {name} not found. Available datasets: alpaca, alpaca_small, glue_<task_name>, arc_<subset>, piqa, hellaswag, alanai_<name>."
        )


class AlpacaDataset(CausalLMDataset):
    """
    Dataset class for the Alpaca instruction-following dataset.
    Splits data into train and validation sets and formats as chat conversations.
    """

    def load(self) -> None:
        """
        Load the Alpaca dataset and split into train and validation sets.
        """
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        validation_size = len(alpaca_ds["train"]) // 10
        self["train"] = DataSplit.from_iterable(alpaca_ds["train"].select(range(len(alpaca_ds["train"]) - validation_size)))
        self["validation"] = DataSplit.from_iterable(
            alpaca_ds["train"].select(range(len(alpaca_ds["train"]) - validation_size, len(alpaca_ds["train"])))
        )

    def build_chat(self, sample: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from an Alpaca example.

        Args:
            sample (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        conversation = LMConversation()
        conversation.add_turn("user", f"{sample['instruction']} {sample['input']}")
        conversation.add_turn("assistant", sample["output"])
        return conversation


class AlpacaSmallDatasetTruncated(AlpacaDataset):
    """
    Truncated version of the Alpaca dataset for quick experiments.
    Limits the number of samples and truncates tokenized sequences to max_len.
    """

    num_samples: int
    max_len: int

    def __init__(self, num_samples: int = 1000, max_len: int = 10, *args, **kwargs) -> None:
        """
        Initialize the truncated Alpaca dataset.

        Args:
            num_samples (int): Number of samples to use.
            max_len (int): Maximum length of tokenized sequences.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.num_samples = num_samples
        self.max_len = max_len

    def load(self) -> None:
        """
        Load and truncate the Alpaca dataset to num_samples and max_len.
        """
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        validation_size = self.num_samples // 10
        self["train"] = DataSplit.from_iterable(alpaca_ds["train"].select(range(self.num_samples - validation_size)))
        self["validation"] = DataSplit.from_iterable(
            alpaca_ds["train"].select(range(self.num_samples - validation_size, self.num_samples))
        )

    def prepare(self, tokenizer: PreTrainedTokenizer) -> None:
        """
        Format and tokenize, then truncate input_ids and labels to max_len for all splits.

        Args:
            tokenizer (PreTrainedTokenizer): Tokenizer to use.
        """
        super().prepare(tokenizer)

        # iterate all splits and samples and trucate to 10 tokens
        for split in self.splits().keys():
            for i, sample in enumerate(self[split]):
                if len(sample["input_ids"]) > 10:
                    self[split][i]["input_ids"] = sample["input_ids"][-self.max_len :]
                    self[split][i]["labels"] = sample["labels"][-self.max_len :]


class GlueDatasets(CausalLMDataset):
    """
    Dataset class for GLUE benchmark tasks, with chat-style formatting and tokenization.

    Args:
        task_name (str): Name of the GLUE task.
        train_set_size (int, optional): Number of training samples to use.
    """

    task_name: str
    train_set_size: Optional[int]

    def __init__(self, task_name: str, train_set_size: Optional[int] = None, *args, **kwargs) -> None:
        """
        Initialize the GlueDatasets for a specific GLUE task.

        Args:
            task_name (str): Name of the GLUE task.
            train_set_size (Optional[int]): Number of training samples to use.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.task_name = task_name
        self.train_set_size = train_set_size
        super().__init__(*args, **kwargs)
        assert task_name in self.available_tasks(), f"Task {task_name} not available."

    def available_tasks(self) -> List[str]:
        """
        Get the list of available GLUE tasks.

        Returns:
            List[str]: List of task names.
        """
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

    def get_task_classes(self, task: Optional[str] = None) -> List[str]:
        """
        Get the class labels for the specified GLUE task.

        Args:
            task (Optional[str]): Task name. Defaults to self.task_name.

        Returns:
            List[str]: List of class labels for the task.
        """
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

    def load(self) -> None:
        """
        Load the GLUE dataset for the specified task and split into train, validation, and generation sets.

        Returns:
            None
        """
        ds = load_dataset("nyu-mll/glue", self.task_name)
        if "mnli" in self.task_name:
            train_ds = load_dataset("nyu-mll/glue", "mnli")
            ds["train"] = train_ds["train"]
        self.train_set_size = self.train_set_size or len(ds["train"])
        self["train"] = DataSplit.from_iterable(ds["train"].select(range(self.train_set_size)))
        self["validation"] = DataSplit.from_iterable(ds["validation"])

    def build_chat(self, example: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from a GLUE example for a given split.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
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

        conversation = LMConversation().add_turn("user", user_content).add_turn("assistant", assistant_content)
        return conversation


class AlpacaMMLUDataset(CausalLMDataset):
    """
    Dataset class for combining Alpaca and MMLU datasets, supporting few-shot learning.
    """

    num_few_shot: int

    def __init__(self, num_few_shot: int = 0, *args, **kwargs) -> None:
        """
        Initialize the AlpacaMMLUDataset.

        Args:
            num_few_shot (int): Number of few-shot examples to use.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.num_few_shot = num_few_shot
        super().__init__(*args, **kwargs)

    def load(self) -> None:
        """
        Load the Alpaca and MMLU datasets, and prepare splits.

        Returns:
            None
        """
        alpaca_ds = load_dataset("tatsu-lab/alpaca")
        mmlu_ds = load_dataset("cais/mmlu", "all")

        mmlu_validation = pd.DataFrame(mmlu_ds["validation"])

        if self.num_few_shot > 0:
            few_shot_df = pd.DataFrame(mmlu_ds["dev"])
            few_shot_agg_df = few_shot_df.groupby("subject")[["question", "choices", "answer"]].apply(
                lambda x: "\n".join([self.mmlu_fewshot_example(row) for _, row in list(x.iterrows())[: self.num_few_shot]])
            )
            mmlu_validation = mmlu_validation.merge(few_shot_agg_df.reset_index(), on="subject", how="left").rename(
                columns={0: "few_shot"}
            )

        self["train"] = DataSplit.from_iterable(alpaca_ds["train"])
        self["validation"] = DataSplit.from_pandas(mmlu_validation)
        self["generation"] = DataSplit.from_pandas(mmlu_validation)

    def mmlu_prompt(self, example: Dict[str, Any]) -> str:
        """
        Build a prompt for an MMLU example.

        Args:
            example (Dict[str, Any]): The MMLU example.

        Returns:
            str: The formatted prompt.
        """
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{example['question']}\n{''.join([f'{alpha[i]}: {el}\n' for i, el in enumerate(example['choices'])])}"

    def mmlu_fewshot_example(self, example: Dict[str, Any]) -> str:
        """
        Build a few-shot example prompt for MMLU.

        Args:
            example (Dict[str, Any]): The MMLU example.

        Returns:
            str: The formatted few-shot example.
        """
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{self.mmlu_prompt(example)}\nassistant:\n{alpha[example['answer']]}\n"

    def build_chat(self, example: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from an Alpaca or MMLU example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        few_shot = f"{example['few_shot']}\n" if "few_shot" in example else ""
        if split_name == "train":
            conversation = (
                LMConversation()
                .add_turn("user", f"{example['instruction']} {example['input']}")
                .add_turn("assistant", example["output"])
            )
        elif split_name == "validation":
            conversation = (
                LMConversation()
                .add_turn("user", f"{few_shot}{self.mmlu_prompt(example)}")
                .add_turn("assistant", alpha[example["answer"]])
            )
        else:
            conversation = LMConversation().add_turn("user", f"{few_shot}{self.mmlu_prompt(example)}")
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for MMLU tasks.

        Returns:
            List[str]: List of class labels.
        """
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class ArcDataset(CausalLMDataset):
    """
    Dataset class for the ARC dataset, supporting both 'easy' and 'challenge' subsets.
    """

    subset: str

    def __init__(self, subset: str = "easy", *args, **kwargs) -> None:
        """
        Initialize the ArcDataset.

        Args:
            subset (str): Subset to use ('easy' or 'challenge').
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.subset = subset
        super().__init__(*args, **kwargs)
        assert subset in ["easy", "challenge"], f"Subset {subset} not available. Choose 'easy' or 'challenge'."

    def load(self) -> Any:
        """
        Load the ARC dataset and prepare splits.

        Returns:
            Any: The loaded dataset.
        """
        ds = load_dataset("allenai/ai2_arc", f"ARC-{self.subset.capitalize()}")
        self["train"] = DataSplit.from_iterable(ds["train"])
        self["validation"] = DataSplit.from_iterable(ds["validation"])
        return ds

    def build_chat(self, example: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from an ARC example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        passage = example["passage"]
        question = example["question"]
        answer = example["answer"]
        conversation = LMConversation().add_turn("user", f"{passage}\nQuestion: {question}?").add_turn("assistant", answer)
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for ARC tasks.

        Returns:
            List[str]: List of class labels.
        """
        return ["False", "True"]


class PIQADataset(CausalLMDataset):
    """
    Dataset class for the PIQA dataset.
    """

    def load(self) -> None:
        """
        Load the PIQA dataset and prepare splits.

        Returns:
            None
        """
        ds = load_dataset("nthngdy/piqa")
        self["train"] = DataSplit.from_iterable(ds["train"])
        self["validation"] = DataSplit.from_iterable(ds["validation"])

    def build_chat(self, example: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from a PIQA example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        question = example["goal"]
        options = "A: " + example["sol1"] + "\nB: " + example["sol2"] + "\n"
        answer = "A" if example["label"] == 0 else "B"
        conversation = LMConversation().add_turn("user", f"{question}\n{options}").add_turn("assistant", answer)
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for PIQA tasks.

        Returns:
            List[str]: List of class labels.
        """
        return ["A", "B"]


class HellaSwagDataset(CausalLMDataset):
    """
    Dataset class for the HellaSwag dataset.
    """

    def load(self) -> Any:
        """
        Load the HellaSwag dataset and prepare splits.

        Returns:
            Any: The loaded dataset.
        """
        ds = load_dataset("Rowan/hellaswag")
        self["train"] = DataSplit.from_iterable(ds["train"])
        self["validation"] = DataSplit.from_iterable(ds["validation"])
        return ds

    def spacing(self, context_b: str, ending: str) -> str:
        """
        Ensures that the context_b and ending are properly spaced.
        If context_b ends with a punctuation mark, it adds a space before the ending.

        Args:
            context_b (str): The context string.
            ending (str): The ending string.

        Returns:
            str: The spacing string.
        """
        if ending.startswith(","):
            return ""
        return " "

    def build_chat(self, example: Dict[str, Any], split_name: str) -> LMConversation:
        """
        Build a chat from a HellaSwag example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split_name (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        activity_label = example["activity_label"]
        context_a = example["ctx_a"]
        context_b = example["ctx_b"]
        context_b = context_b[0].upper() + context_b[1:] if context_b else ""
        alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        options = "".join(
            [f"{alpha[i]}: {context_b}{self.spacing(context_b, el)}{el}\n" for i, el in enumerate(example["endings"])]
        ).strip()
        answer = alpha[int(example["label"])]
        conversation = (
            LMConversation()
            .add_turn("user", f"{activity_label}\n{context_a}\nWhat happens next?\n{options}")
            .add_turn("assistant", answer)
        )
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for HellaSwag tasks.

        Returns:
            List[str]: List of class labels.
        """
        return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class AlanAIDataset(CausalLMDataset):
    """
    Dataset class for AllenAI Natural Instructions datasets.
    """

    name: str
    seed: int

    def __init__(self, name: str, seed: int = 42, *args, **kwargs) -> None:
        """
        Initialize the AlanAIDataset.

        Args:
            name (str): Name of the dataset.
            seed (int): Random seed for shuffling.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.seed = seed

    def load(self) -> None:
        """
        Load the AlanAI dataset and prepare splits.

        Returns:
            None
        """
        base_url = "https://raw.githubusercontent.com/allenai/natural-instructions/refs/heads/master/tasks/"
        ds_dir = os.path.join(os.environ.get("SHARED_DIR", ""), "datasets/alanai")
        if f"{self.name}.json" not in os.listdir(ds_dir):
            urllib.request.urlretrieve(
                f"{base_url}{self.name}.json",
                os.path.join(ds_dir, f"{self.name}.json"),
            )
        ds_json = load_json(os.path.join(ds_dir, f"{self.name}.json"))
        ds_df = pd.DataFrame(ds_json["Instances"])
        ds_df = ds_df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        ds_df["output"] = ds_df["output"].apply(lambda x: x[0] if isinstance(x, list) else x)
        train_validation_mask = np.random.rand(len(ds_df)) < 0.9
        train_data = ds_df[train_validation_mask]
        validation_data = ds_df[~train_validation_mask]
        self["train"] = DataSplit.from_pandas(train_data)
        self["validation"] = DataSplit.from_pandas(validation_data)
        self["generation"] = DataSplit.from_pandas(validation_data)

    def build_chat(self, example: Dict[str, Any], split: str = "") -> LMConversation:
        """
        Build a chat from an AlanAI example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        conversation = LMConversation().add_turn("user", example["input"])
        if split in ["train", "validation"]:
            conversation.add_turn("assistant", example["output"])
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for AlanAI tasks.

        Returns:
            List[str]: List of class labels.
        """
        classes = {}
        if self.name in classes:
            return classes[self.name]
        return []


class BoolQDataset(CausalLMDataset):
    """
    Dataset class for the BoolQ dataset.
    """

    def load(self) -> Any:
        """
        Load the BoolQ dataset and prepare splits.

        Returns:
            Any: The loaded dataset.
        """
        ds = load_dataset("google/boolq")
        self["train"] = DataSplit.from_iterable(ds["train"])
        self["validation"] = DataSplit.from_iterable(ds["validation"])
        return ds

    def build_chat(self, example: Dict[str, Any], split: str = "") -> LMConversation:
        """
        Build a chat from a BoolQ example.

        Args:
            example (Dict[str, Any]): The example to build the chat from.
            split (str): The name of the split.

        Returns:
            LMConversation: The chat built from the example.
        """
        passage = example["passage"]
        question = example["question"]
        answer = example["answer"]
        conversation = (
            LMConversation().add_turn("user", f"{passage}\nQuestion: {question}?").add_turn("assistant", str(answer))
        )
        return conversation

    def get_task_classes(self) -> List[str]:
        """
        Get the class labels for BoolQ tasks.

        Returns:
            List[str]: List of class labels.
        """
        return ["False", "True"]
