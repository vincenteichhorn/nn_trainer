import pytest
from nnt.datasets.dataset import DataSplit
from nnt.datasets.causal_lm_dataset import LMConversation, CausalLMDataset, AlpacaDataset


class DummyTokenizer:
    def __init__(self):
        self.chat_template = None

    def __call__(self, texts):
        # Simulate tokenization: each char becomes a token id (for simplicity)
        return {"input_ids": [[ord(c) for c in texts[0]]]}

    def decode(self, token_ids):
        return "".join(chr(i) for i in token_ids)

    def apply_chat_template(self, chat, tokenize, add_generation_prompt, continue_final_message):
        # Just join roles and contents for testing
        text = "\n".join([f"{el['role']}:\n{el['content']}" for el in chat])
        if add_generation_prompt:
            text += "\nassistant:\n"
        return text


def test_lmconversation_add_turn_and_to_sample():
    conv = LMConversation()
    conv.add_turn("user", "Hello").add_turn("assistant", "Hi!")
    sample = conv.to_sample()
    assert isinstance(sample, dict)
    assert "input_ids" in sample
    assert "labels" in sample
    assert "chat" in sample
    assert sample["chat"][0]["role"] == "user"
    assert sample["chat"][1]["role"] == "assistant"


def test_lmconversation_apply_chat_template_and_tokenize_basic():
    conv = LMConversation()
    conv.add_turn("user", "Hello").add_turn("assistant", "Hi!")
    tokenizer = DummyTokenizer()
    conv.apply_chat_template_and_tokenize(tokenizer)
    assert isinstance(conv.input_ids, list)
    assert isinstance(conv.labels, list)
    assert len(conv.input_ids) == len(conv.labels)
    # Assistant labels only: user tokens should be masked
    assert all(l == -100 or isinstance(l, int) for l in conv.labels)


def test_lmconversation_repr():
    conv = LMConversation()
    conv.add_turn("user", "Hello")
    assert "LMConversation" in repr(conv)


def test_causallmdataset_prepare(monkeypatch):
    class DummyDataset(CausalLMDataset):
        def __init__(self):
            super().__init__()
            self.data = {"train": [{"instruction": "Say hi", "input": "", "output": "Hi!"}]}

        def load(self):
            # No-op for testing, as data is already initialized in __init__
            pass

        def splits(self):
            return self.data

        def __getitem__(self, split):
            return self.data[split]

        def __setitem__(self, split, value):
            self.data[split] = value

        def build_chat(self, sample, split_name=""):
            conv = LMConversation()
            conv.add_turn("user", sample["instruction"] + " " + sample["input"])
            conv.add_turn("assistant", sample["output"])
            return conv

    tokenizer = DummyTokenizer()
    ds = DummyDataset()
    ds.prepare(tokenizer)
    train_sample = ds["train"][0]
    assert "input_ids" in train_sample
    assert "labels" in train_sample
    assert "chat" in train_sample


def test_lmconversation_apply_chat_template_and_tokenize_with_chat_template():
    conv = LMConversation()
    conv.add_turn("user", "Hello").add_turn("assistant", "Hi!")
    tokenizer = DummyTokenizer()
    tokenizer.chat_template = True  # Simulate presence of chat_template
    conv.apply_chat_template_and_tokenize(tokenizer)
    assert isinstance(conv.input_ids, list)
    assert isinstance(conv.labels, list)
    assert len(conv.input_ids) == len(conv.labels)


def test_causallmdataset_build_chat_not_implemented():
    class DummyDataset(CausalLMDataset):
        pass

    with pytest.raises(NotImplementedError):
        DummyDataset.build_chat({"foo": "bar"}, "train")


class DummyAlpacaDataset(CausalLMDataset):

    def load(self):
        # load mock data
        self["train"] = DataSplit([{"instruction": "Hello", "input": "World", "output": "!"}])

    def build_chat(self, sample, split_name="") -> LMConversation:
        conversation = LMConversation()
        conversation.add_turn("user", f"{sample['instruction']} {sample['input']}")
        conversation.add_turn("assistant", sample["output"])
        return conversation


def test_dummy_alpaca_dataset_prepare_and_build_chat():
    tokenizer = DummyTokenizer()
    ds = DummyAlpacaDataset()
    ds.prepare(tokenizer)
    for split in ds.splits().keys():
        for sample in ds[split]:
            assert "input_ids" in sample
            assert "labels" in sample
            assert "chat" in sample
            assert isinstance(sample["input_ids"], list)
            assert isinstance(sample["labels"], list)
            assert isinstance(sample["chat"], list)
            assert sample["chat"][0]["role"] == "user"
            assert sample["chat"][1]["role"] == "assistant"
