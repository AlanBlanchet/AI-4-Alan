from abc import abstractmethod

from pydantic import BaseModel
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..preprocess import Preprocess


class Tokenizer(Preprocess):
    sos_name: str
    """The name of the start-of-sequence token to be fed to the model"""

    @property
    @abstractmethod
    def sos(self):
        """The start-of-sequence token"""
        ...

    @property
    @abstractmethod
    def eos(self):
        """The end-of-sequence token"""
        ...

    @property
    @abstractmethod
    def pad(self):
        """The padding token"""
        ...

    @abstractmethod
    def __len__(self):
        """Size of the vocabulary"""
        ...


class Token(BaseModel):
    id: int
    text: str


class HFTokenizer(Tokenizer):
    model_config = {"arbitrary_types_allowed": True}

    path: str

    tokenizer: PreTrainedTokenizerBase

    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {"hf"}

    @classmethod
    def configure(cls, config):
        cls.log_info("Loading tokenizer")
        config["tokenizer"] = AutoTokenizer.from_pretrained(config["path"])
        return super().configure(config)

    def __call__(self, text: str | list[str]) -> dict:
        return self.tokenizer(text)["input_ids"]

    @property
    def pad(self):
        return Token(id=self.tokenizer.pad_token_id, text=self.tokenizer.pad_token)

    @property
    def sos(self) -> Token:
        bos_id, bos = self.tokenizer.bos_token_id, self.tokenizer.bos_token
        if all((bos_id is None, bos is None)):
            return self.pad
        return Token(id=bos_id, text=bos)

    @property
    def eos(self):
        return Token(id=self.tokenizer.eos_token_id, text=self.tokenizer.eos_token)

    def __len__(self):
        return self.tokenizer.vocab_size
