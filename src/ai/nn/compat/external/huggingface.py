from typing import Literal, Optional

from transformers import AutoModel, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

from ...compat.module import Module


class HuggingFaceModule(Module):
    path: str

    task: Optional[Literal["qa"]] = None

    @classmethod
    def get_identifiers(cls):
        return super().get_identifiers() | {"hf"}

    def init(self):
        super().init()

        self.model = self.load()

    def load(self):
        match self.task:
            case "qa":
                return AutoModelForQuestionAnswering.from_pretrained(self.path)
            case "seq2seq":
                return AutoModelForSeq2SeqLM.from_pretrained(self.path)
            case _:
                return AutoModel.from_pretrained(self.path)

    def forward(self, data):
        return dict(self.model(**data))
