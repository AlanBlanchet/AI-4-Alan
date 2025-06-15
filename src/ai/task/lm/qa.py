from .task import LLM


class QuestionAnsweringLLM(LLM):
    alias = "qa"

    answers: str
    """The key in the batch that contains the answers"""

    # def process(self, model, batch, split, item_idx):
    #     out, losses = super().process(model, batch, split, item_idx)

    #     return out, losses
