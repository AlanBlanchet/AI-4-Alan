from ...modality.text import Text, Tokenizer
from ..task import Task


class LLM(Task, buildable=False):
    alias = "llm"

    tokenizer: Tokenizer
    """The tokenizer to use for the task"""

    @classmethod
    def configure(cls, config):
        model_config = config["model"]
        # Try resolve empty tokenizer from model
        if "type" not in config["tokenizer"] and "type" in model_config:
            config["tokenizer"]["type"] = model_config["type"]
            if "path" not in config["tokenizer"] and "path" in model_config:
                config["tokenizer"]["path"] = model_config["path"]
        # Tokenizer
        config["tokenizer"] = Tokenizer.from_config(config["tokenizer"])
        # If we have an HF module, we need to add the task to load the specific module
        if model_config["type"] == "hf":
            config["model"]["task"] = cls.alias
        config = super().configure(config)
        config["datasets"].add_preprocess(config["tokenizer"], modality=Text)
        return config

    # def prepare_input(self, batch: dict):
    #     batch = super().prepare_input(batch)

    #     ex = batch[list(batch.keys())[0]]
    #     B = ex.shape[0]

    #     start_id = self.tokenizer.sos.id
    #     decoder_input_ids = torch.full(
    #         (B, 1), start_id, dtype=torch.long, device=ex.device
    #     )
    #     batch[self.tokenizer.sos_name] = decoder_input_ids
    #     return batch

    # def forward(self, x):
    #     ex = x[list(x.keys())[0]]
    #     B = ex.shape[0]
    #     # Regress on every batches
    #     batches = [{} for _ in range(B)]
    #     for k, v in x.items():
    #         for b in range(B):
    #             batches[b][k] = v[b]

    #     for batch in batches:
    #         out = super().forward(x)

    #     print(out)

    #     return out
