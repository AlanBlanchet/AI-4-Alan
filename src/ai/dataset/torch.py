from pydantic import BaseModel


class TorchDataset(BaseModel):
    class Config:
        arbitrary_types_allowed = True
