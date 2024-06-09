from sqlalchemy import Column, ForeignKey, String
from sqlalchemy.orm import mapped_column, relationship

from .base import Base, CustomBase, engine


class Grid(CustomBase):
    __tablename__ = "grid"
    name = Column(String, index=True)
    blocks = relationship("Block")


class Block(CustomBase):
    __tablename__ = "block"
    name = Column(String, index=True)
    grid_id = mapped_column(ForeignKey("grid.id"), nullable=False)


class Model(CustomBase):
    __tablename__ = "model"
    name = Column(String, index=True)


class Dataset(CustomBase):
    __tablename__ = "dataset"
    name = Column(String, index=True)


Base.metadata.create_all(bind=engine)
