import logging

from ai.registry.registers import MODEL

from .models import Model


def init():
    logger = logging.getLogger("uvicorn")
    models = Model.all()
    logger.info("Initializing database items")
    model_names = [model.name for model in models]
    for name in MODEL.names:
        if name not in model_names:
            Model(name=name).save()
