import importlib
from dataset.BaseScene import BaseScene


def get_scene(name, attributes) -> BaseScene:
    module = importlib.import_module('dataset.' + name)
    return getattr(module, name)(**attributes)

