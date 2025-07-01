"""Top-level package for ALPineFOREst."""


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

__author__ = """Nicholas Herringer"""
__email__ = "nherringer@uchicago.edu"
__version__ = "0.1.3"

# src/alpfore/__init__.py
import pkgutil
import importlib
import pathlib
import sys

# for loader, module_name, is_pkg in pkgutil.walk_packages(
#     __path__, prefix=__name__ + "."
# ):
#     module = importlib.import_module(module_name)
#     # attach the top-level module to the alpfore namespace
#     name_parts = module_name.split(".")
#     if len(name_parts) == 2:  # e.g., alpfore.encoder
#         setattr(sys.modules[__name__], name_parts[1], module)

# Auto-discover all submodules
from . import encoder, loaders, evaluators, pipeline, utils, models, candidate_selectors
from .pipeline import Pipeline
from .utils.dataset_utils import (
    make_labeled_dataset,
    save_labeled_dataset,
    load_labeled_dataset,
    append_new_data,
)

