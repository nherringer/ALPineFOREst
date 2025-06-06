"""Top-level package for ALPine_FOREst."""

__author__ = """Nicholas Herringer"""
__email__ = "nherringer@uchicago.edu"
__version__ = "0.1.0"

# src/alpfore/__init__.py
import pkgutil
import importlib
import pathlib

# Automatically discover and import all submodules under alpfore
__path__ = __path__  # needed for pkgutil to work
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, prefix=__name__ + "."):
    importlib.import_module(module_name)

