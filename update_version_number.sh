#!/bin/bash

sed -i '' "s/^version = .*/version = \"$1\"/" pyproject.toml
sed -i '' "s/^__version__ = .*/__version__ = \"$1\"/" src/alpfore/__init__.py
