#!/bin/sh

set -e -x

# Mypy does not yet handle all Python v3.10 features
mypy *.py
black . --diff --color --check
pflake8
