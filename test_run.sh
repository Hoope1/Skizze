#!/bin/sh
python -m py_compile $(git ls-files '*.py')
python Skizze.py --help
