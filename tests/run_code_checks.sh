#!/usr/bin/env bash
set -e

find noddingpigeon -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --errors-only
find noddingpigeon -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --exit-zero
find noddingpigeon -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict --check-untyped-defs --ignore-missing-imports --implicit-reexport
find tests -iname "*.py" | xargs -L 1 python3 -m unittest