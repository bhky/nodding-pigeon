#!/usr/bin/env bash
set -e

find hgd -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --errors-only
find hgd -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 pylint --exit-zero
find hgd -iname "*.py" | grep -v -e "__init__.py" | xargs -L 1 mypy --strict --check-untyped-defs --implicit-reexport
find tests -iname "*.py" | xargs -L 1 python3 -m unittest