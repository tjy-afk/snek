# The absolute path to the repository root
REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST))))

# Python Poetry setup & config
setup-poetry:
	poetry config virtualenvs.in-project true
	poetry install