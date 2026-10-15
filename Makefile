# `make` is not installed on the author's Windows machine, so this Makefile is untested there.
# The tested entry point is: python -m src.run_all --stage {smoke,probe,main,analyze,replay}
PY ?= .venv/Scripts/python.exe

.PHONY: all smoke probe main analyze replay lint test
all: smoke probe main analyze replay
smoke probe main analyze replay:
	$(PY) -W ignore -m src.run_all --stage $@
lint:
	$(PY) -m ruff check src tests && $(PY) -m ruff format --check src tests
test:
	$(PY) -m unittest tests.test_analyze -v
