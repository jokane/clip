.PHONY: all check lint test docs clean clean-docs

UVRUN=uv run --group dev

all: docs check install

check: lint test

lint:
	$(UVRUN) pylint clip/*.py test/*.py examples/*.py

test:
	xvfb-run uv run --group dev coverage run --omit=.venv* -m pytest --durations=5
	$(UVRUN) coverage report -m --omit "/usr*","/opt*","*config*"

docs: clip/*.py docs/*.rst docs/*.py
	$(UVRUN) --group example python3 docs/generate.py
	$(MAKE) -C docs html

clean: clean-docs
	rm -rfv build *.egg-info .coverage test/.test_files */__pycache__ dist

clean-docs:
	rm -rfv docs/_build docs/_generated docs/__pycache__

install:
	uv pip install -e .

