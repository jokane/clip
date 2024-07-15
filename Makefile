.PHONY: all check lint test docs clean clean-docs

all: docs check install

check: lint test

lint:
	pylint clip/*.py test/*.py

test:
	NUMBA_DISABLE_JIT=1 coverage run --omit=.venv* -m pytest --durations=5
	coverage report -m --omit "/usr*","/opt*","*config*"

docs: clip/*.py docs/*.rst docs/*.py
	python3 docs/generate.py
	$(MAKE) -C docs html

clean: clean-docs
	rm -rfv build *.egg-info .coverage test/.test_files */__pycache__

clean-docs:
	rm -rfv docs/_build docs/_user docs/api

install:
	pip install .

