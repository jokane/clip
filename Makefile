.PHONY: all check lint test doc clean

all: doc check install

check: lint test

lint:
	pylint clip/*.py test/*.py

test:
	NUMBA_DISABLE_JIT=1 coverage run --omit=.venv* -m pytest --durations=5
	coverage report -m --omit "/usr*","/opt*","*config*"

doc: clip/*.py doc/*.rst doc/*.py
	python3 doc/generate.py
	$(MAKE) -C doc html

clean:
	rm -rfv doc/_build doc/_user build *.egg-info .coverage .test_files */__pycache__

install:
	pip install .

