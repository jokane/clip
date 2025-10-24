.PHONY: Makefile

UVRUN=uv run --group dev --group example

SPHINXOPTS    ?=
SPHINXBUILD   ?= $(UVRUN) sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
