.PHONY: Makefile

UVRUN=uv run --group dev

SPHINXOPTS    ?=
SPHINXBUILD   ?= $(UVRUN) sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
