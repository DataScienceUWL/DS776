# Makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR    = source
BUILDDIR     = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for introdl
html:
	@echo "Building HTML documentation..."
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)
	@echo "HTML documentation built in build/html/"

markdown:
	@echo "Building Markdown documentation for Custom GPT..."
	@$(SPHINXBUILD) -b markdown "$(SOURCEDIR)" "$(BUILDDIR)/markdown" $(SPHINXOPTS)
	@echo "Markdown documentation built in build/markdown/"

both: html markdown
	@echo "Built both HTML and Markdown documentation"

clean:
	@echo "Cleaning build directory..."
	@rm -rf $(BUILDDIR)/*
	@echo "Build directory cleaned"

# Quick rebuild (clean + build both formats)
rebuild: clean both
	@echo "Documentation rebuilt completely"

# Catch-all target: route all unknown targets to Sphinx
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
