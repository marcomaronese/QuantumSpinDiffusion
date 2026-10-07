PYTHON ?= ../.venv/bin/python

.PHONY: all audit figures pdf package
all: audit figures pdf package
audit:
	$(PYTHON) scripts/audit_evidence.py
figures:
	$(PYTHON) scripts/build_assets.py
pdf:
	mkdir -p build
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass1.txt
	cd build && BIBINPUTS=..: bibtex main > bibtex.txt
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass2.txt
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex > build/pass3.txt
	cp build/main.pdf manuscript.pdf
package:
	$(PYTHON) scripts/package.py
