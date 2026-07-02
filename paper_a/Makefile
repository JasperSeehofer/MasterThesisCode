# Paper A build. No TeX on the dev box — build on the bwUniCluster login
# node (TeX Live 2020 + the vendored mnras.cls/mnras.bst in this dir):
#   rsync -az --delete paper_a/ bwunicluster:paper_a_build/
#   ssh bwunicluster 'cd paper_a_build && make'
#   rsync -az bwunicluster:paper_a_build/main.pdf paper_a/main.pdf
# main.tex is GENERATED from PAPER-CONFIG.json (gpd paper-build) — edit
# section files in sections/, not main.tex.

main.pdf: main.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf build*.log bib.log

.PHONY: clean
