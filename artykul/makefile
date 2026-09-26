
.PHONY: raz full 

raz : artykul.tex
	pdflatex artykul.tex

full : artykul.tex 
	pdflatex artykul.tex
	bibtex   artykul.aux
	pdflatex artykul.tex
	pdflatex artykul.tex
