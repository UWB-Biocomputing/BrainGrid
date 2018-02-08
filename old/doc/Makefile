all: docs


docs:
	# Need to run the pdflatex command twice to generate a TOC
	pdflatex BrainGrid_Documentation.tex
	pdflatex BrainGrid_Documentation.tex
	rm *.log *.out *.aux *.toc

