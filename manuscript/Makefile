LATEXMK = latexmk
LATEXMK_FLAGS = -pdf -interaction=nonstopmode

.PHONY: all clean distclean

all: MOSAIC.pdf MOSAIC_supplementary.pdf

MOSAIC.pdf: main.tex references.bib
	$(LATEXMK) $(LATEXMK_FLAGS) -jobname=MOSAIC main.tex

MOSAIC_supplementary.pdf: supplementary.tex
	$(LATEXMK) $(LATEXMK_FLAGS) -jobname=MOSAIC_supplementary supplementary.tex

clean:
	$(LATEXMK) -c -jobname=MOSAIC main.tex
	$(LATEXMK) -c -jobname=MOSAIC_supplementary supplementary.tex

distclean:
	$(LATEXMK) -C -jobname=MOSAIC main.tex
	$(LATEXMK) -C -jobname=MOSAIC_supplementary supplementary.tex
