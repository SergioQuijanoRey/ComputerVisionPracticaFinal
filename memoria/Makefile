# Parametros del programa
# FILE_NAME puede llevar espacios sin problemas, sin necesidad de comillas dobles
FILE_NAME=Memoria

all:
	# Tenemos que ejecutar esta secuencia para que se cargue correctamente la bibliografia
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode || echo "Hubo algun problema compilando el latex"
	bibtex "${FILE_NAME}" || echo "Hubo algun problema compilando la bibliografia"
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode || echo "Hubo algun problema compilando el latex"
	pdflatex "${FILE_NAME}.tex" -interaction=nonstopmode || echo "Hubo algun problema compilando el latex"
clean:
	@echo "Borrando archivos auxiliares"
	@rm -rf *.aux *.log *.toc *.bbl *.blg *.log *.lof *.lot *.out *.lol
