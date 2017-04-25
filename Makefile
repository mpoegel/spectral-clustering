# Compile the project
benchmark:
	g++ src/benchmark.cpp -o bin/benchmark -O2 -larmadillo

# Make the pdf slides from the markdown files
SLIDES := $(wildcard slides/*.md)
slides: $(SLIDES:md=pdf)

slides/%.pdf: slides/%.md
	pandoc $< --from markdown --to beamer -o $@

# Download and munge the data sets
data: init
	./setup/data.sh
	python ./src/preprocess.py

init:
	./setup/init.sh
	./setup/lib.sh
