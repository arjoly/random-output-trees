# Author: Arnaud Joly

all: clean inplace test

clean:
	python setup.py clean

in: inplace

inplace:
	python setup.py build_ext --inplace

test:
	nosetests randomized_output_forest

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

view-doc: doc
	open doc/_build/html/index.html
