# Author: Arnaud Joly

all: clean inplace

clean:
	python setup.py clean

in: inplace

inplace:
	python setup.py build_ext --inplace

