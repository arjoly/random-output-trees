# Author: Arnaud Joly

clean:
	python setup.py clean

in: inplace

inplace:
	python setup.py build_ext --inplace

all: clean inplace
