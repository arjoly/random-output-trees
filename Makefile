# Author: Arnaud Joly

all: clean inplace test

clean:
	python setup.py clean

in: inplace

inplace:
	python setup.py build_ext --inplace

test:
	nosetests random_output_trees

doc: inplace
	$(MAKE) -C doc html

doc-noplot: inplace
	$(MAKE) -C doc html-noplot

view-doc: doc
	open doc/_build/html/index.html

gh-pages:
    git checkout master
    make doc
    git checkout gh-pages
    mv -fv doc/_build/html/* .
    git add -A
    git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
    git push origin gh-pages
    git checkout master
