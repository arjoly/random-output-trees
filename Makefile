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
	echo 'Mv file'
	mv -fv doc/_build/html/* ./
	echo 'Add new file to git'
	git add *.html *.js *.inv generated auto_examples
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
	git push origin gh-pages
	git checkout master
