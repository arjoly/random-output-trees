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
	rm -rf ../random-output-trees-doc
	cp -a doc/_build/html ../random-output-trees-doc
	git checkout gh-pages
	cp -a ../random-output-trees-doc/* .
	echo 'Add new file to git'
	git add `ls ../random-output-trees-doc`
	git commit -m "Generated gh-pages for `git log master -1 --pretty=short --abbrev-commit`"
	git push origin gh-pages
	git checkout master
