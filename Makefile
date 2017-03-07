.PHONY: init test lint coverage

init:
	pip install -r requirements.txt
	pip install -e .

test:
	coverage run --source ziphmm setup.py test

lint:
	flake8

coverage:
	coverage report
