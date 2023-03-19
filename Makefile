# Install Python Dependencies for test
test_requirements:
	pip install -r requirements/test_requirements.txt

# Install Python Dependencies
requirements:
	pip install -r requirements/requirements.txt

# Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

#  Lint using
## flake8: PEP8 based lint
### there is an error in pytest-flake8
## mypy  : type check
lint:
	flake8 catchMinor test
	mypy catchMinor

#  formatting
## black: formatting
## isort: import formatting
format:
	black catchMinor test
	isort --profile black catchMinor test