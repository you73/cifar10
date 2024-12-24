.PHONY: clean install prepare-data benchmark test

# Target to create and install dependencies in a virtual environment
install:
	@echo "Creating virtual environment..."
	@python3 -m venv myenv
	@echo "Upgrading pip..."
	@./myenv/bin/pip install --upgrade pip
	@echo "Installing dependencies..."
	@./myenv/bin/pip install -r requirements.txt
	@./myenv/bin/pip install -r requirements-dev.txt

# Target to prepare data
prepare-data:
	@echo "Preparing data..."
	@PYTHONPATH=$(shell pwd)/src ./myenv/bin/python src/data/dataset.py

# Target to run benchmarks
benchmark:
	@echo "Running benchmarks..."
	@PYTHONPATH=$(shell pwd)/src ./myenv/bin/python src/benchmark.py --preprocessor normalize --feature-extractor flatten --classifier sgd --batch-size 32 --learning-rate 0.01 --kernel linear

# Target to run tests
test:
	@echo "Running tests..."
	@PYTHONPATH=$(shell pwd)/src ./myenv/bin/pytest tests/

# Target to clean up generated files
clean:
	@echo "Cleaning up..."
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@rm -rf data/processed/*
	@rm -rf .pytest_cache
	@rm -rf .coverage
	@rm -rf htmlcov
	@rm -rf .mypy_cache
	@rm -rf .pylint.d
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf notebooks/data
	@rm -rf myenv/
	@rm -rf data/processed/*
	@rm -rf data/raw/*


