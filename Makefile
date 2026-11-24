# =============================================================================
# Makefile for A/B Testing Platform
# Targets: setup, test, run-app, run-simulations, clean
# =============================================================================

.PHONY: help setup test run-app run-simulations clean

PYTHON ?= python
PIP ?= pip
STREAMLIT ?= streamlit
PYTEST ?= pytest

help:
	@echo "Available commands:"
	@echo "  make setup            - Install required dependencies from requirements.txt"
	@echo "  make test             - Execute all pytest test suites"
	@echo "  make run-app          - Launch the Streamlit interactive dashboard"
	@echo "  make run-simulations  - Run simulations, regenerate figures, and analyze dataset"
	@echo "  make clean            - Remove cache files and build artifacts"

setup:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m $(PYTEST) -v

run-app:
	$(STREAMLIT) run app/app.py

run-simulations:
	$(PYTHON) -c "from src.simulate import generate_all_plots; generate_all_plots(); print('Figures generated in reports/figures/')"
	$(PYTHON) run_marketing_ab_test.py

clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ app/__pycache__ .pytest_cache
