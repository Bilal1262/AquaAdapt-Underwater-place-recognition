.PHONY: install test quick full
install:
	python3 -m pip install -e .
test:
	pytest -q
quick:
	bash scripts/run_quick_pipeline.sh
full:
	bash scripts/run_full_pipeline.sh

