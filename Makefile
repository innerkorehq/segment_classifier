.PHONY: install test build publish clean

install:
	poetry install

test:
	poetry run python test_pipeline.py

build:
	poetry build

publish:
	poetry publish

clean:
	rm -rf dist/ .cache/
