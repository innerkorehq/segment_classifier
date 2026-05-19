.PHONY: install test build publish clean bump-version

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

bump-version:
	@if [ -z "$(VERSION)" ]; then \
		echo "Error: VERSION variable not set. Use VERSION=1.2.3 make bump-version"; \
		exit 1; \
	fi
	@sed -i '' -E "s/^version = \".*\"$$/version = \"$(VERSION)\"/" pyproject.toml
	@git add pyproject.toml
	@if git diff --cached --quiet; then \
		echo "Version is already $(VERSION), nothing to commit."; \
	else \
		git commit -m "Bump version to $(VERSION)" && \
		git tag v$(VERSION) && \
		echo "Bumped version to $(VERSION)"; \
	fi
