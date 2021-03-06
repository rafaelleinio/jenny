.PHONY: requirements-test
requirements-test:
	@PYTHONPATH=. python -m pip install -r requirements.test.txt

.PHONY: requirements-lint
requirements-lint:
	@PYTHONPATH=. python -m pip install -r requirements.lint.txt

.PHONY: minimum-requirements
minimum-requirements:
	@PYTHONPATH=. python -m pip install -U -r requirements.txt

.PHONY: dev-requirements
dev-requirements:
	@PYTHONPATH=. python -m pip install -U -r requirements.dev.txt

.PHONY: requirements
## install all requirements
requirements: requirements-test requirements-lint dev-requirements minimum-requirements

.PHONY: drone-install
drone-install:
	@pip install --upgrade pip
	@python -m pip install -U -r requirements.test.txt -r requirements.lint.txt -r requirements.dev.txt -r requirements.txt -t ./pip/deps --cache-dir ./pip/cache

.PHONY: tests
tests:
	@python -m pytest -n=auto --cov-config=.coveragerc --cov=jenny --cov-report term --cov-report html:htmlcov --cov-report xml:coverage.xml tests
	@python -m coverage xml -i

.PHONY: unit-tests
unit-tests:
	@echo ""
	@echo "Unit Tests"
	@echo "=========="
	@echo ""
	@python -m pytest -n auto --cov-config=.coveragerc --cov-report term --cov-report html:unit-tests-cov --cov=jenny --cov-fail-under=75 tests/unit

.PHONY: integration-tests
integration-tests:
	@echo ""
	@echo "Integration Tests"
	@echo "================="
	@echo ""
	@python -m pytest -n auto --cov-config=.coveragerc --cov-report term --cov-report xml:integration-tests-cov.xml --cov=jenny --cov-fail-under=60 tests/integration

.PHONY: style-check
style-check:
	@echo ""
	@echo "Code Style"
	@echo "=========="
	@echo ""
	@python -m black --check -t py36 --exclude="build/|buck-out/|dist/|_build/|pip/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" . && echo "\n\nSuccess" || echo "\n\nFailure\n\nRun \"make black\" to apply style formatting to your code"
	@echo ""

.PHONY: check-flake8
check-flake8:
	@echo ""
	@echo "Flake 8"
	@echo "======="
	@echo ""
	@python -m flake8 jenny/ && echo "jenny module success"
	@python -m flake8 tests/ && echo "tests module success"
	@echo ""

.PHONY: checks
checks:
	@echo ""
	@echo "Code Style & Flake 8"
	@echo "--------------------"
	@echo ""
	@make style-check
	@make check-flake8
	@echo ""

.PHONY: black
black:
	@python -m black -t py36 --exclude="build/|buck-out/|dist/|_build/|pip/|\.pip/|\.git/|\.hg/|\.mypy_cache/|\.tox/|\.venv/" .

.PHONY: clean
clean:
	@find ./ -type d -name 'htmlcov' -exec rm -rf {} +;
	@find ./ -type d -name 'coverage.xml' -exec rm -rf {} +;
	@find ./ -type f -name 'coverage-badge.svg' -exec rm -f {} \;
	@find ./ -type f -name '.coverage' -exec rm -f {} \;
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;

.PHONY: version
version:
	@grep __version__ setup.py | head -1 | cut -d \" -f2 | cut -d \' -f2 > .version

.PHONY: package-name
package-name:
	@grep __package_name__ setup.py | head -1 | cut -d \" -f2 | cut -d \' -f2 > .package_name

.PHONY: repository-url
repository-url:
	@grep __repository_url__ setup.py | head -1 | cut -d \" -f2 | cut -d \' -f2 > .repository_url

.PHONY: package
package:
	@python -m setup sdist bdist_wheel

.PHONY: upload-pypi
upload-pypi:
	@python -m twine upload dist/*
