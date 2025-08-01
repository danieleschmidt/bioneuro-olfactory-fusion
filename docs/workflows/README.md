# GitHub Actions Workflows

This directory contains workflow templates and documentation for CI/CD automation.

> **Note**: Actual workflow files must be created in `.github/workflows/` directory.
> These are templates and documentation for manual setup.

## Required Workflows

### 1. Test Workflow (`test.yml`)
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev,test]"
      - name: Run tests
        run: |
          pytest --cov=bioneuro_olfactory --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### 2. Security Workflow (`security.yml`)
```yaml
name: Security
on: [push, pull_request]
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Bandit Security Scan
        run: bandit -r bioneuro_olfactory/
      - name: Run Safety Check
        run: safety check
      - name: Initialize CodeQL
        uses: github/codeql-action/init@v2
      - name: Perform CodeQL Analysis
        uses: github/codeql-action/analyze@v2
```

### 3. Quality Workflow (`quality.yml`)
```yaml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Run Black
        run: black --check bioneuro_olfactory/
      - name: Run Ruff
        run: ruff check bioneuro_olfactory/
      - name: Run MyPy
        run: mypy bioneuro_olfactory/
```

### 4. Release Workflow (`release.yml`)
```yaml
name: Release
on:
  release:
    types: [published]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Build package
        run: |
          pip install build
          python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## Setup Instructions

1. Create `.github/workflows/` directory in repository root
2. Copy each workflow template into separate `.yml` files
3. Configure required secrets in GitHub repository settings:
   - `PYPI_API_TOKEN` for PyPI publishing
   - `CODECOV_TOKEN` for coverage reporting
4. Enable Dependabot for dependency updates
5. Configure branch protection rules requiring workflow success

## Workflow Integration

- **Pre-commit hooks** run locally during development
- **GitHub Actions** provide CI/CD automation
- **Dependabot** manages dependency updates
- **CodeQL** performs security analysis
- **Codecov** tracks test coverage

## Security Considerations

- All workflows use pinned action versions
- Secrets are properly scoped and rotated
- Dependencies are scanned for vulnerabilities
- SBOM generation for compliance tracking