# Contributing to BioNeuro-Olfactory-Fusion

Thank you for your interest in contributing to BioNeuro-Olfactory-Fusion! This project aims to advance neuromorphic computing for safety-critical gas detection applications.

## Code of Conduct

This project adheres to the Contributor Covenant [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Search existing issues before creating a new one
- Include detailed information about your environment and steps to reproduce

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/your-username/bioneuro-olfactory-fusion.git
   cd bioneuro-olfactory-fusion
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

### Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests and checks**
   ```bash
   pytest
   black .
   ruff check .
   mypy bioneuro_olfactory/
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Pull Request Process

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request**
   - Use a clear title and description
   - Reference any related issues
   - Include testing instructions

3. **Review Process**
   - Maintainers will review your PR
   - Address feedback promptly
   - Ensure all checks pass

## Development Guidelines

### Code Style

- Follow PEP 8 conventions
- Use Black for formatting (88 character line limit)
- Use type hints for all functions
- Write docstrings for all public APIs

### Testing

- Write unit tests for all new code
- Maintain test coverage above 90%
- Include integration tests for hardware interfaces
- Test with multiple Python versions (3.9-3.12)

### Documentation

- Update README.md for new features
- Add docstrings following Google style
- Include examples in the `examples/` directory
- Update API documentation

## Areas for Contribution

### High Priority
- Core spiking neural network implementations
- Sensor interface drivers
- Hardware backend integrations
- Performance optimizations

### Medium Priority
- Additional encoding schemes
- New fusion algorithms
- Calibration utilities
- Visualization tools

### Documentation
- Tutorial notebooks
- Hardware setup guides
- Performance benchmarks
- Use case examples

## Security Considerations

This project handles safety-critical applications. Please:

- Follow secure coding practices
- Report security vulnerabilities privately via email
- Ensure sensor data handling complies with privacy requirements
- Test thoroughly in controlled environments

## Getting Help

- Join discussions in GitHub Discussions
- Check the documentation at [docs/](docs/)
- Contact maintainers for complex contributions

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Academic papers (with permission)

Thank you for contributing to safer environments through neuromorphic computing!