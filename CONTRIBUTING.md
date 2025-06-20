# Contributing to Mesa-Optimizer Detection Framework
I
welcome contributions to the Mesa-Optimizer Detection Framework! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Test your changes
6. Submit a pull request

## How to Contribute

### Reporting Bugs

- Use the GitHub issue tracker
- Include a clear description of the bug
- Provide steps to reproduce
- Include relevant system information
- Attach relevant log files or error messages

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement
- Explain why it would be useful
- Provide examples if possible

### Contributing Code

- Detection methods (new analyzers)
- Model integrations
- Performance improvements
- Bug fixes
- Documentation improvements
- Tests

## Development Setup

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone your fork
git clone https://github.com/sanowl/Mesa-Optimizer-Detection-Framework
cd Mesa-Optimizer-Detection-Framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for code formatting
- Use type hints where appropriate
- Write docstrings for all public functions and classes

### Code Organization

- Keep functions focused and small
- Use descriptive variable names
- Add comments for complex logic
- Organize imports alphabetically

### Example

```python
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from mesa_optimizer_detection.core.results import DetectionResults


class NewDetectionMethod:
    """
    Brief description of the detection method.
    
    This class implements a new detection method for identifying
    mesa-optimization patterns in neural networks.
    
    Args:
        model: The model to analyze
        config: Configuration options
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        config: Optional[Dict] = None
    ):
        self.model = model
        self.config = config or {}
    
    def analyze(self, input_batch: torch.Tensor) -> DetectionResults:
        """
        Analyze the model for mesa-optimization patterns.
        
        Args:
            input_batch: Input data for analysis
            
        Returns:
            Detection results
        """
        # Implementation here
        pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=mesa_optimizer_detection

# Run specific test file
pytest tests/test_detection.py

# Run with verbose output
pytest -v
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Include both positive and negative test cases
- Mock external dependencies

### Test Example

```python
def test_new_detection_method():
    """Test that the new detection method works correctly."""
    model = create_test_model()
    detector = NewDetectionMethod(model)
    
    input_data = torch.randn(4, 10)
    results = detector.analyze(input_data)
    
    assert isinstance(results, DetectionResults)
    assert 0 <= results.risk_score <= 1
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose
    and behavior of the function.
    
    Args:
        param1: Description of parameter 1
        param2: Description of parameter 2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: Description of when this is raised
        
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    return True
```

### README Updates

- Update README.md if adding new features
- Include usage examples
- Update installation instructions if needed

## Pull Request Process

### Before Submitting

1. Ensure all tests pass
2. Run code formatting tools
3. Update documentation
4. Write clear commit messages

### Commit Messages

Use conventional commit format:

```
type(scope): brief description

Longer description if needed

Fixes #123
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Review Process

1. Automated checks must pass
2. At least one maintainer review required
3. Address all review feedback
4. Squash commits if requested
5. Maintainer will merge when approved

## Areas for Contribution

### High Priority

- New detection methods
- Performance optimizations
- Better visualization tools
- Integration with popular ML frameworks

### Medium Priority

- Additional model architectures
- Configuration presets
- Example notebooks
- Benchmarking tools

### Low Priority

- UI improvements
- Additional export formats
- Advanced statistical analysis

## Getting Help

- GitHub Issues: For bugs and feature requests
- Discussions: For questions and general discussion
- Email: For security issues or private concerns

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Academic papers (if applicable)

Thank you for contributing to making AI systems safer! 