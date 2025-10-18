# Contributing to AbstractVoice

Thank you for your interest in contributing to AbstractVoice! This document provides guidelines and best practices for contributors.

## Development Setup

### Prerequisites
- Python 3.11.7+ (3.11 recommended)
- Git
- Virtual environment tool (venv, conda, etc.)

### Installation for Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lpalbou/abstractvoice.git
   cd abstractvoice
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install espeak-ng (recommended for best TTS quality):**
   - **macOS**: `brew install espeak-ng`
   - **Linux**: `sudo apt-get install espeak-ng`
   - **Windows**: See [installation guide](../README.md#installing-espeak-ng-recommended-for-best-quality)

### Development Dependencies
- `pytest>=7.0.0` - Testing framework
- `black>=22.0.0` - Code formatting
- `flake8>=5.0.0` - Code linting

## Code Style and Standards

### Python Code Style
We follow PEP 8 with some modifications:

```bash
# Format code with black
black abstractvoice/

# Check linting with flake8
flake8 abstractvoice/
```

### Code Organization Principles
- **Single Responsibility**: One file = one responsibility
- **Small Files**: Keep files under 600 lines when possible
- **Clear Separation**: Separate concerns between modules
- **OOP Design**: Follow SOLID principles (except "O" for pre-release)

### Documentation Standards
- **Docstrings**: Use Google-style docstrings for all public methods
- **Comments**: Explain "why", not just "what"
- **Type Hints**: Use type hints for public APIs
- **Examples**: Include usage examples in docstrings

Example:
```python
def speak(self, text: str, speed: float = 1.0, callback: Optional[Callable] = None) -> bool:
    """Convert text to speech and play audio.
    
    Args:
        text: Text to convert to speech
        speed: Speech speed multiplier (0.5-2.0, default 1.0)
        callback: Optional callback function called when speech completes
        
    Returns:
        True if speech started, False if text was empty
        
    Example:
        >>> vm = VoiceManager()
        >>> vm.speak("Hello world", speed=1.2)
        True
    """
```

## Testing Guidelines

### Test Philosophy
- **Tests illustrate desired behavior**
- **Code must work for general cases**, not just test cases
- **Never add special case handling** from test files in production code
- **Design for robustness and generality**

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=abstractvoice

# Run specific test file
pytest tests/test_voice_manager.py

# Run with verbose output
pytest -v
```

### Writing Tests
```python
import pytest
from abstractvoice import VoiceManager

def test_voice_manager_initialization():
    """Test that VoiceManager initializes correctly."""
    vm = VoiceManager()
    assert vm is not None
    assert not vm.is_speaking()
    assert not vm.is_paused()

def test_pause_resume_functionality():
    """Test immediate pause/resume functionality."""
    vm = VoiceManager()
    
    # Start speech
    result = vm.speak("This is a test of pause and resume functionality.")
    assert result is True
    
    # Pause should work immediately
    pause_result = vm.pause_speaking()
    assert pause_result is True
    assert vm.is_paused()
    
    # Resume should work immediately
    resume_result = vm.resume_speaking()
    assert resume_result is True
    assert not vm.is_paused()
    
    vm.cleanup()
```

## Error Handling Best Practices

### Error Handling Philosophy
- **Consider multiple possible causes** before deciding
- **Reason through root causes**, don't jump to conclusions
- **Solutions must work for all inputs**, not just known test cases
- **Fix issues without breaking existing functionality**

### Exception Handling
```python
def robust_method(self, input_data):
    """Example of robust error handling."""
    try:
        # Validate input
        if not input_data:
            return False
            
        # Main logic
        result = self._process_data(input_data)
        return result
        
    except SpecificException as e:
        # Handle specific known issues
        if self.debug_mode:
            print(f"Known issue: {e}")
        return self._fallback_method(input_data)
        
    except Exception as e:
        # Handle unexpected issues
        if self.debug_mode:
            print(f"Unexpected error: {e}")
            import traceback
            traceback.print_exc()
        return False
```

## Architecture Guidelines

### Component Design
- **Modular**: Each component should be independently testable
- **Loosely Coupled**: Components communicate via well-defined interfaces
- **Thread-Safe**: All public methods must be thread-safe
- **Resource Management**: Proper cleanup and resource management

### Threading Best Practices
- **Use locks** for shared state
- **Avoid blocking operations** in main thread
- **Use queues** for thread communication
- **Proper cleanup** of threads and resources

Example:
```python
class ThreadSafeComponent:
    def __init__(self):
        self._lock = threading.Lock()
        self._state = {}
    
    def update_state(self, key, value):
        """Thread-safe state update."""
        with self._lock:
            self._state[key] = value
    
    def get_state(self, key):
        """Thread-safe state access."""
        with self._lock:
            return self._state.get(key)
```

## Documentation Updates

### When to Update Documentation
- **New features**: Add examples and API documentation
- **Bug fixes**: Update if behavior changes
- **Breaking changes**: Update migration guides
- **Performance improvements**: Update benchmarks

### Documentation Files to Update
- `README.md` - User-facing documentation
- `CHANGELOG.md` - Version history and changes
- `docs/architecture.md` - Technical architecture
- `docs/DEVELOPMENT.md` - Development insights
- Docstrings in code

### Changelog Format
```markdown
## [0.2.1] - 2025-01-15

### Added
- New feature X with immediate response
- Support for Y configuration

### Changed
- Improved performance of Z by 30%
- Updated default behavior of W

### Fixed
- Fixed issue with A causing B
- Resolved thread safety issue in C

### Deprecated
- Method D is deprecated, use E instead
```

## Pull Request Process

### Before Submitting
1. **Run tests**: Ensure all tests pass
2. **Check formatting**: Run `black` and `flake8`
3. **Update documentation**: Update relevant docs
4. **Test manually**: Test your changes manually
5. **Update changelog**: Add entry to `CHANGELOG.md`

### Pull Request Template
```markdown
## Description
Brief description of changes and motivation.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Manual testing completed

## Documentation
- [ ] Updated README.md if needed
- [ ] Updated CHANGELOG.md
- [ ] Updated docstrings
- [ ] Updated architecture docs if needed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] No breaking changes (or properly documented)
```

### Review Process
1. **Automated checks**: CI/CD runs tests and linting
2. **Code review**: Maintainer reviews code quality and design
3. **Testing**: Manual testing of new features
4. **Documentation review**: Ensure docs are accurate and complete
5. **Merge**: After approval, changes are merged

## Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist
1. **Update version** in `abstractvoice/__init__.py` and `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite**
4. **Build and test package** locally
5. **Create release tag**
6. **Publish to PyPI**
7. **Update documentation**

## Getting Help

### Resources
- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Documentation**: Check existing docs first
- **Code Examples**: Look at examples in `abstractvoice/examples/`

### Communication Guidelines
- **Be respectful** and constructive
- **Provide context** when asking questions
- **Include code examples** when reporting issues
- **Search existing issues** before creating new ones

## Code of Conduct

### Our Standards
- **Respectful**: Treat all contributors with respect
- **Inclusive**: Welcome diverse perspectives and backgrounds
- **Collaborative**: Work together constructively
- **Professional**: Maintain professional communication

### Unacceptable Behavior
- Harassment or discrimination
- Trolling or inflammatory comments
- Personal attacks
- Spam or off-topic content

### Enforcement
Violations may result in temporary or permanent ban from the project.

## Recognition

Contributors are recognized in:
- **ACKNOWLEDGMENTS.md**: All contributors listed
- **Release notes**: Major contributors highlighted
- **GitHub**: Contributor statistics and graphs

Thank you for contributing to AbstractVoice! 🎉
