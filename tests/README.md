# RAGitect Tests

This directory contains the test suite for RAGitect.

## Running Tests

### Run all tests
```bash
uv run pytest
```

### Run with verbose output
```bash
uv run pytest -v
```

### Run specific test file
```bash
uv run pytest tests/services/test_config.py
```

### Run with coverage report
```bash
uv run pytest --cov=ragitect --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see detailed coverage.

## Test Structure

```
tests/
├── services/
│   ├── processor/
│   │   ├── test_factory.py      # Processor factory tests
│   │   └── test_simple.py       # Simple text processor tests
│   ├── test_config.py           # Configuration loading tests
│   ├── test_document_processor.py  # Document processing tests
│   ├── test_query_service.py    # Query reformulation tests
│   └── test_vector_store.py     # Vector store tests
└── test_engine.py               # ChatEngine tests
```

## Coverage

Current coverage: **60%**

### High Coverage Modules (90%+)
- `config.py` - 100%
- `processor/factory.py` - 100%
- `processor/simple.py` - 100%
- `vector_store.py` - 97%

### Areas for Future Testing
- LLM integration tests (currently 33%)
- Embedding tests (currently 43%)
- Docling processor (currently 35%)
- Full engine integration (currently 42%)

## Writing Tests

### Pure Logic Tests
Focus on testing business logic without external dependencies:
```python
def test_formats_chat_history():
    history = [{"role": "user", "content": "Hello"}]
    result = format_chat_history(history)
    assert '<message role="user">Hello</message>' in result
```

### Configuration Tests
Use `unittest.mock.patch` for environment variables:
```python
@patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True)
def test_loads_from_environment():
    config = load_config_from_env()
    assert config.provider == "openai"
```

## CI/CD

Tests run automatically on:
- Push to `main` or `dev` branches
- Pull requests to `main` or `dev` branches

See `.github/workflows/test.yml` for CI configuration.
