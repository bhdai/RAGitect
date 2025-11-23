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

### Run without coverage (faster)

```bash
uv run pytest --no-cov
```

### Run with coverage report

```bash
uv run pytest --cov=ragitect --cov-report=html
```

Then open `htmlcov/index.html` in your browser to see detailed coverage.

### Run integration tests only

```bash
uv run pytest -m integration
```

### Skip integration tests

```bash
uv run pytest -m "not integration"
```

### Running Integration Tests with a Dedicated Test Database

For integration tests that interact with a PostgreSQL database, it is crucial to use a dedicated test database (e.g., `ragitect_test`) to prevent data loss in your main development database.

1.  **Create `.env.test`**: In your project root, create a file named `.env.test` with your test database credentials:
    ```
    DATABASE_URL="postgresql+asyncpg://admin:admin@localhost:5432/ragitect_test"
    ```
    (Adjust credentials and port if different). **Do NOT commit this file to Git.** Consider adding `.env.test` to your `.gitignore`.

2.  **Run with `--env-file`**: Execute tests using the following command:
    ```bash
    uv run --env-file .env.test pytest -m integration
    ```
    This ensures your tests use the specified test database.

## Test Structure

```
tests/
├── services/
│   ├── database/
│   │   └── test_connection.py     # Database connection tests
│   ├── processor/
│   │   ├── test_factory.py        # Processor factory tests
│   │   └── test_simple.py         # Simple text processor tests
│   ├── test_config.py             # Configuration loading tests
│   ├── test_document_processor.py # Document processing tests
│   ├── test_query_service.py      # Query reformulation tests
│   └── test_vector_store.py       # Vector store tests
└── test_engine.py                 # ChatEngine tests
```

## Coverage

Current coverage: **70%** (improved with database model tests!)

### High Coverage Modules (90%+)

- `config.py` - 100%
- `processor/factory.py` - 100%
- `processor/simple.py` - 100%
- `database/models.py` - 100%
- `database/connection.py` - 93%
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

### Async Tests

Use `@pytest.mark.asyncio` for async functions:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Integration Tests

Mark tests that require external services (database, etc.):

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_operations():
    import os
    if not os.getenv('DATABASE_URL'):
        pytest.skip("DATABASE_URL not set")
    # ... test code ...
```

## Database Connection Tests

The `test_connection.py` file provides comprehensive coverage for the database layer:

### Unit Tests (run without database)

- **Singleton Pattern**: Verifies DatabaseManager follows singleton pattern
- **Initialization**: Tests engine and session factory creation
- **Configuration**: Tests config value usage
- **Error Handling**: Tests connection failure scenarios
- **Session Management**: Tests context managers and auto-commit
- **Cleanup**: Tests proper resource disposal

### Integration Tests (require DATABASE_URL)

- **Real Connection**: Tests actual database connectivity
- **pgvector Extension**: Verifies pgvector support
- **Vector Operations**: Tests vector insert/query/similarity
- **Transactions**: Tests commit and rollback behavior

To run integration tests, set `DATABASE_URL` environment variable:

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/dbname"
uv run pytest -m integration
```

## Database Model Tests

The `test_models.py` file ensures the correctness of the SQLAlchemy ORM models through a combination of unit and integration tests.

### Unit Tests

- **Model Structure**: Verifies that each model (`Workspace`, `Document`, `DocumentChunk`) has the correct table name, fields, and relationships.
- **Constraints**: Checks that `UNIQUE`, `CHECK`, and `FOREIGN KEY` constraints are properly defined.
- **Timestamp Defaults**: Confirms that timestamp fields like `created_at` and `processed_at` have server-side defaults.
- **Representation**: Tests the `__repr__` method of each model for correct output.

### Integration Tests (require `DATABASE_URL`)

- **CRUD Operations**: Basic creation of each model to ensure they can be persisted to the database.
- **Constraint Enforcement**: Verifies that database-level constraints (e.g., unique names) raise `IntegrityError` when violated.
- **Cascade Behavior**: Tests that deleting a `Workspace` cascades to `Document`s and `DocumentChunk`s, and `Document` deletion cascades to `DocumentChunk`s.
- **`onupdate` Timestamps**: Ensures that the `updated_at` timestamp on `Workspace` is automatically updated when the record is modified.

## CI/CD

Tests run automatically on:

- Push to `main` or `dev` branches
- Pull requests to `main` or `dev` branches

See `.github/workflows/test.yml` for CI configuration.
