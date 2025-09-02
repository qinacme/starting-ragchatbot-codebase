# RAG System Testing Framework

This directory contains comprehensive tests for the RAG system backend, focusing on API endpoint testing and integration scenarios.

## Test Structure

### Files
- `conftest.py` - Shared fixtures and test configuration
- `test_api_endpoints.py` - API endpoint tests for `/api/query`, `/api/courses`, and root endpoint
- `test_app_integration.py` - Integration tests with static file handling (optional)

### Key Features
- **FastAPI TestClient Integration** - Uses FastAPI's built-in testing client
- **Mock RAG System** - Avoids dependencies on external services during testing
- **Pydantic Model Validation** - Tests request/response model validation
- **Error Handling Coverage** - Tests various error conditions and edge cases
- **Async Support** - Full async/await support with pytest-asyncio

## Running Tests

### All Tests
```bash
uv run pytest tests/ -v
```

### By Category (using markers)
```bash
# API endpoint tests only
uv run pytest tests/ -v -m "api"

# Integration tests only  
uv run pytest tests/ -v -m "integration"

# Unit tests only
uv run pytest tests/ -v -m "unit"
```

### With Coverage (if pytest-cov is installed)
```bash
uv run pytest tests/ --cov=. --cov-report=html
```

## Test Configuration

The pytest configuration in `pyproject.toml` includes:
- Test discovery settings
- Async mode configuration
- Custom markers for test categorization
- Output formatting options

## Key Test Categories

### API Endpoint Tests (`test_api_endpoints.py`)

**QueryEndpoint Tests:**
- Valid requests with/without session ID
- Request validation and error handling
- Response model validation
- Empty and malformed requests

**CoursesEndpoint Tests:**
- Course statistics retrieval
- Empty state handling  
- Error condition handling

**RootEndpoint Tests:**
- Basic API information endpoint

**RequestValidation Tests:**
- JSON parsing errors
- Content-type validation
- Malformed request bodies

**EndpointIntegration Tests:**
- Multi-step user workflows
- Session management across requests

## Fixtures (from `conftest.py`)

### Core Fixtures
- `test_app` - FastAPI application with inline endpoint definitions
- `client` - TestClient for making HTTP requests
- `mock_rag_system` - Mocked RAG system with preset responses
- `test_config` - Test configuration with temporary paths

### Data Fixtures
- `sample_query_request` - Example query request data
- `sample_query_response` - Example query response data
- `sample_course_stats` - Example course statistics data

### Utility Fixtures  
- `temp_dir` - Temporary directory for test files
- `mock_env_vars` - Mocked environment variables
- `cleanup_test_files` - Automatic cleanup after tests

## Static File Handling

The test framework addresses the static file mounting issue from the main app by:
1. Creating a separate test app without static file dependencies
2. Defining API endpoints inline in the test fixtures
3. Using mocked components to avoid external dependencies

This approach ensures tests run reliably without requiring frontend files or external services.

## Extending Tests

To add new tests:

1. **API Tests**: Add new test methods to the appropriate test class in `test_api_endpoints.py`
2. **New Fixtures**: Add shared test data or mocks to `conftest.py`  
3. **Integration Tests**: Create new test files following the naming pattern `test_*.py`
4. **Markers**: Use existing markers (`@pytest.mark.api`, `@pytest.mark.integration`, etc.) or add new ones in `pyproject.toml`

## Dependencies

Required testing packages (already added to `pyproject.toml`):
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `httpx` - HTTP client for FastAPI testing
- `pytest-mock` - Enhanced mocking capabilities