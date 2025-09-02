"""
Test fixtures and configuration for the RAG system tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, List, Generator, AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel
from typing import List, Optional


# Define API models locally to avoid import issues
class QueryRequest(BaseModel):
    """Request model for course queries"""
    query: str
    session_id: Optional[str] = None

class QueryResponse(BaseModel):
    """Response model for course queries"""
    answer: str
    sources: List[str]
    session_id: str

class CourseStats(BaseModel):
    """Response model for course statistics"""
    total_courses: int
    course_titles: List[str]


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temporary paths."""
    # Import config locally to avoid path issues
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from config import Config
    
    config = Config()
    config.CHROMA_DB_PATH = str(temp_dir / "test_chroma_db")
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    config.MAX_RESULTS = 3
    config.MAX_HISTORY = 1
    return config


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for testing."""
    # Import locally to avoid path issues
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from rag_system import RAGSystem
    
    mock_rag = Mock(spec=RAGSystem)
    mock_rag.query = AsyncMock(return_value=("Test answer", ["Test source 1", "Test source 2"]))
    mock_rag.get_course_analytics = Mock(return_value={
        "total_courses": 2,
        "course_titles": ["Course 1", "Course 2"]
    })
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session = Mock(return_value="test-session-id")
    mock_rag.add_course_folder = Mock(return_value=(2, 10))
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting."""
    app = FastAPI(title="Test Course Materials RAG System", root_path="")
    
    # Add middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"]
    )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Define API endpoints inline to avoid import issues
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = await mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/")
    async def read_root():
        return {"message": "Course Materials RAG System API"}
    
    return app


@pytest.fixture
def client(test_app):
    """Create a test client for the FastAPI app."""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {
        "query": "What is machine learning?",
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_query_response():
    """Sample query response data."""
    return {
        "answer": "Machine learning is a subset of artificial intelligence...",
        "sources": ["Course 1 Lesson 2", "Course 2 Lesson 1"],
        "session_id": "test-session-123"
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics data."""
    return {
        "total_courses": 3,
        "course_titles": ["Introduction to AI", "Machine Learning Basics", "Deep Learning"]
    }


@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'test-api-key',
        'CHROMA_DB_PATH': './test_chroma_db'
    }):
        yield


@pytest.fixture(autouse=True)
def cleanup_test_files():
    """Cleanup test files after each test."""
    yield
    # Cleanup logic can be added here if needed
    test_db_path = Path("./test_chroma_db")
    if test_db_path.exists():
        import shutil
        shutil.rmtree(test_db_path, ignore_errors=True)