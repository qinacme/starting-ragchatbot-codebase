"""
Test fixtures and configuration for the RAG system tests.
Combined fixtures for both unit tests and API integration tests.
"""

import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from typing import Dict, List, Any, Optional, Generator, AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from config import Config


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


# Event loop fixture for async tests
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Temporary directory fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# Configuration fixtures
@pytest.fixture
def test_config(temp_chroma_db):
    """Create a test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_db
    config.CHROMA_DB_PATH = temp_chroma_db  # Support both naming conventions
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    return config


# Course data fixtures
@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Introduction",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Advanced Topics",
                lesson_link="https://example.com/lesson2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Test Course Lesson 1 content: This is the introduction lesson content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Course Test Course Lesson 1 content: This is more introduction content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Course Test Course Lesson 2 content: This is advanced topic content",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_course_data():
    """Mock course data for testing document processing"""
    return """Course Title: Test Course
Course Link: https://example.com/course
Course Instructor: Test Instructor

Lesson 1: Introduction
Lesson Link: https://example.com/lesson1
This is the introduction lesson content. It explains the basics of the course material.

Lesson 2: Advanced Topics  
Lesson Link: https://example.com/lesson2
This lesson covers advanced topics in detail. It builds on the introduction material.
"""


# Vector store and search fixtures
@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore for isolated testing"""
    mock = Mock(spec=VectorStore)
    mock.max_results = 5
    return mock


@pytest.fixture
def mock_search_results():
    """Create mock search results"""
    return SearchResults(
        documents=["Document 1 content", "Document 2 content"],
        metadata=[
            {"course_title": "Test Course", "lesson_number": 1},
            {"course_title": "Test Course", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def mock_empty_search_results():
    """Create empty search results"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def mock_error_search_results():
    """Create error search results"""
    return SearchResults.empty("Test error message")


# AI generator and tool fixtures
@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "Test response"
    mock_response.stop_reason = "end_turn"
    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_anthropic_tool_response():
    """Create a mock Anthropic response with tool use"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"

    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "test-tool-id"
    tool_block.input = {"query": "test query"}

    mock_response.content = [tool_block]
    return mock_response


@pytest.fixture
def sample_tool_definitions():
    """Sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "course_name": {"type": "string", "description": "Course title"},
                    "lesson_number": {"type": "integer", "description": "Lesson number"},
                },
                "required": ["query"],
            },
        }
    ]


class MockToolManager:
    """Mock tool manager for testing"""

    def __init__(self):
        self.tools = {}
        self.last_sources = []

    def register_tool(self, tool):
        tool_def = tool.get_tool_definition()
        self.tools[tool_def["name"]] = tool

    def get_tool_definitions(self):
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name in self.tools:
            return self.tools[tool_name].execute(**kwargs)
        return f"Tool '{tool_name}' not found"

    def get_last_sources(self):
        return self.last_sources

    def reset_sources(self):
        self.last_sources = []


@pytest.fixture
def mock_tool_manager():
    """Create a mock tool manager"""
    return MockToolManager()


# RAG system fixtures
@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for testing."""
    from rag_system import RAGSystem

    mock_rag = Mock(spec=RAGSystem)
    mock_rag.query = AsyncMock(
        return_value=("Test answer", ["Test source 1", "Test source 2"])
    )
    mock_rag.get_course_analytics = Mock(
        return_value={"total_courses": 2, "course_titles": ["Course 1", "Course 2"]}
    )
    mock_rag.session_manager = Mock()
    mock_rag.session_manager.create_session = Mock(return_value="test-session-id")
    mock_rag.session_manager.clear_session = Mock()
    mock_rag.add_course_folder = Mock(return_value=(2, 10))
    return mock_rag


# FastAPI test fixtures
@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting."""
    app = FastAPI(title="Test Course Materials RAG System", root_path="")

    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

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

            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            from fastapi import HTTPException

            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
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


# Sample data fixtures
@pytest.fixture
def sample_query_request():
    """Sample query request data."""
    return {"query": "What is machine learning?", "session_id": "test-session-123"}


@pytest.fixture
def sample_query_response():
    """Sample query response data."""
    return {
        "answer": "Machine learning is a subset of artificial intelligence...",
        "sources": ["Course 1 Lesson 2", "Course 2 Lesson 1"],
        "session_id": "test-session-123",
    }


@pytest.fixture
def sample_course_stats():
    """Sample course statistics data."""
    return {
        "total_courses": 3,
        "course_titles": ["Introduction to AI", "Machine Learning Basics", "Deep Learning"],
    }


# Environment and cleanup fixtures
@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    with patch.dict(
        os.environ,
        {"ANTHROPIC_API_KEY": "test-api-key", "CHROMA_DB_PATH": "./test_chroma_db"},
    ):
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
