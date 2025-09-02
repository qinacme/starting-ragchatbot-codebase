"""
Shared test fixtures and utilities for RAG system tests
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, MagicMock
from typing import Dict, List, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import Course, Lesson, CourseChunk
from vector_store import VectorStore, SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from config import Config

@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Test Course",
        course_link="https://example.com/course",
        instructor="Test Instructor",
        lessons=[
            Lesson(lesson_number=1, title="Introduction", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link="https://example.com/lesson2")
        ]
    )

@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Test Course Lesson 1 content: This is the introduction lesson content",
            course_title="Test Course",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Test Course Lesson 1 content: This is more introduction content",
            course_title="Test Course", 
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Test Course Lesson 2 content: This is advanced topic content",
            course_title="Test Course",
            lesson_number=2,
            chunk_index=2
        )
    ]

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
            {"course_title": "Test Course", "lesson_number": 2}
        ],
        distances=[0.1, 0.2]
    )

@pytest.fixture
def mock_empty_search_results():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )

@pytest.fixture
def mock_error_search_results():
    """Create error search results"""
    return SearchResults.empty("Test error message")

@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config(temp_chroma_db):
    """Create a test configuration"""
    config = Config()
    config.CHROMA_PATH = temp_chroma_db
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    return config

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
                    "lesson_number": {"type": "integer", "description": "Lesson number"}
                },
                "required": ["query"]
            }
        }
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