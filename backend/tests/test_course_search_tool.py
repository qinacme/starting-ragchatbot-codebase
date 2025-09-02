"""
Unit tests for CourseSearchTool execute method
Tests the search tool logic independently of VectorStore implementation
"""
import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from models import Course, Lesson


class TestCourseSearchTool:
    """Test cases for CourseSearchTool"""

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly formatted"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

    def test_execute_successful_search(self, mock_vector_store, mock_search_results):
        """Test successful search execution"""
        # Setup mock
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Verify VectorStore was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Test Course - Lesson 1]" in result
        assert "Document 1 content" in result
        assert len(tool.last_sources) == 2

    def test_execute_with_course_filter(self, mock_vector_store, mock_search_results):
        """Test search with course name filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course")
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self, mock_vector_store, mock_search_results):
        """Test search with lesson number filter"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name=None,
            lesson_number=1
        )

    def test_execute_with_both_filters(self, mock_vector_store, mock_search_results):
        """Test search with both course and lesson filters"""
        mock_vector_store.search.return_value = mock_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=1)
        
        mock_vector_store.search.assert_called_once_with(
            query="test query",
            course_name="Test Course",
            lesson_number=1
        )

    def test_execute_empty_results(self, mock_vector_store, mock_empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = mock_empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0

    def test_execute_empty_results_with_filters(self, mock_vector_store, mock_empty_search_results):
        """Test handling of empty results with filters"""
        mock_vector_store.search.return_value = mock_empty_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query", course_name="Test Course", lesson_number=1)
        
        assert "No relevant content found in course 'Test Course' in lesson 1" in result

    def test_execute_error_handling(self, mock_vector_store, mock_error_search_results):
        """Test handling of search errors"""
        mock_vector_store.search.return_value = mock_error_search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert result == "Test error message"

    def test_format_results_with_lesson_links(self, mock_vector_store):
        """Test result formatting includes lesson links"""
        # Create search results with metadata
        search_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Test Course", "lesson_number": 1},
                {"course_title": "Test Course", "lesson_number": 2}
            ],
            distances=[0.1, 0.2]
        )
        
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson1",
            "https://example.com/lesson2"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        # Check that sources were populated with links
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Test Course - Lesson 1"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson1"
        assert tool.last_sources[1]["text"] == "Test Course - Lesson 2"
        assert tool.last_sources[1]["link"] == "https://example.com/lesson2"

    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test result formatting when lesson numbers are missing"""
        search_results = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1]
        )
        
        mock_vector_store.search.return_value = search_results
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")
        
        assert "[Test Course]" in result
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None

    def test_sources_tracking_and_reset(self, mock_vector_store, mock_search_results):
        """Test that sources are properly tracked and can be reset"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        tool = CourseSearchTool(mock_vector_store)
        
        # First search
        tool.execute("first query")
        assert len(tool.last_sources) == 2
        
        # Sources should persist until explicitly reset
        first_sources = tool.last_sources.copy()
        
        # Second search
        tool.execute("second query")
        assert len(tool.last_sources) == 2
        
        # Sources should be updated, not accumulated
        assert tool.last_sources != first_sources or mock_search_results.documents == ["Document 1 content", "Document 2 content"]


class TestToolManager:
    """Test cases for ToolManager"""
    
    def test_register_tool(self, mock_vector_store):
        """Test registering a tool"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        definitions = manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_execute_tool_success(self, mock_vector_store, mock_search_results):
        """Test successful tool execution through manager"""
        mock_vector_store.search.return_value = mock_search_results
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "[Test Course - Lesson 1]" in result

    def test_execute_nonexistent_tool(self, mock_vector_store):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"

    def test_get_last_sources(self, mock_vector_store, mock_search_results):
        """Test retrieving sources from last search"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search
        manager.execute_tool("search_course_content", query="test query")
        
        # Get sources
        sources = manager.get_last_sources()
        assert len(sources) == 2

    def test_reset_sources(self, mock_vector_store, mock_search_results):
        """Test resetting sources"""
        mock_vector_store.search.return_value = mock_search_results
        mock_vector_store.get_lesson_link.return_value = None
        
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute search to populate sources
        manager.execute_tool("search_course_content", query="test query")
        assert len(manager.get_last_sources()) == 2
        
        # Reset sources
        manager.reset_sources()
        assert len(manager.get_last_sources()) == 0

    def test_register_tool_without_name_raises_error(self, mock_vector_store):
        """Test that registering a tool without name raises ValueError"""
        manager = ToolManager()
        
        # Create a mock tool without a name in definition
        mock_tool = Mock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            manager.register_tool(mock_tool)