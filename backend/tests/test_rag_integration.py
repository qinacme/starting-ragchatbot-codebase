"""
Integration tests for the full RAG system
Tests the complete flow from user query to response
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system import RAGSystem
from config import Config
from models import Course, Lesson, CourseChunk
from document_processor import DocumentProcessor


class TestRAGSystemIntegration:
    """Integration tests for RAG System"""

    @pytest.fixture
    def integration_config(self):
        """Create config for integration testing"""
        temp_dir = tempfile.mkdtemp()
        config = Config()
        config.CHROMA_PATH = temp_dir
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.CHUNK_SIZE = 400  # Smaller for testing
        config.CHUNK_OVERLAP = 50
        config.MAX_RESULTS = 3
        config.MAX_HISTORY = 2
        yield config
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_course_document(self, tmp_path):
        """Create a sample course document file"""
        content = """Course Title: Test Integration Course
Course Link: https://example.com/integration-course
Course Instructor: Integration Tester

Lesson 1: Introduction to Testing
Lesson Link: https://example.com/lesson1
This is the introduction lesson content. It explains the basics of integration testing.
Integration testing verifies that different components work together correctly.

Lesson 2: Advanced Integration Patterns
Lesson Link: https://example.com/lesson2
This lesson covers advanced integration testing patterns and best practices.
It includes topics like mocking, stubbing, and test doubles for effective testing.
"""
        doc_file = tmp_path / "test_course.txt"
        doc_file.write_text(content)
        return str(doc_file)

    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_without_tools(self, mock_anthropic, integration_config, sample_course_document):
        """Test complete query flow when Claude doesn't use tools"""
        # Setup Anthropic mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "This is a general response without using search tools."
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Initialize RAG system
        rag_system = RAGSystem(integration_config)
        
        # Add course document
        course, num_chunks = rag_system.add_course_document(sample_course_document)
        assert course is not None
        assert num_chunks > 0
        
        # Make query
        response, sources = rag_system.query("What is Python?")
        
        # Verify response
        assert response == "This is a general response without using search tools."
        assert sources == []  # No sources since no tools were used
        
        # Verify Claude was called with tools available
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert len(call_args[1]["tools"]) == 2  # search_course_content and get_course_outline

    @patch('ai_generator.anthropic.Anthropic')
    def test_full_query_flow_with_search_tool(self, mock_anthropic, integration_config, sample_course_document):
        """Test complete query flow when Claude uses search tool"""
        # Setup Anthropic mock for tool use
        mock_client = Mock()
        
        # First response with tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test-tool-id"
        tool_block.input = {"query": "integration testing"}
        tool_response.content = [tool_block]
        
        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Based on the course materials, integration testing verifies that different components work together correctly."
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Initialize RAG system and add document
        rag_system = RAGSystem(integration_config)
        course, num_chunks = rag_system.add_course_document(sample_course_document)
        assert course is not None
        assert num_chunks > 0
        
        # Make query that should trigger search
        response, sources = rag_system.query("What is integration testing?")
        
        # Verify tool was used and response generated
        assert "integration testing" in response.lower()
        assert len(sources) > 0  # Should have sources from search
        
        # Verify tool execution happened
        assert mock_client.messages.create.call_count == 2

    @patch('ai_generator.anthropic.Anthropic')
    def test_session_management(self, mock_anthropic, integration_config, sample_course_document):
        """Test session management and conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(integration_config)
        rag_system.add_course_document(sample_course_document)
        
        # First query with new session
        response1, sources1 = rag_system.query("First question", session_id="test-session")
        
        # Second query with same session - should include history
        response2, sources2 = rag_system.query("Second question", session_id="test-session")
        
        # Verify history was included in second call
        assert mock_client.messages.create.call_count == 2
        
        # Check that second call includes conversation history
        second_call_args = mock_client.messages.create.call_args_list[1]
        system_content = second_call_args[1]["system"]
        assert "Previous conversation:" in system_content

    def test_document_processing_and_storage(self, integration_config, sample_course_document):
        """Test that documents are properly processed and stored"""
        rag_system = RAGSystem(integration_config)
        
        # Add document
        course, num_chunks = rag_system.add_course_document(sample_course_document)
        
        # Verify course was processed
        assert course.title == "Test Integration Course"
        assert course.instructor == "Integration Tester"
        assert len(course.lessons) == 2
        assert course.lessons[0].title == "Introduction to Testing"
        assert course.lessons[1].title == "Advanced Integration Patterns"
        
        # Verify chunks were created
        assert num_chunks > 0
        
        # Verify data was stored in vector store
        existing_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Test Integration Course" in existing_titles
        
        course_count = rag_system.vector_store.get_course_count()
        assert course_count == 1

    def test_add_course_folder(self, integration_config, tmp_path):
        """Test adding multiple course documents from folder"""
        # Create multiple course files
        course1_content = """Course Title: Course One
Course Instructor: Teacher One

Lesson 1: First Topic
This is content for course one.
"""
        
        course2_content = """Course Title: Course Two  
Course Instructor: Teacher Two

Lesson 1: Different Topic
This is content for course two.
"""
        
        (tmp_path / "course1.txt").write_text(course1_content)
        (tmp_path / "course2.txt").write_text(course2_content)
        
        rag_system = RAGSystem(integration_config)
        
        # Add folder
        num_courses, num_chunks = rag_system.add_course_folder(str(tmp_path))
        
        assert num_courses == 2
        assert num_chunks > 0
        
        # Verify both courses were added
        existing_titles = rag_system.vector_store.get_existing_course_titles()
        assert "Course One" in existing_titles
        assert "Course Two" in existing_titles

    def test_duplicate_course_handling(self, integration_config, sample_course_document):
        """Test that duplicate courses are not re-processed"""
        rag_system = RAGSystem(integration_config)
        
        # Add document first time
        course1, chunks1 = rag_system.add_course_document(sample_course_document)
        
        # Add same document again
        course2, chunks2 = rag_system.add_course_document(sample_course_document)
        
        # Second add should be skipped
        assert course1 is not None
        assert chunks1 > 0
        # The behavior depends on implementation - it might return None or the existing course
        
        # Verify only one copy in vector store
        count = rag_system.vector_store.get_course_count()
        assert count == 1

    @patch('ai_generator.anthropic.Anthropic')
    def test_course_outline_tool_integration(self, mock_anthropic, integration_config, sample_course_document):
        """Test integration of course outline tool"""
        # Setup mock for outline tool use
        mock_client = Mock()
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "get_course_outline"
        tool_block.id = "outline-tool-id"
        tool_block.input = {"course_title": "Test Integration"}
        tool_response.content = [tool_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Here's the course outline with lessons."
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(integration_config)
        rag_system.add_course_document(sample_course_document)
        
        # Query for course outline
        response, sources = rag_system.query("Show me the outline for Test Integration course")
        
        assert "outline" in response.lower()
        assert mock_client.messages.create.call_count == 2

    def test_error_handling_in_document_processing(self, integration_config, tmp_path):
        """Test error handling when document processing fails"""
        # Create invalid document file
        invalid_file = tmp_path / "invalid.txt"
        invalid_file.write_text("")  # Empty file
        
        rag_system = RAGSystem(integration_config)
        
        # Should handle error gracefully
        course, chunks = rag_system.add_course_document(str(invalid_file))
        
        # Behavior may vary - could return None or empty course
        # Main thing is it shouldn't crash
        assert chunks == 0  # No chunks from empty file

    def test_get_course_analytics(self, integration_config, sample_course_document):
        """Test course analytics functionality"""
        rag_system = RAGSystem(integration_config)
        
        # Initially empty
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 0
        assert analytics["course_titles"] == []
        
        # Add course
        rag_system.add_course_document(sample_course_document)
        
        # Check analytics
        analytics = rag_system.get_course_analytics()
        assert analytics["total_courses"] == 1
        assert "Test Integration Course" in analytics["course_titles"]

    @patch('ai_generator.anthropic.Anthropic')
    def test_search_with_course_and_lesson_filters(self, mock_anthropic, integration_config, sample_course_document):
        """Test search functionality with course and lesson filters"""
        # Setup mock for filtered search
        mock_client = Mock()
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "search-tool-id"
        tool_block.input = {
            "query": "testing patterns",
            "course_name": "Test Integration", 
            "lesson_number": 2
        }
        tool_response.content = [tool_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Advanced integration patterns include mocking and stubbing."
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(integration_config)
        rag_system.add_course_document(sample_course_document)
        
        # This would normally be triggered by Claude's decision to use filters
        # We're testing that the system can handle such requests
        response, sources = rag_system.query("What are testing patterns in lesson 2?")
        
        assert mock_client.messages.create.call_count == 2
        assert "patterns" in response.lower()

    @patch('ai_generator.anthropic.Anthropic')  
    def test_source_tracking_and_reset(self, mock_anthropic, integration_config, sample_course_document):
        """Test that sources are properly tracked and reset between queries"""
        # Setup mock
        mock_client = Mock()
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "search-tool-id"
        tool_block.input = {"query": "integration"}
        tool_response.content = [tool_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Response with sources"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        rag_system = RAGSystem(integration_config)
        rag_system.add_course_document(sample_course_document)
        
        # First query
        response1, sources1 = rag_system.query("First query about integration")
        
        # Should have sources
        assert len(sources1) > 0
        
        # Reset mock for second call
        mock_client.messages.create.side_effect = [tool_response, final_response]
        
        # Second query  
        response2, sources2 = rag_system.query("Second query about integration")
        
        # Should have fresh sources (not accumulated)
        assert len(sources2) > 0