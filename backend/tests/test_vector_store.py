"""
Unit tests for VectorStore search functionality
Tests ChromaDB integration and search capabilities
"""
import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults helper class"""
    
    def test_from_chroma_with_data(self):
        """Test creating SearchResults from ChromaDB results with data"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key1': 'val1'}, {'key2': 'val2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key1': 'val1'}, {'key2': 'val2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None

    def test_from_chroma_empty(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.is_empty()

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("Test error")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "Test error"
        assert results.is_empty()

    def test_is_empty(self):
        """Test is_empty method"""
        empty_results = SearchResults([], [], [])
        assert empty_results.is_empty()
        
        non_empty_results = SearchResults(['doc'], [{}], [0.1])
        assert not non_empty_results.is_empty()


class TestVectorStore:
    """Test VectorStore functionality"""
    
    def test_init_creates_collections(self, temp_chroma_db):
        """Test that VectorStore initializes with proper collections"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        assert store.max_results == 5
        assert store.course_catalog is not None
        assert store.course_content is not None

    def test_add_course_metadata(self, temp_chroma_db, sample_course):
        """Test adding course metadata to catalog"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Add course metadata
        store.add_course_metadata(sample_course)
        
        # Verify it was added
        results = store.course_catalog.get()
        assert len(results['ids']) == 1
        assert results['ids'][0] == sample_course.title
        
        metadata = results['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link

    def test_add_course_content(self, temp_chroma_db, sample_course_chunks):
        """Test adding course content chunks"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Add course content
        store.add_course_content(sample_course_chunks)
        
        # Verify it was added
        results = store.course_content.get()
        assert len(results['ids']) == 3
        assert len(results['documents']) == 3

    def test_add_empty_course_content(self, temp_chroma_db):
        """Test adding empty course content list"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Should not raise an error
        store.add_course_content([])
        
        results = store.course_content.get()
        assert len(results.get('ids', [])) == 0

    def test_get_existing_course_titles(self, temp_chroma_db, sample_course):
        """Test retrieving existing course titles"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Initially empty
        titles = store.get_existing_course_titles()
        assert titles == []
        
        # Add course and check again
        store.add_course_metadata(sample_course)
        titles = store.get_existing_course_titles()
        assert sample_course.title in titles

    def test_get_course_count(self, temp_chroma_db, sample_course):
        """Test getting course count"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Initially zero
        count = store.get_course_count()
        assert count == 0
        
        # Add course and check again
        store.add_course_metadata(sample_course)
        count = store.get_course_count()
        assert count == 1

    def test_clear_all_data(self, temp_chroma_db, sample_course, sample_course_chunks):
        """Test clearing all data"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Add some data
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Verify data exists
        assert store.get_course_count() == 1
        content_results = store.course_content.get()
        assert len(content_results.get('ids', [])) == 3
        
        # Clear all data
        store.clear_all_data()
        
        # Verify data is gone
        assert store.get_course_count() == 0
        content_results = store.course_content.get()
        assert len(content_results.get('ids', [])) == 0

    def test_resolve_course_name_exact_match(self, temp_chroma_db, sample_course):
        """Test resolving course name with exact match"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        resolved = store._resolve_course_name("Test Course")
        assert resolved == "Test Course"

    def test_resolve_course_name_fuzzy_match(self, temp_chroma_db, sample_course):
        """Test resolving course name with fuzzy matching"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        # Should find the course with partial match
        resolved = store._resolve_course_name("Test")
        assert resolved == "Test Course"

    def test_resolve_course_name_no_match(self, temp_chroma_db):
        """Test resolving course name when no match exists"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        resolved = store._resolve_course_name("Nonexistent Course")
        assert resolved is None

    def test_build_filter_no_filters(self, temp_chroma_db):
        """Test building filter with no parameters"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        filter_dict = store._build_filter(None, None)
        assert filter_dict is None

    def test_build_filter_course_only(self, temp_chroma_db):
        """Test building filter with course title only"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        filter_dict = store._build_filter("Test Course", None)
        assert filter_dict == {"course_title": "Test Course"}

    def test_build_filter_lesson_only(self, temp_chroma_db):
        """Test building filter with lesson number only"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        filter_dict = store._build_filter(None, 1)
        assert filter_dict == {"lesson_number": 1}

    def test_build_filter_both_parameters(self, temp_chroma_db):
        """Test building filter with both course and lesson"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        filter_dict = store._build_filter("Test Course", 1)
        expected = {
            "$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 1}
            ]
        }
        assert filter_dict == expected

    @patch('chromadb.PersistentClient')
    def test_search_with_exception(self, mock_client, temp_chroma_db):
        """Test search method handles exceptions properly"""
        # Mock client to raise exception
        mock_collection = Mock()
        mock_collection.query.side_effect = Exception("Database error")
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.course_content = mock_collection
        
        results = store.search("test query")
        
        assert results.error == "Search error: Database error"
        assert results.is_empty()

    def test_search_course_not_found(self, temp_chroma_db):
        """Test search with non-existent course name"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        results = store.search("test query", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()

    def test_search_successful_with_results(self, temp_chroma_db, sample_course, sample_course_chunks):
        """Test successful search with results"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Add course and content
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Search for content
        results = store.search("introduction")
        
        # Should return results
        assert not results.is_empty()
        assert results.error is None

    def test_get_all_courses_metadata(self, temp_chroma_db, sample_course):
        """Test retrieving all courses metadata with lessons parsed"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        all_courses = store.get_all_courses_metadata()
        
        assert len(all_courses) == 1
        course_meta = all_courses[0]
        assert course_meta['title'] == sample_course.title
        assert course_meta['instructor'] == sample_course.instructor
        assert 'lessons' in course_meta
        assert len(course_meta['lessons']) == 2

    def test_get_course_link(self, temp_chroma_db, sample_course):
        """Test retrieving course link"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        link = store.get_course_link(sample_course.title)
        assert link == sample_course.course_link

    def test_get_course_link_not_found(self, temp_chroma_db):
        """Test retrieving course link for non-existent course"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        link = store.get_course_link("Nonexistent Course")
        assert link is None

    def test_get_lesson_link(self, temp_chroma_db, sample_course):
        """Test retrieving lesson link"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        link = store.get_lesson_link(sample_course.title, 1)
        assert link == "https://example.com/lesson1"

    def test_get_lesson_link_not_found(self, temp_chroma_db, sample_course):
        """Test retrieving lesson link for non-existent lesson"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        store.add_course_metadata(sample_course)
        
        link = store.get_lesson_link(sample_course.title, 999)
        assert link is None

    def test_search_with_custom_limit(self, temp_chroma_db, sample_course, sample_course_chunks):
        """Test search with custom result limit"""
        store = VectorStore(temp_chroma_db, "all-MiniLM-L6-v2", max_results=5)
        
        # Add course and content
        store.add_course_metadata(sample_course)
        store.add_course_content(sample_course_chunks)
        
        # Mock the collection to verify limit is passed correctly
        with patch.object(store.course_content, 'query') as mock_query:
            mock_query.return_value = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            store.search("test query", limit=2)
            
            # Verify custom limit was used
            mock_query.assert_called_once()
            args, kwargs = mock_query.call_args
            assert kwargs['n_results'] == 2