"""
API endpoint tests for the RAG system FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import Mock, AsyncMock, patch


class TestQueryEndpoint:
    """Test the /api/query endpoint."""
    
    @pytest.mark.api
    def test_query_with_session_id(self, client, sample_query_request):
        """Test query endpoint with existing session ID."""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == sample_query_request["session_id"]
        assert isinstance(data["sources"], list)
        assert len(data["sources"]) >= 0
    
    @pytest.mark.api
    def test_query_without_session_id(self, client):
        """Test query endpoint without session ID (should create new session)."""
        request_data = {"query": "What is artificial intelligence?"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-id"  # From mock
    
    @pytest.mark.api
    def test_query_invalid_request_missing_query(self, client):
        """Test query endpoint with missing query field."""
        request_data = {"session_id": "test-123"}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.api
    def test_query_empty_query(self, client):
        """Test query endpoint with empty query string."""
        request_data = {"query": ""}
        
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
    
    @pytest.mark.api
    def test_query_response_model_validation(self, client, sample_query_request):
        """Test that response matches the expected model."""
        response = client.post("/api/query", json=sample_query_request)
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure matches QueryResponse model
        required_fields = ["answer", "sources", "session_id"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)
        
        # Validate sources are strings
        for source in data["sources"]:
            assert isinstance(source, str)
    
    @pytest.mark.api
    def test_query_error_handling(self, client, mock_rag_system):
        """Test query endpoint error handling when RAG system fails."""
        # Configure mock to raise exception
        mock_rag_system.query.side_effect = Exception("Test error")
        
        request_data = {"query": "What is AI?"}
        response = client.post("/api/query", json=request_data)
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]


class TestCoursesEndpoint:
    """Test the /api/courses endpoint."""
    
    @pytest.mark.api
    def test_get_courses_success(self, client, sample_course_stats):
        """Test successful retrieval of course statistics."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] == 2  # From mock
        assert len(data["course_titles"]) == 2
    
    @pytest.mark.api
    def test_courses_response_model_validation(self, client):
        """Test that response matches the CourseStats model."""
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        # Validate response structure matches CourseStats model
        required_fields = ["total_courses", "course_titles"]
        for field in required_fields:
            assert field in data
        
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert data["total_courses"] >= 0
        
        # Validate course titles are strings
        for title in data["course_titles"]:
            assert isinstance(title, str)
    
    @pytest.mark.api
    def test_courses_empty_state(self, client, mock_rag_system):
        """Test courses endpoint when no courses are available."""
        # Configure mock to return empty state
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = client.get("/api/courses")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["total_courses"] == 0
        assert data["course_titles"] == []
    
    @pytest.mark.api
    def test_courses_error_handling(self, client, mock_rag_system):
        """Test courses endpoint error handling when analytics fail."""
        # Configure mock to raise exception
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = client.get("/api/courses")
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics error" in data["detail"]


class TestRootEndpoint:
    """Test the root endpoint."""
    
    @pytest.mark.api
    def test_root_endpoint(self, client):
        """Test the root endpoint returns basic API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert isinstance(data["message"], str)
        assert "RAG System" in data["message"]


class TestRequestValidation:
    """Test request validation and error handling."""
    
    @pytest.mark.api
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request body."""
        response = client.post(
            "/api/query", 
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        
        assert response.status_code == 422
    
    @pytest.mark.api
    def test_malformed_request_body(self, client):
        """Test handling of malformed request body."""
        response = client.post("/api/query", json={"invalid_field": "value"})
        
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class TestEndpointIntegration:
    """Integration tests for endpoint interactions."""
    
    @pytest.mark.integration
    def test_query_then_courses_workflow(self, client):
        """Test a typical user workflow: query, then check courses."""
        # First, make a query
        query_response = client.post("/api/query", json={
            "query": "What courses are available?"
        })
        
        assert query_response.status_code == 200
        query_data = query_response.json()
        assert "session_id" in query_data
        
        # Then check available courses
        courses_response = client.get("/api/courses")
        
        assert courses_response.status_code == 200
        courses_data = courses_response.json()
        assert courses_data["total_courses"] >= 0
    
    @pytest.mark.integration
    def test_multiple_queries_same_session(self, client):
        """Test multiple queries with the same session ID."""
        session_id = "test-session-multiple"
        
        # First query
        response1 = client.post("/api/query", json={
            "query": "What is AI?",
            "session_id": session_id
        })
        
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["session_id"] == session_id
        
        # Second query with same session
        response2 = client.post("/api/query", json={
            "query": "What is ML?",
            "session_id": session_id
        })
        
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["session_id"] == session_id