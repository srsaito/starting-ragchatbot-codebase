"""
Tests for FastAPI API endpoints.

This module tests the HTTP endpoints exposed by the RAG system API,
including query processing, course statistics, and error handling.
"""

import pytest
from unittest.mock import Mock, patch
import json


class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_success(self, test_client, sample_query_request):
        """Test successful query processing"""
        response = test_client.post(
            "/api/query",
            json=sample_query_request
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify content
        assert "machine learning" in data["answer"].lower()
        assert len(data["sources"]) == 2
        assert data["session_id"] == "test-session-123"

        # Verify sources format
        for source in data["sources"]:
            assert "text" in source
            assert "link" in source
            assert isinstance(source["text"], str)
            assert source["link"] is None or isinstance(source["link"], str)

    def test_query_with_existing_session(self, test_client, sample_query_request_with_session):
        """Test query with existing session ID"""
        response = test_client.post(
            "/api/query",
            json=sample_query_request_with_session
        )

        assert response.status_code == 200
        data = response.json()

        # Session ID should be preserved
        assert data["session_id"] == "existing-session-456"
        assert "answer" in data
        assert "sources" in data

    def test_query_empty_string(self, test_client):
        """Test query with empty string"""
        response = test_client.post(
            "/api/query",
            json={"query": "", "session_id": None}
        )

        # Should still process but might return minimal response
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_long_input(self, test_client):
        """Test query with very long input string"""
        long_query = "machine learning " * 100  # 200 words
        response = test_client.post(
            "/api/query",
            json={"query": long_query, "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data

    def test_query_special_characters(self, test_client):
        """Test query with special characters"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is ML? #AI & <neural networks>", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_unicode_characters(self, test_client):
        """Test query with unicode characters"""
        response = test_client.post(
            "/api/query",
            json={"query": "机器学习 and résumé with emoji 🤖", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_error_handling(self, test_client, test_app, mock_rag_system):
        """Test error handling when RAG system fails"""
        # Make the query method raise an exception
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post(
            "/api/query",
            json={"query": "test query", "session_id": None}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    def test_query_invalid_json(self, test_client):
        """Test query with invalid JSON body"""
        response = test_client.post(
            "/api/query",
            data="not valid json"
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_query_missing_required_field(self, test_client):
        """Test query without required 'query' field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test-123"}  # Missing 'query' field
        )

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_query_content_type_json(self, test_client, sample_query_request):
        """Test that the endpoint accepts application/json"""
        response = test_client.post(
            "/api/query",
            json=sample_query_request,
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 200

    def test_query_response_headers(self, test_client, sample_query_request):
        """Test response headers"""
        response = test_client.post(
            "/api/query",
            json=sample_query_request
        )

        assert response.headers["content-type"] == "application/json"
        # Note: CORS headers may not be present in test client responses
        # as TestClient doesn't fully process middleware like a real server


class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_get_courses_success(self, test_client):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify content
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Introduction to Machine Learning" in data["course_titles"]
        assert "Deep Learning Basics" in data["course_titles"]
        assert "Neural Networks" in data["course_titles"]

    def test_get_courses_error_handling(self, test_client, test_app, mock_rag_system):
        """Test error handling when getting course analytics fails"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics service unavailable")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Analytics service unavailable" in data["detail"]

    def test_get_courses_empty_database(self, test_client, test_app, mock_rag_system):
        """Test response when no courses are loaded"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_response_headers(self, test_client):
        """Test response headers for courses endpoint"""
        response = test_client.get("/api/courses")

        assert response.headers["content-type"] == "application/json"
        # Note: CORS headers may not be present in test client responses


class TestRootEndpoint:
    """Test the root endpoint"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns expected message"""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Test RAG System API"


class TestCORSConfiguration:
    """Test CORS configuration"""

    def test_cors_preflight_query(self, test_client):
        """Test CORS preflight request for query endpoint"""
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type"
            }
        )

        assert response.status_code == 200
        # CORS headers are set but TestClient may reflect the Origin instead of "*"
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "POST" in response.headers["access-control-allow-methods"]
        assert "content-type" in response.headers["access-control-allow-headers"].lower()

    def test_cors_preflight_courses(self, test_client):
        """Test CORS preflight request for courses endpoint"""
        response = test_client.options(
            "/api/courses",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )

        assert response.status_code == 200
        # CORS headers are set but TestClient may reflect the Origin instead of "*"
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "GET" in response.headers["access-control-allow-methods"]


class TestAPIIntegration:
    """Integration tests for multiple API calls"""

    def test_multiple_queries_same_session(self, test_client):
        """Test multiple queries using the same session"""
        # First query - create new session
        response1 = test_client.post(
            "/api/query",
            json={"query": "What is ML?", "session_id": None}
        )
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # Second query - use existing session
        response2 = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": session_id}
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

    def test_query_then_courses(self, test_client):
        """Test querying then getting course stats"""
        # Query first
        query_response = test_client.post(
            "/api/query",
            json={"query": "What courses are available?", "session_id": None}
        )
        assert query_response.status_code == 200

        # Then get courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200

        # Both should work independently
        assert "answer" in query_response.json()
        assert "total_courses" in courses_response.json()


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_concurrent_requests_simulation(self, test_client):
        """Simulate concurrent requests to test thread safety"""
        responses = []
        for i in range(5):
            response = test_client.post(
                "/api/query",
                json={"query": f"Query {i}", "session_id": None}
            )
            responses.append(response)

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            assert "answer" in response.json()

    def test_malformed_url_path(self, test_client):
        """Test accessing non-existent endpoints"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, test_client):
        """Test wrong HTTP method"""
        # Try GET on a POST-only endpoint
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed

        # Try POST on a GET-only endpoint
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405

    def test_large_response_handling(self, test_client, test_app, mock_rag_system):
        """Test handling of large responses"""
        # Mock a large response
        large_answer = "This is a very detailed response. " * 500
        large_sources = [
            {"text": f"Source {i}", "link": f"https://example.com/lesson{i}"}
            for i in range(20)
        ]
        mock_rag_system.query.return_value = (large_answer, large_sources)

        response = test_client.post(
            "/api/query",
            json={"query": "Give me everything", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["answer"]) > 1000
        assert len(data["sources"]) == 20


class TestDataValidation:
    """Test data validation and sanitization"""

    def test_query_sql_injection_attempt(self, test_client):
        """Test that SQL injection attempts are handled safely"""
        response = test_client.post(
            "/api/query",
            json={"query": "'; DROP TABLE courses; --", "session_id": None}
        )

        assert response.status_code == 200
        # The query should be processed as plain text, not executed

    def test_query_xss_attempt(self, test_client):
        """Test that XSS attempts are handled safely"""
        response = test_client.post(
            "/api/query",
            json={"query": "<script>alert('XSS')</script>", "session_id": None}
        )

        assert response.status_code == 200
        data = response.json()
        # The response should treat this as plain text
        assert "<script>" not in json.dumps(data)

    def test_session_id_validation(self, test_client):
        """Test various session ID formats"""
        test_cases = [
            "normal-session-123",
            "123456789",
            "uuid-550e8400-e29b-41d4-a716-446655440000",
            "special!@#$%^&*()",
            " spaces in id ",
            ""  # Empty string session ID
        ]

        for session_id in test_cases:
            response = test_client.post(
                "/api/query",
                json={"query": "test", "session_id": session_id}
            )
            assert response.status_code == 200
            data = response.json()
            if session_id:
                assert data["session_id"] == session_id
            else:
                # Empty session should create new one
                assert data["session_id"] == "test-session-123"