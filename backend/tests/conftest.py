"""
Pytest configuration and fixtures for RAG system testing.
"""

import os
import shutil
import sys
import tempfile
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add backend to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course, CourseChunk, Lesson
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults, VectorStore


@pytest.fixture
def temp_chroma_path():
    """Create a temporary directory for ChromaDB testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_course():
    """Sample course for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is ML?",
                lesson_link="https://example.com/lesson1",
            ),
            Lesson(
                lesson_number=2,
                title="Supervised Learning",
                lesson_link="https://example.com/lesson2",
            ),
            Lesson(
                lesson_number=3,
                title="Unsupervised Learning",
                lesson_link="https://example.com/lesson3",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Supervised learning uses labeled training data to learn a mapping function.",
            course_title=sample_course.title,
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Unsupervised learning finds hidden patterns in data without labels.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    mock = Mock(spec=VectorStore)

    # Default successful search results
    mock.search.return_value = SearchResults(
        documents=["Sample content about machine learning", "More ML content"],
        metadata=[
            {"course_title": "Introduction to Machine Learning", "lesson_number": 1},
            {"course_title": "Introduction to Machine Learning", "lesson_number": 2},
        ],
        distances=[0.1, 0.2],
    )

    mock.get_lesson_link.return_value = "https://example.com/lesson1"
    mock.get_course_outline.return_value = {
        "course_title": "Introduction to Machine Learning",
        "course_link": "https://example.com/ml-course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "What is ML?"},
            {"lesson_number": 2, "lesson_title": "Supervised Learning"},
        ],
    }

    return mock


@pytest.fixture
def mock_vector_store_empty():
    """Mock VectorStore that returns empty results"""
    mock = Mock(spec=VectorStore)
    mock.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
    return mock


@pytest.fixture
def mock_vector_store_error():
    """Mock VectorStore that returns error results"""
    mock = Mock(spec=VectorStore)
    mock.search.return_value = SearchResults.empty("Search error occurred")
    return mock


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool with mocked VectorStore"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI generator"""
    mock_client = Mock()

    # Mock successful response without tools
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a response from Claude")]

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_tool_use():
    """Mock Anthropic client that uses tools"""
    mock_client = Mock()

    # Mock initial response with tool use
    mock_initial_response = Mock()
    mock_initial_response.stop_reason = "tool_use"

    # Mock tool use content block
    mock_tool_use = Mock()
    mock_tool_use.type = "tool_use"
    mock_tool_use.name = "search_course_content"
    mock_tool_use.id = "tool_12345"
    mock_tool_use.input = {"query": "machine learning"}

    mock_initial_response.content = [mock_tool_use]

    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.stop_reason = "end_turn"
    mock_final_response.content = [
        Mock(text="Based on the course content, machine learning is...")
    ]

    # Set up the call sequence
    mock_client.messages.create.side_effect = [
        mock_initial_response,
        mock_final_response,
    ]

    return mock_client


@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-key-12345"
    config.MAX_RESULTS = 5  # Fix the MAX_RESULTS=0 issue for testing
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def test_config_zero_results():
    """Test configuration with MAX_RESULTS=0 to reproduce the bug"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-key-12345"
    config.MAX_RESULTS = 0  # This is the problematic setting
    config.CHROMA_PATH = "./test_chroma_db"
    return config


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing"""
    mock = Mock(spec=ToolManager)
    mock.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
            },
        }
    ]
    mock.execute_tool.return_value = "Mock search results"
    mock.get_last_sources.return_value = [
        {"text": "Test Course", "link": "https://example.com"}
    ]
    mock.reset_sources.return_value = None
    return mock


class MockSessionManager:
    """Mock session manager for testing"""

    def __init__(self):
        self.sessions = {}

    def create_session(self):
        return "test-session-123"

    def get_conversation_history(self, session_id):
        return "Previous conversation context..."

    def add_exchange(self, session_id, query, response):
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append({"query": query, "response": response})


@pytest.fixture
def mock_session_manager():
    """Mock session manager fixture"""
    return MockSessionManager()


# Helper functions for test assertions


def assert_search_results_format(result_str: str):
    """Assert that search results are properly formatted"""
    assert isinstance(result_str, str)
    assert len(result_str) > 0
    # Should contain course/lesson headers
    assert "[" in result_str and "]" in result_str


def assert_sources_format(sources: List[Dict]):
    """Assert that sources are properly formatted"""
    assert isinstance(sources, list)
    for source in sources:
        assert "text" in source
        assert isinstance(source["text"], str)
        # link is optional but if present should be string or None
        if "link" in source:
            assert source["link"] is None or isinstance(source["link"], str)


# API Testing Fixtures

@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API testing"""
    mock = Mock()

    # Mock query method
    mock.query.return_value = (
        "Based on the course materials, machine learning is a subset of AI.",
        [
            {"text": "Introduction to Machine Learning - Lesson 1", "link": "https://example.com/lesson1"},
            {"text": "Introduction to Machine Learning - Lesson 2", "link": "https://example.com/lesson2"}
        ]
    )

    # Mock get_course_analytics
    mock.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Introduction to Machine Learning", "Deep Learning Basics", "Neural Networks"]
    }

    # Mock session manager
    mock.session_manager.create_session.return_value = "test-session-123"

    # Mock add_course_folder
    mock.add_course_folder.return_value = (2, 10)  # 2 courses, 10 chunks

    return mock


@pytest.fixture
def test_app(mock_rag_system):
    """Create test FastAPI app without static file mounting"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app
    app = FastAPI(title="Test RAG System")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class SourceItem(BaseModel):
        text: str
        link: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceItem]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources = mock_rag_system.query(request.query, session_id)

            formatted_sources = []
            for source in sources:
                if isinstance(source, dict):
                    formatted_sources.append(SourceItem(**source))
                else:
                    formatted_sources.append(SourceItem(text=str(source)))

            return QueryResponse(
                answer=answer,
                sources=formatted_sources,
                session_id=session_id
            )
        except Exception as e:
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
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/")
    async def root():
        return {"message": "Test RAG System API"}

    return app


@pytest.fixture
def test_client(test_app):
    """Test client for FastAPI app"""
    return TestClient(test_app)


@pytest.fixture
def sample_query_request():
    """Sample query request data"""
    return {
        "query": "What is machine learning?",
        "session_id": None
    }


@pytest.fixture
def sample_query_request_with_session():
    """Sample query request with existing session"""
    return {
        "query": "Tell me more about supervised learning",
        "session_id": "existing-session-456"
    }
