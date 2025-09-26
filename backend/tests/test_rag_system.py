"""
Tests for RAG system query handling.

Tests the overall orchestration and integration of components in the RAG system.
"""

from unittest.mock import Mock, patch

import pytest
from rag_system import RAGSystem


class TestRAGSystemInitialization:
    """Test RAG system initialization and setup"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_rag_system_initialization(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_ai_gen,
        mock_vector_store,
        test_config,
    ):
        """Test that RAG system initializes all components correctly"""

        # Create mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_ai_gen_instance = Mock()
        mock_ai_gen.return_value = mock_ai_gen_instance

        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance

        mock_session_mgr_instance = Mock()
        mock_session_mgr.return_value = mock_session_mgr_instance

        # Initialize RAG system
        rag_system = RAGSystem(test_config)

        # Verify all components were initialized with correct parameters
        mock_vector_store.assert_called_once_with(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS,
        )

        mock_ai_gen.assert_called_once_with(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )

        mock_doc_proc.assert_called_once_with(
            test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP
        )

        mock_session_mgr.assert_called_once_with(test_config.MAX_HISTORY)

        # Verify components are assigned
        assert rag_system.vector_store == mock_vector_store_instance
        assert rag_system.ai_generator == mock_ai_gen_instance
        assert rag_system.document_processor == mock_doc_proc_instance
        assert rag_system.session_manager == mock_session_mgr_instance

        # Verify tool manager setup
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        print("✓ RAG system initialization working correctly")

    def test_tool_manager_setup(self, test_config, temp_chroma_path):
        """Test that tool manager is set up with correct tools"""
        test_config.CHROMA_PATH = temp_chroma_path

        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor") as mock_dp,
            patch("rag_system.SessionManager") as mock_sm,
        ):

            # Create mock instances
            mock_vs.return_value = Mock()
            mock_ai.return_value = Mock()
            mock_dp.return_value = Mock()
            mock_sm.return_value = Mock()

            rag_system = RAGSystem(test_config)

            # Check tool definitions
            tool_definitions = rag_system.tool_manager.get_tool_definitions()
            assert (
                len(tool_definitions) >= 2
            )  # Should have at least search and outline tools

            tool_names = [tool["name"] for tool in tool_definitions]
            assert "search_course_content" in tool_names
            assert "get_course_outline" in tool_names

            print(f"✓ Tool manager configured with {len(tool_definitions)} tools:")
            for tool in tool_definitions:
                print(f"   - {tool['name']}")


class TestRAGSystemQuery:
    """Test RAG system query processing"""

    def setup_mocked_rag_system(self, test_config):
        """Helper to create RAG system with mocked components"""
        with (
            patch("rag_system.VectorStore") as mock_vs,
            patch("rag_system.AIGenerator") as mock_ai,
            patch("rag_system.DocumentProcessor") as mock_dp,
            patch("rag_system.SessionManager") as mock_sm,
        ):

            # Create mock instances
            mock_vector_store = Mock()
            mock_vs.return_value = mock_vector_store

            mock_ai_generator = Mock()
            mock_ai.return_value = mock_ai_generator

            mock_doc_processor = Mock()
            mock_dp.return_value = mock_doc_processor

            mock_session_manager = Mock()
            mock_sm.return_value = mock_session_manager

            rag_system = RAGSystem(test_config)

            return rag_system, {
                "vector_store": mock_vector_store,
                "ai_generator": mock_ai_generator,
                "doc_processor": mock_doc_processor,
                "session_manager": mock_session_manager,
            }

    def test_query_without_session(self, test_config):
        """Test query processing without session ID"""
        rag_system, mocks = self.setup_mocked_rag_system(test_config)

        # Mock AI generator response
        mocks["ai_generator"].generate_response.return_value = (
            "AI response about machine learning"
        )

        # Mock tool manager (this is real, not mocked)
        with (
            patch.object(rag_system.tool_manager, "get_last_sources") as mock_sources,
            patch.object(rag_system.tool_manager, "reset_sources") as mock_reset,
        ):

            mock_sources.return_value = [
                {"text": "ML Course", "link": "https://example.com"}
            ]

            response, sources = rag_system.query("What is machine learning?")

            # Verify response
            assert response == "AI response about machine learning"
            assert len(sources) == 1
            assert sources[0]["text"] == "ML Course"

            # Verify AI generator was called correctly
            mocks["ai_generator"].generate_response.assert_called_once()
            call_args = mocks["ai_generator"].generate_response.call_args

            assert (
                "Answer this question about course materials: What is machine learning?"
                in call_args[1]["query"]
            )
            assert call_args[1]["conversation_history"] is None
            assert call_args[1]["tools"] is not None
            assert call_args[1]["tool_manager"] == rag_system.tool_manager

            # Verify sources were retrieved and reset
            mock_sources.assert_called_once()
            mock_reset.assert_called_once()

            print("✓ Query without session working correctly")

    def test_query_with_session(self, test_config):
        """Test query processing with session ID"""
        rag_system, mocks = self.setup_mocked_rag_system(test_config)

        # Mock session manager
        mocks["session_manager"].get_conversation_history.return_value = (
            "Previous: Hello"
        )

        # Mock AI generator response
        mocks["ai_generator"].generate_response.return_value = "Contextual AI response"

        with (
            patch.object(rag_system.tool_manager, "get_last_sources") as mock_sources,
            patch.object(rag_system.tool_manager, "reset_sources") as mock_reset,
        ):

            mock_sources.return_value = []

            response, sources = rag_system.query(
                "Follow-up question", session_id="session123"
            )

            # Verify session manager was used
            mocks["session_manager"].get_conversation_history.assert_called_once_with(
                "session123"
            )
            mocks["session_manager"].add_exchange.assert_called_once_with(
                "session123", "Follow-up question", "Contextual AI response"
            )

            # Verify AI generator got conversation history
            call_args = mocks["ai_generator"].generate_response.call_args
            assert call_args[1]["conversation_history"] == "Previous: Hello"

            print("✓ Query with session working correctly")

    def test_query_exception_handling(self, test_config):
        """Test that query exceptions are properly handled"""
        rag_system, mocks = self.setup_mocked_rag_system(test_config)

        # Make AI generator throw an exception
        mocks["ai_generator"].generate_response.side_effect = Exception(
            "AI service failed"
        )

        # This should raise the exception (not caught by RAG system)
        with pytest.raises(Exception, match="AI service failed"):
            rag_system.query("test query")

        print("✓ Query exception handling tested")

    def test_query_prompt_construction(self, test_config):
        """Test that query prompts are constructed correctly"""
        rag_system, mocks = self.setup_mocked_rag_system(test_config)

        mocks["ai_generator"].generate_response.return_value = "Test response"

        with (
            patch.object(rag_system.tool_manager, "get_last_sources") as mock_sources,
            patch.object(rag_system.tool_manager, "reset_sources"),
        ):

            mock_sources.return_value = []

            rag_system.query("What is deep learning?")

            # Check that the prompt was constructed correctly
            call_args = mocks["ai_generator"].generate_response.call_args
            query_param = call_args[1]["query"]

            assert "Answer this question about course materials:" in query_param
            assert "What is deep learning?" in query_param

            print("✓ Query prompt construction working")


class TestRAGSystemDocumentManagement:
    """Test document loading and management functionality"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document(
        self,
        mock_sm,
        mock_dp,
        mock_ai,
        mock_vs,
        test_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test adding a single course document"""

        # Set up mocks
        mock_vector_store = Mock()
        mock_vs.return_value = mock_vector_store

        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )
        mock_dp.return_value = mock_doc_processor

        mock_ai.return_value = Mock()
        mock_sm.return_value = Mock()

        rag_system = RAGSystem(test_config)

        # Add course document
        course, chunk_count = rag_system.add_course_document("/fake/path/course.pdf")

        # Verify document processing
        mock_doc_processor.process_course_document.assert_called_once_with(
            "/fake/path/course.pdf"
        )

        # Verify vector store operations
        mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

        # Verify return values
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        print("✓ Add course document working correctly")

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    @patch("os.path.exists")
    @patch("os.listdir")
    def test_add_course_folder(
        self, mock_listdir, mock_exists, mock_sm, mock_dp, mock_ai, mock_vs, test_config
    ):
        """Test adding courses from a folder"""

        # Set up mocks
        mock_exists.return_value = True
        mock_listdir.return_value = [
            "course1.pdf",
            "course2.txt",
            "course3.docx",
            "readme.md",
        ]

        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles.return_value = (
            []
        )  # No existing courses
        mock_vs.return_value = mock_vector_store

        mock_doc_processor = Mock()
        # Mock successful processing for each course
        mock_doc_processor.process_course_document.side_effect = [
            (Mock(title="Course 1"), [Mock(), Mock()]),  # 2 chunks
            (Mock(title="Course 2"), [Mock(), Mock(), Mock()]),  # 3 chunks
            (Mock(title="Course 3"), [Mock()]),  # 1 chunk
        ]
        mock_dp.return_value = mock_doc_processor

        mock_ai.return_value = Mock()
        mock_sm.return_value = Mock()

        rag_system = RAGSystem(test_config)

        # Add course folder
        total_courses, total_chunks = rag_system.add_course_folder(
            "/fake/docs", clear_existing=False
        )

        # Verify results
        assert total_courses == 3
        assert total_chunks == 6  # 2 + 3 + 1

        # Verify document processor was called for each PDF/DOCX/TXT file
        assert mock_doc_processor.process_course_document.call_count == 3

        print("✓ Add course folder working correctly")

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_document_processing_error_handling(
        self, mock_sm, mock_dp, mock_ai, mock_vs, test_config
    ):
        """Test error handling during document processing"""

        # Set up mocks
        mock_vector_store = Mock()
        mock_vs.return_value = mock_vector_store

        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document.side_effect = Exception(
            "Processing failed"
        )
        mock_dp.return_value = mock_doc_processor

        mock_ai.return_value = Mock()
        mock_sm.return_value = Mock()

        rag_system = RAGSystem(test_config)

        # Should handle the exception gracefully
        course, chunk_count = rag_system.add_course_document("/fake/path/bad_file.pdf")

        assert course is None
        assert chunk_count == 0

        print("✓ Document processing error handling working")


class TestRAGSystemAnalytics:
    """Test analytics and course information functionality"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics(
        self, mock_sm, mock_dp, mock_ai, mock_vs, test_config
    ):
        """Test getting course analytics"""

        # Set up mocks
        mock_vector_store = Mock()
        mock_vector_store.get_course_count.return_value = 5
        mock_vector_store.get_existing_course_titles.return_value = [
            "Course 1",
            "Course 2",
            "Course 3",
            "Course 4",
            "Course 5",
        ]
        mock_vs.return_value = mock_vector_store

        mock_ai.return_value = Mock()
        mock_dp.return_value = Mock()
        mock_sm.return_value = Mock()

        rag_system = RAGSystem(test_config)

        # Get analytics
        analytics = rag_system.get_course_analytics()

        # Verify results
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

        # Verify vector store methods were called
        mock_vector_store.get_course_count.assert_called_once()
        mock_vector_store.get_existing_course_titles.assert_called_once()

        print("✓ Course analytics working correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
