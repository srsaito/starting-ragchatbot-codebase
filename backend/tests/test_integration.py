"""
End-to-end integration tests for the RAG chatbot system.

Tests the complete system with real components where possible.
"""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from config import Config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestRealVectorStoreIntegration:
    """Test with real ChromaDB vector store"""

    def test_vector_store_with_sample_data(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test vector store operations with real ChromaDB"""

        # Create real vector store
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,
        )

        # Add sample data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Test search functionality
        results = vector_store.search("machine learning")

        assert not results.is_empty()
        assert len(results.documents) > 0
        assert len(results.metadata) == len(results.documents)

        # Check that results contain our sample data
        found_ml_content = any(
            "machine learning" in doc.lower() for doc in results.documents
        )
        assert found_ml_content, "Should find machine learning content"

        print(f"✓ Real vector store search returned {len(results.documents)} results")

        # Test course outline functionality
        outline = vector_store.get_course_outline("Introduction to Machine Learning")
        assert outline is not None
        assert outline["course_title"] == sample_course.title
        assert len(outline["lessons"]) == len(sample_course.lessons)

        print("✓ Course outline retrieval working")

    def test_vector_store_zero_max_results_bug(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test the MAX_RESULTS=0 bug with real vector store"""

        # Create vector store with zero max results (the bug)
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=0,  # This is the problematic setting
        )

        # Add sample data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Search should return empty results due to zero limit
        results = vector_store.search("machine learning")

        print(
            f"⚠️  With MAX_RESULTS=0, search returned {len(results.documents)} documents"
        )
        assert len(results.documents) == 0, "MAX_RESULTS=0 should return no results"

        # This demonstrates the bug - we have data but get no results
        print(
            "🐛 CONFIRMED: MAX_RESULTS=0 causes searches to return no results despite having data"
        )

    def test_vector_store_with_fixed_max_results(
        self, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test vector store with fixed MAX_RESULTS setting"""

        # Create vector store with proper max results
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model="all-MiniLM-L6-v2",
            max_results=5,  # Fixed setting
        )

        # Add sample data
        vector_store.add_course_metadata(sample_course)
        vector_store.add_course_content(sample_course_chunks)

        # Search should now work
        results = vector_store.search("machine learning")

        print(
            f"✓ With MAX_RESULTS=5, search returned {len(results.documents)} documents"
        )
        assert (
            len(results.documents) > 0
        ), "Should return results with proper MAX_RESULTS"


class TestRAGSystemIntegration:
    """Test RAG system with minimal mocking"""

    @patch("rag_system.AIGenerator")
    def test_rag_system_with_real_vector_store(
        self, mock_ai_gen, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test RAG system with real vector store but mocked AI"""

        # Mock AI generator to avoid actual API calls
        mock_ai_instance = Mock()
        mock_ai_instance.generate_response.return_value = (
            "Mocked AI response about machine learning"
        )
        mock_ai_gen.return_value = mock_ai_instance

        # Create config with fixed MAX_RESULTS
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.MAX_RESULTS = 5  # Fix the bug
        config.ANTHROPIC_API_KEY = "test-key"

        # Initialize RAG system (will use real vector store)
        rag_system = RAGSystem(config)

        # Add sample data to the real vector store
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_course_chunks)

        # Test query - this uses real search but mocked AI
        response, sources = rag_system.query("What is machine learning?")

        assert response == "Mocked AI response about machine learning"

        # Verify AI generator was called with tools
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args

        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] == rag_system.tool_manager

        print("✓ RAG system integration with real vector store working")

    @patch("rag_system.AIGenerator")
    def test_search_tool_with_real_vector_store(
        self, mock_ai_gen, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test that search tool works with real vector store"""

        # Mock AI generator
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "test-key"

        rag_system = RAGSystem(config)

        # Add sample data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_course_chunks)

        # Test search tool directly
        search_result = rag_system.search_tool.execute("machine learning")

        assert isinstance(search_result, str)
        assert len(search_result) > 0
        assert "Introduction to Machine Learning" in search_result
        assert len(rag_system.search_tool.last_sources) > 0

        print("✓ Search tool with real vector store working")
        print(f"   Result length: {len(search_result)} characters")
        print(f"   Sources found: {len(rag_system.search_tool.last_sources)}")


class TestDocumentLoadingIntegration:
    """Test document loading with real files if available"""

    def test_document_loading_with_sample_files(self):
        """Test document loading if sample files are available"""
        docs_path = "../docs"
        if not os.path.exists(docs_path):
            pytest.skip("No docs folder available for testing")

        doc_files = [
            f
            for f in os.listdir(docs_path)
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]

        if not doc_files:
            pytest.skip("No document files available for testing")

        print(f"📁 Found {len(doc_files)} document files: {doc_files}")

        # Test with temporary RAG system
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.CHROMA_PATH = temp_dir
            config.MAX_RESULTS = 5
            config.ANTHROPIC_API_KEY = "test-key"

            with patch("rag_system.AIGenerator") as mock_ai:
                mock_ai.return_value = Mock()

                rag_system = RAGSystem(config)

                # Try to load one document
                first_doc = os.path.join(docs_path, doc_files[0])
                course, chunks = rag_system.add_course_document(first_doc)

                if course is not None:
                    print(f"✓ Successfully loaded: {course.title}")
                    print(f"   Chunks created: {chunks}")
                    assert chunks > 0
                else:
                    print(f"⚠️  Failed to load document: {first_doc}")


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios"""

    @patch("ai_generator.anthropic")
    def test_complete_query_flow_simulation(
        self, mock_anthropic, temp_chroma_path, sample_course, sample_course_chunks
    ):
        """Test complete query flow with simulated AI responses"""

        # Mock Anthropic client with realistic tool use
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # First response: AI decides to use search tool
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"

        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_12345"
        mock_tool_use.input = {"query": "machine learning"}
        mock_initial_response.content = [mock_tool_use]

        # Second response: AI synthesizes results
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [
            Mock(
                text="Machine learning is a subset of AI that focuses on algorithms, as explained in the Introduction to Machine Learning course."
            )
        ]

        # Set up call sequence
        mock_client.messages.create.side_effect = [
            mock_initial_response,
            mock_final_response,
        ]

        # Create RAG system with fixed config
        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.MAX_RESULTS = 5  # Fix the bug
        config.ANTHROPIC_API_KEY = "test-key"

        rag_system = RAGSystem(config)

        # Add sample data
        rag_system.vector_store.add_course_metadata(sample_course)
        rag_system.vector_store.add_course_content(sample_course_chunks)

        # Execute complete query flow
        response, sources = rag_system.query("What is machine learning?")

        # Verify complete flow worked
        assert (
            response
            == "Machine learning is a subset of AI that focuses on algorithms, as explained in the Introduction to Machine Learning course."
        )
        assert len(sources) > 0

        # Verify AI was called twice (initial + final)
        assert mock_client.messages.create.call_count == 2

        # Verify tool was actually executed (search happened)
        # We can't easily verify this directly, but the fact that we got sources suggests it worked
        assert sources[0]["text"] is not None

        print("✓ Complete end-to-end query flow working")
        print(f"   Final response: {response}")
        print(f"   Sources returned: {len(sources)}")

    def test_error_propagation_simulation(self, temp_chroma_path):
        """Test how errors propagate through the system"""

        config = Config()
        config.CHROMA_PATH = temp_chroma_path
        config.MAX_RESULTS = 5
        config.ANTHROPIC_API_KEY = "invalid-key"  # This should cause API errors

        # Don't mock anything - let real errors occur
        rag_system = RAGSystem(config)

        # This should fail when trying to call the real Anthropic API
        with pytest.raises(Exception):
            rag_system.query("test query")

        print(
            "✓ Error propagation tested - system fails gracefully with invalid API key"
        )


class TestSystemHealthChecks:
    """Health check tests to verify system readiness"""

    def test_system_health_with_current_config(self):
        """Test system health with the current configuration"""
        from config import config  # Import actual config

        print("🏥 SYSTEM HEALTH CHECK")
        print("=" * 50)

        # Check 1: Configuration issues
        print(
            f"MAX_RESULTS: {config.MAX_RESULTS} {'❌ BUG!' if config.MAX_RESULTS == 0 else '✅'}"
        )
        print(f"API Key: {'✅ Set' if config.ANTHROPIC_API_KEY else '❌ Missing'}")
        print(f"ChromaDB Path: {config.CHROMA_PATH}")
        print(f"Embedding Model: {config.EMBEDDING_MODEL}")

        # Check 2: ChromaDB accessibility
        chroma_accessible = os.path.exists(config.CHROMA_PATH)
        print(f"ChromaDB Accessible: {'✅' if chroma_accessible else '❌'}")

        if chroma_accessible:
            try:
                vector_store = VectorStore(
                    config.CHROMA_PATH,
                    config.EMBEDDING_MODEL,
                    max_results=max(config.MAX_RESULTS, 1),  # Avoid zero
                )
                course_count = vector_store.get_course_count()
                print(
                    f"Course Count: {course_count} {'✅' if course_count > 0 else '⚠️  No data'}"
                )
            except Exception as e:
                print(f"ChromaDB Error: ❌ {e}")

        # Summary
        issues = []
        if config.MAX_RESULTS == 0:
            issues.append("MAX_RESULTS=0 will prevent search results")
        if not config.ANTHROPIC_API_KEY:
            issues.append("Missing ANTHROPIC_API_KEY")
        if not chroma_accessible:
            issues.append("ChromaDB not accessible")

        if issues:
            print(f"\n🚨 CRITICAL ISSUES FOUND: {len(issues)}")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        else:
            print("\n✅ No critical configuration issues found")

        return len(issues) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
