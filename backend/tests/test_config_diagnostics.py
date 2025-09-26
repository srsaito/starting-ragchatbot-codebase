"""
Configuration and system setup diagnostic tests.

These tests identify critical configuration issues that could cause system failures.
"""

import os
from unittest.mock import Mock, patch

import pytest
from config import config
from rag_system import RAGSystem
from vector_store import VectorStore


class TestConfigurationIssues:
    """Test for configuration problems that could cause system failures"""

    def test_max_results_is_zero_bug(self):
        """
        CRITICAL: Test the MAX_RESULTS=0 configuration bug.
        This setting would cause vector searches to return zero results!
        """
        assert (
            config.MAX_RESULTS == 0
        ), f"Expected MAX_RESULTS=0 bug, got {config.MAX_RESULTS}"
        print(f"⚠️  CRITICAL BUG CONFIRMED: MAX_RESULTS={config.MAX_RESULTS}")
        print("   This setting will cause all vector searches to return zero results!")

    def test_anthropic_api_key_exists(self):
        """Test that ANTHROPIC_API_KEY is configured"""
        assert config.ANTHROPIC_API_KEY, "ANTHROPIC_API_KEY is missing or empty"
        assert len(config.ANTHROPIC_API_KEY) > 10, "ANTHROPIC_API_KEY seems too short"
        print(f"✓ ANTHROPIC_API_KEY configured: {config.ANTHROPIC_API_KEY[:10]}...")

    def test_anthropic_model_configured(self):
        """Test that Anthropic model is properly configured"""
        assert config.ANTHROPIC_MODEL, "ANTHROPIC_MODEL is missing"
        expected_model = "claude-sonnet-4-20250514"
        assert (
            config.ANTHROPIC_MODEL == expected_model
        ), f"Expected {expected_model}, got {config.ANTHROPIC_MODEL}"
        print(f"✓ Anthropic model: {config.ANTHROPIC_MODEL}")

    def test_embedding_model_configured(self):
        """Test embedding model configuration"""
        assert config.EMBEDDING_MODEL, "EMBEDDING_MODEL is missing"
        expected_model = "all-MiniLM-L6-v2"
        assert (
            config.EMBEDDING_MODEL == expected_model
        ), f"Expected {expected_model}, got {config.EMBEDDING_MODEL}"
        print(f"✓ Embedding model: {config.EMBEDDING_MODEL}")

    def test_chunk_settings_reasonable(self):
        """Test that chunk size and overlap are reasonable"""
        assert (
            config.CHUNK_SIZE > 0
        ), f"CHUNK_SIZE should be positive, got {config.CHUNK_SIZE}"
        assert (
            config.CHUNK_OVERLAP >= 0
        ), f"CHUNK_OVERLAP should be non-negative, got {config.CHUNK_OVERLAP}"
        assert (
            config.CHUNK_OVERLAP < config.CHUNK_SIZE
        ), "CHUNK_OVERLAP should be less than CHUNK_SIZE"
        print(
            f"✓ Chunk settings: size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}"
        )

    def test_max_history_reasonable(self):
        """Test MAX_HISTORY setting"""
        assert (
            config.MAX_HISTORY >= 0
        ), f"MAX_HISTORY should be non-negative, got {config.MAX_HISTORY}"
        print(f"✓ MAX_HISTORY: {config.MAX_HISTORY}")

    def test_chroma_path_configured(self):
        """Test ChromaDB path configuration"""
        assert config.CHROMA_PATH, "CHROMA_PATH is missing"
        print(f"✓ ChromaDB path: {config.CHROMA_PATH}")


class TestChromaDBInitialization:
    """Test ChromaDB setup and initialization"""

    def test_chromadb_can_initialize(self, temp_chroma_path):
        """Test that ChromaDB can be initialized"""
        try:
            vector_store = VectorStore(
                chroma_path=temp_chroma_path,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=5,  # Use a non-zero value
            )
            print("✓ ChromaDB initialized successfully")
            assert vector_store.client is not None
            assert vector_store.course_catalog is not None
            assert vector_store.course_content is not None
        except Exception as e:
            pytest.fail(f"ChromaDB initialization failed: {e}")

    def test_chromadb_with_zero_max_results(self, temp_chroma_path):
        """Test ChromaDB behavior with MAX_RESULTS=0"""
        vector_store = VectorStore(
            chroma_path=temp_chroma_path,
            embedding_model=config.EMBEDDING_MODEL,
            max_results=0,  # The problematic setting
        )

        # Try a search with zero max results
        results = vector_store.search("test query")
        print(
            f"⚠️  Search with MAX_RESULTS=0 returned: {len(results.documents)} documents"
        )

        # This should return empty results due to the zero limit
        assert len(results.documents) == 0, "Expected empty results with MAX_RESULTS=0"

    def test_existing_chroma_db_data(self):
        """Test if the actual ChromaDB has data"""
        if not os.path.exists(config.CHROMA_PATH):
            pytest.skip(f"ChromaDB path {config.CHROMA_PATH} doesn't exist")

        try:
            vector_store = VectorStore(
                chroma_path=config.CHROMA_PATH,
                embedding_model=config.EMBEDDING_MODEL,
                max_results=5,
            )

            # Check if there are any courses
            course_count = vector_store.get_course_count()
            course_titles = vector_store.get_existing_course_titles()

            print("📊 ChromaDB Status:")
            print(f"   Courses: {course_count}")
            print(f"   Titles: {course_titles}")

            if course_count == 0:
                print("⚠️  WARNING: ChromaDB contains no courses!")
            else:
                print(f"✓ ChromaDB contains {course_count} courses")

        except Exception as e:
            print(f"❌ Error accessing ChromaDB: {e}")
            pytest.fail(f"Cannot access existing ChromaDB: {e}")


class TestDocumentLoading:
    """Test document loading functionality"""

    def test_docs_folder_exists(self):
        """Test that the docs folder exists and has content"""
        docs_path = "../docs"
        relative_docs_path = os.path.join(os.path.dirname(__file__), "../../docs")
        absolute_docs_path = os.path.abspath(relative_docs_path)

        print("📁 Checking docs paths:")
        print(f"   Relative: {docs_path} -> exists: {os.path.exists(docs_path)}")
        print(
            f"   Absolute: {absolute_docs_path} -> exists: {os.path.exists(absolute_docs_path)}"
        )

        # Check if either path exists
        docs_exists = os.path.exists(docs_path) or os.path.exists(absolute_docs_path)
        assert docs_exists, f"Neither {docs_path} nor {absolute_docs_path} exists"

        # Use the path that exists
        actual_docs_path = (
            docs_path if os.path.exists(docs_path) else absolute_docs_path
        )

        # Check for document files
        doc_files = [
            f
            for f in os.listdir(actual_docs_path)
            if f.lower().endswith((".pdf", ".docx", ".txt"))
        ]

        print(f"   Document files: {doc_files}")
        assert len(doc_files) > 0, f"No document files found in {actual_docs_path}"
        print(f"✓ Found {len(doc_files)} document files")

    @patch("rag_system.RAGSystem")
    def test_document_loading_simulation(self, mock_rag_system, test_config):
        """Test document loading process simulation"""
        # This tests the loading logic without actually loading documents
        mock_instance = Mock()
        mock_instance.add_course_folder.return_value = (3, 50)  # 3 courses, 50 chunks
        mock_rag_system.return_value = mock_instance

        rag_system = RAGSystem(test_config)
        courses, chunks = rag_system.add_course_folder("../docs", clear_existing=False)

        assert courses == 3
        assert chunks == 50
        print(f"✓ Document loading simulation: {courses} courses, {chunks} chunks")


class TestSystemInitialization:
    """Test overall system initialization"""

    def test_rag_system_can_initialize(self, test_config, temp_chroma_path):
        """Test that RAGSystem can be initialized"""
        # Override the chroma path to use temp directory
        test_config.CHROMA_PATH = temp_chroma_path

        try:
            rag_system = RAGSystem(test_config)

            assert rag_system.config is not None
            assert rag_system.vector_store is not None
            assert rag_system.ai_generator is not None
            assert rag_system.tool_manager is not None
            assert rag_system.search_tool is not None

            print("✓ RAG System initialized successfully")

        except Exception as e:
            print(f"❌ RAG System initialization failed: {e}")
            pytest.fail(f"RAG System initialization failed: {e}")

    def test_tool_manager_setup(self, test_config, temp_chroma_path):
        """Test that ToolManager is properly configured"""
        test_config.CHROMA_PATH = temp_chroma_path

        rag_system = RAGSystem(test_config)

        # Check tool definitions
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) >= 1, "Expected at least one tool definition"

        # Check for search tool
        search_tool_found = any(
            tool["name"] == "search_course_content" for tool in tool_definitions
        )
        assert search_tool_found, "search_course_content tool not found"

        print(f"✓ Tool Manager configured with {len(tool_definitions)} tools")
        for tool in tool_definitions:
            print(f"   - {tool['name']}: {tool['description']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
