"""
Unit tests for CourseSearchTool.execute() method.

Tests the core search functionality in isolation with various scenarios.
"""

import pytest
from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults

from .conftest import assert_search_results_format, assert_sources_format


class TestCourseSearchToolExecute:
    """Test CourseSearchTool.execute() method with various scenarios"""

    def test_execute_successful_search(self, course_search_tool):
        """Test execute with successful search results"""
        result = course_search_tool.execute("machine learning")

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Introduction to Machine Learning" in result
        assert "Sample content about machine learning" in result

        # Check result formatting
        assert_search_results_format(result)

        # Check that sources were tracked
        assert len(course_search_tool.last_sources) > 0
        assert_sources_format(course_search_tool.last_sources)

        print(f"✓ Successful search result: {len(result)} chars")
        print(f"✓ Sources tracked: {len(course_search_tool.last_sources)}")

    def test_execute_with_course_filter(self, mock_vector_store):
        """Test execute with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("algorithms", course_name="Machine Learning")

        # Verify the vector store was called with the right parameters
        mock_vector_store.search.assert_called_once_with(
            query="algorithms", course_name="Machine Learning", lesson_number=None
        )

        assert isinstance(result, str)
        print("✓ Course filter search executed successfully")

    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test execute with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("neural networks", lesson_number=3)

        # Verify the vector store was called with the right parameters
        mock_vector_store.search.assert_called_once_with(
            query="neural networks", course_name=None, lesson_number=3
        )

        assert isinstance(result, str)
        print("✓ Lesson filter search executed successfully")

    def test_execute_with_both_filters(self, mock_vector_store):
        """Test execute with both course and lesson filters"""
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("deep learning", course_name="AI Basics", lesson_number=2)

        # Verify the vector store was called with all parameters
        mock_vector_store.search.assert_called_once_with(
            query="deep learning", course_name="AI Basics", lesson_number=2
        )

        assert isinstance(result, str)
        print("✓ Combined filters search executed successfully")

    def test_execute_empty_results(self, mock_vector_store_empty):
        """Test execute when search returns no results"""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute("nonexistent topic")

        assert isinstance(result, str)
        assert "No relevant content found" in result
        assert len(tool.last_sources) == 0  # No sources for empty results

        print("✓ Empty results handled correctly")

    def test_execute_empty_results_with_filters(self, mock_vector_store_empty):
        """Test execute with filters when no results found"""
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute("topic", course_name="Test Course", lesson_number=1)

        assert "No relevant content found" in result
        assert "in course 'Test Course'" in result
        assert "in lesson 1" in result

        print("✓ Empty results with filters handled correctly")

    def test_execute_search_error(self, mock_vector_store_error):
        """Test execute when search returns an error"""
        tool = CourseSearchTool(mock_vector_store_error)
        result = tool.execute("any query")

        assert isinstance(result, str)
        assert "Search error occurred" in result
        assert len(tool.last_sources) == 0  # No sources for error results

        print("✓ Search error handled correctly")

    def test_execute_vector_store_exception(self, mock_vector_store):
        """Test execute when vector store throws an exception"""
        # Make the search method throw an exception
        mock_vector_store.search.side_effect = Exception("Database connection failed")

        tool = CourseSearchTool(mock_vector_store)

        # This should be handled gracefully by the SearchResults error handling
        with pytest.raises(Exception):
            tool.execute("query")

        print("✓ Vector store exception handling tested")


class TestCourseSearchToolResultFormatting:
    """Test result formatting and source tracking"""

    def test_format_results_basic(self, mock_vector_store):
        """Test basic result formatting"""
        # Set up specific search results
        mock_vector_store.search.return_value = SearchResults(
            documents=[
                "Content about supervised learning",
                "Content about neural networks",
            ],
            metadata=[
                {"course_title": "ML Course", "lesson_number": 1},
                {"course_title": "ML Course", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("supervised learning")

        # Check formatting
        assert "[ML Course - Lesson 1]" in result
        assert "[ML Course - Lesson 2]" in result
        assert "Content about supervised learning" in result
        assert "Content about neural networks" in result

        print("✓ Result formatting correct")

    def test_format_results_no_lesson_number(self, mock_vector_store):
        """Test formatting when lesson number is missing"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["General course content"],
            metadata=[{"course_title": "General Course", "lesson_number": None}],
            distances=[0.1],
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("general topic")

        # Should format without lesson number
        assert "[General Course]" in result
        assert "Lesson" not in result or "- Lesson None" not in result

        print("✓ Formatting without lesson number correct")

    def test_source_tracking_with_links(self, mock_vector_store):
        """Test that sources are tracked with lesson links"""
        # Set up mock to return lesson links
        mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson1"

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Check that sources have links
        assert len(tool.last_sources) > 0
        for source in tool.last_sources:
            assert "text" in source
            # Should have attempted to get lesson link
            mock_vector_store.get_lesson_link.assert_called()

        print("✓ Source tracking with links working")

    def test_source_tracking_link_failure(self, mock_vector_store):
        """Test source tracking when lesson link retrieval fails"""
        # Make get_lesson_link throw an exception
        mock_vector_store.get_lesson_link.side_effect = Exception(
            "Link retrieval failed"
        )

        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute("test query")

        # Should still work, just without links
        assert len(tool.last_sources) > 0
        for source in tool.last_sources:
            assert "text" in source
            assert source["link"] is None  # Should be None when link retrieval fails

        print("✓ Source tracking resilient to link failures")


class TestCourseSearchToolDefinition:
    """Test tool definition for Anthropic API compatibility"""

    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()

        assert isinstance(definition, dict)
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        # Check input schema structure
        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert schema["required"] == ["query"]

        # Check optional parameters
        assert "course_name" in schema["properties"]
        assert "lesson_number" in schema["properties"]

        print("✓ Tool definition correctly formatted")
        print(f"   Name: {definition['name']}")
        print(f"   Required: {schema['required']}")
        print(
            f"   Optional: {[k for k in schema['properties'].keys() if k not in schema['required']]}"
        )


class TestToolManager:
    """Test ToolManager functionality with CourseSearchTool"""

    def test_tool_manager_registration(self, mock_vector_store):
        """Test registering CourseSearchTool with ToolManager"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        tool_manager.register_tool(search_tool)

        # Check tool is registered
        assert "search_course_content" in tool_manager.tools

        # Check tool definitions
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

        print("✓ Tool registration working")

    def test_tool_manager_execution(self, mock_vector_store):
        """Test executing tool through ToolManager"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute tool through manager
        result = tool_manager.execute_tool("search_course_content", query="test query")

        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once()

        print("✓ Tool execution through manager working")

    def test_tool_manager_source_tracking(self, mock_vector_store):
        """Test that ToolManager can retrieve sources from tools"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute tool to generate sources
        tool_manager.execute_tool("search_course_content", query="test query")

        # Get sources
        sources = tool_manager.get_last_sources()
        assert isinstance(sources, list)
        assert len(sources) > 0

        # Reset sources
        tool_manager.reset_sources()
        sources_after_reset = tool_manager.get_last_sources()
        assert len(sources_after_reset) == 0

        print("✓ Source tracking and reset working")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
