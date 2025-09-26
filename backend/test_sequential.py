#!/usr/bin/env python
"""
Test script to verify sequential tool calling functionality.
This simulates complex queries that require multiple tool calls.
"""

import os

from ai_generator import AIGenerator
from dotenv import load_dotenv
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore

# Load environment variables
load_dotenv()


def test_sequential_tool_calling():
    """Test that the AI can make sequential tool calls for complex queries"""

    # Initialize components
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY in .env file")
        return

    # Initialize the AI generator
    ai_gen = AIGenerator(
        api_key, "claude-3-haiku-20240307"
    )  # Using Haiku for faster testing

    # Initialize vector store (assuming it's already populated)
    vector_store = VectorStore()

    # Initialize tools
    tool_manager = ToolManager()
    search_tool = CourseSearchTool(vector_store)
    outline_tool = CourseOutlineTool(vector_store)

    tool_manager.register_tool(search_tool)
    tool_manager.register_tool(outline_tool)

    # Test queries that should trigger sequential tool calls
    test_queries = [
        # Query 1: Should use outline tool then search tool
        "What specific topics are covered in lesson 2 of the Introduction to AI course?",
        # Query 2: Should use multiple searches to compare
        "Compare the content about APIs between different courses",
        # Query 3: Should use outline to find lessons then search for specific content
        "Find all lessons that discuss neural networks and tell me what each course says about them",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"Test Query {i}: {query}")
        print("=" * 60)

        try:
            # Call the AI with tools available
            response = ai_gen.generate_response(
                query=query,
                tools=tool_manager.get_tool_definitions(),
                tool_manager=tool_manager,
                max_rounds=2,  # Allow up to 2 rounds of tool calling
            )

            print(f"\nResponse:\n{response}")

            # Check if sources were found
            sources = tool_manager.get_last_sources()
            if sources:
                print(f"\nSources used: {len(sources)}")
                for source in sources:
                    print(f"  - {source.get('text', 'Unknown')}")

            # Reset sources for next query
            tool_manager.reset_sources()

        except Exception as e:
            print(f"Error processing query: {str(e)}")


if __name__ == "__main__":
    print("Testing Sequential Tool Calling Implementation")
    print("=" * 60)
    test_sequential_tool_calling()
    print("\n" + "=" * 60)
    print("Test completed!")
