"""
Unit tests for AIGenerator tool calling functionality.

Tests the AI generator's ability to correctly call CourseSearchTool and handle responses.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock

from ai_generator import AIGenerator


class TestAIGeneratorBasicFunctionality:
    """Test basic AIGenerator functionality"""
    
    @patch('ai_generator.anthropic')
    def test_ai_generator_initialization(self, mock_anthropic):
        """Test that AIGenerator can be initialized"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        assert ai_gen.client == mock_client
        assert ai_gen.model == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["model"] == "claude-sonnet-4-20250514"
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800
        
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test-api-key")
        print("✓ AIGenerator initialization working")
    
    @patch('ai_generator.anthropic')
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test generating response without tools"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock response
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="This is a direct response")]
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response("What is machine learning?")
        
        assert result == "This is a direct response"
        mock_client.messages.create.assert_called_once()
        
        # Check that the call was made without tools
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" not in call_args
        assert call_args["messages"][0]["content"] == "What is machine learning?"
        
        print("✓ Generate response without tools working")
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test generating response with conversation history"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Response with context")]
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        result = ai_gen.generate_response(
            "Follow up question", 
            conversation_history="Previous context..."
        )
        
        # Check that conversation history was included in system prompt
        call_args = mock_client.messages.create.call_args[1]
        assert "Previous context..." in call_args["system"]
        
        print("✓ Conversation history handling working")


class TestAIGeneratorSequentialToolCalling:
    """Test AI generator sequential tool calling functionality"""

    @patch('ai_generator.anthropic')
    def test_sequential_tool_calls_two_rounds(self, mock_anthropic):
        """Test that AI can make sequential tool calls across two rounds"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # First response - tool use for getting course outline
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "get_course_outline"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"course_title": "Introduction to AI"}
        mock_first_response.content = [mock_tool_use_1]

        # Second response - tool use for searching content
        mock_second_response = Mock()
        mock_second_response.stop_reason = "tool_use"
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "search_course_content"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"query": "APIs and web services"}
        mock_second_response.content = [mock_tool_use_2]

        # Final response - synthesis without tools
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on the course outline and content search...")]

        # Set up call sequence
        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response,
            mock_final_response
        ]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course: Introduction to AI\nLessons:\n1. Basics\n2. APIs and Web Services",
            "APIs are interfaces for applications to communicate..."
        ]

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        tool_definitions = [
            {"name": "get_course_outline"},
            {"name": "search_course_content"}
        ]

        result = ai_gen.generate_response(
            "What does lesson 2 of Introduction to AI course cover about APIs?",
            tools=tool_definitions,
            tool_manager=mock_tool_manager
        )

        assert result == "Based on the course outline and content search..."

        # Verify three API calls were made (2 with tools, 1 without)
        assert mock_client.messages.create.call_count == 3

        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline",
            course_title="Introduction to AI"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content",
            query="APIs and web services"
        )

        # Verify final call was made without tools
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args

        print("✓ Sequential tool calls (2 rounds) working correctly")

    @patch('ai_generator.anthropic')
    def test_early_termination_after_one_round(self, mock_anthropic):
        """Test that AI can terminate early after one round if sufficient information gathered"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # First response - tool use
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "machine learning fundamentals"}
        mock_first_response.content = [mock_tool_use]

        # Second response - direct answer without further tool use
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end_turn"
        mock_second_response.content = [Mock(text="Machine learning fundamentals include...")]

        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "ML is a subset of AI that enables..."

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        result = ai_gen.generate_response(
            "What are machine learning fundamentals?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "Machine learning fundamentals include..."

        # Verify only 2 API calls (1 with tools, 1 with tools available but not used)
        assert mock_client.messages.create.call_count == 2

        # Verify tool was executed once
        assert mock_tool_manager.execute_tool.call_count == 1

        print("✓ Early termination after one round working correctly")

    @patch('ai_generator.anthropic')
    def test_max_rounds_termination(self, mock_anthropic):
        """Test that tool calling stops after max rounds (2) even if Claude wants to continue"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Both responses use tools
        mock_tool_response = Mock()
        mock_tool_response.stop_reason = "tool_use"
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_id"
        mock_tool_use.input = {"query": "test"}
        mock_tool_response.content = [mock_tool_use]

        # Final response after max rounds
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Final synthesis after 2 rounds")]

        mock_client.messages.create.side_effect = [
            mock_tool_response,  # Round 1
            mock_tool_response,  # Round 2
            mock_final_response  # Final without tools
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        result = ai_gen.generate_response(
            "Complex query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=2
        )

        assert result == "Final synthesis after 2 rounds"

        # Verify 3 API calls (2 with tools, 1 without)
        assert mock_client.messages.create.call_count == 3

        # Verify tools executed twice
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify final call has no tools
        final_call_args = mock_client.messages.create.call_args_list[2][1]
        assert "tools" not in final_call_args

        print("✓ Max rounds termination working correctly")

    @patch('ai_generator.anthropic')
    def test_tool_execution_error_handling_continues(self, mock_anthropic):
        """Test that tool execution errors are handled gracefully and included in context"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # First response - tool use that will fail
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "test"}
        mock_first_response.content = [mock_tool_use]

        # Second response - handles error and provides answer
        mock_second_response = Mock()
        mock_second_response.stop_reason = "end_turn"
        mock_second_response.content = [Mock(text="I encountered an error but here's what I know...")]

        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_second_response
        ]

        # Mock tool manager that raises exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Database connection failed")

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        # Should not raise exception - error handled gracefully
        result = ai_gen.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        assert result == "I encountered an error but here's what I know..."

        # Verify error was passed to Claude in tool results
        second_call_messages = mock_client.messages.create.call_args_list[1][1]["messages"]
        tool_results = second_call_messages[-1]["content"]
        assert "Tool execution error" in tool_results[0]["content"]
        assert "Database connection failed" in tool_results[0]["content"]

        print("✓ Tool execution error handling working correctly")


class TestAIGeneratorToolCalling:
    """Test AI generator basic tool calling functionality (backward compatibility)"""
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_tools_no_use(self, mock_anthropic):
        """Test response generation when tools are available but not used"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock response without tool use
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Direct answer without using tools")]
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        mock_tool_manager = Mock()
        
        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        result = ai_gen.generate_response(
            "What is 2+2?",
            tools=tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Direct answer without using tools"
        
        # Check that tools were provided in the API call
        call_args = mock_client.messages.create.call_args[1]
        assert "tools" in call_args
        assert call_args["tools"] == tool_definitions
        assert call_args["tool_choice"] == {"type": "auto"}
        
        # Tool manager should not have been called
        mock_tool_manager.execute_tool.assert_not_called()
        
        print("✓ Tools available but not used - working")
    
    @patch('ai_generator.anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic):
        """Test response generation when AI decides to use tools"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        # Mock tool use content block
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_12345"
        mock_tool_use.input = {"query": "machine learning basics"}
        mock_initial_response.content = [mock_tool_use]
        
        # Mock final response after tool execution
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on the search results, machine learning is...")]
        
        # Set up call sequence: first call returns tool use, second returns final answer
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results: ML is a subset of AI..."
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        tool_definitions = [
            {
                "name": "search_course_content",
                "description": "Search course materials",
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        ]
        
        result = ai_gen.generate_response(
            "What is machine learning?",
            tools=tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on the search results, machine learning is..."
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="machine learning basics"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        print("✓ Tool use flow working correctly")
    
    @patch('ai_generator.anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic):
        """Test handling when tool execution fails"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock initial response with tool use
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_12345"
        mock_tool_use.input = {"query": "test query"}
        mock_initial_response.content = [mock_tool_use]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="I apologize, but I encountered an error...")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool manager that throws an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        # This should not raise an exception - it should be handled gracefully
        result = ai_gen.generate_response(
            "test query",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Should get a final response despite the error
        assert result == "I apologize, but I encountered an error..."

        # Verify the error message was passed to Claude
        second_call = mock_client.messages.create.call_args_list[1][1]
        messages = second_call["messages"]
        # Find the tool result message
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert "Tool execution error" in tool_result_msg["content"][0]["content"]

        print("✓ Tool execution error handling tested")
    
    @patch('ai_generator.anthropic')
    def test_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        # Mock initial response with multiple tool uses
        mock_initial_response = Mock()
        mock_initial_response.stop_reason = "tool_use"
        
        mock_tool_use_1 = Mock()
        mock_tool_use_1.type = "tool_use"
        mock_tool_use_1.name = "search_course_content"
        mock_tool_use_1.id = "tool_1"
        mock_tool_use_1.input = {"query": "neural networks"}
        
        mock_tool_use_2 = Mock()
        mock_tool_use_2.type = "tool_use"
        mock_tool_use_2.name = "get_course_outline"
        mock_tool_use_2.id = "tool_2"
        mock_tool_use_2.input = {"course_title": "Deep Learning"}
        
        mock_initial_response.content = [mock_tool_use_1, mock_tool_use_2]
        
        # Mock final response
        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Based on both searches...")]
        
        mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Search result 1...",
            "Course outline result..."
        ]
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        
        result = ai_gen.generate_response(
            "Tell me about neural networks and deep learning courses",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )
        
        assert result == "Based on both searches..."
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="neural networks")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Deep Learning")
        
        print("✓ Multiple tool calls handled correctly")


class TestAIGeneratorSystemPrompt:
    """Test system prompt and instruction handling"""
    
    @patch('ai_generator.anthropic')
    def test_system_prompt_content(self, mock_anthropic):
        """Test that system prompt contains expected instructions"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        ai_gen.generate_response("Test query")
        
        # Check system prompt content
        call_args = mock_client.messages.create.call_args[1]
        system_content = call_args["system"]
        
        # Should contain key instructions
        assert "course materials" in system_content.lower()
        assert "search tool" in system_content.lower()
        assert "content search tool" in system_content.lower()
        assert "course outline tool" in system_content.lower()
        
        print("✓ System prompt contains expected instructions")
    
    @patch('ai_generator.anthropic')
    def test_api_parameters(self, mock_anthropic):
        """Test that API parameters are correctly set"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client
        
        mock_response = Mock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [Mock(text="Test response")]
        mock_client.messages.create.return_value = mock_response
        
        ai_gen = AIGenerator("test-api-key", "claude-sonnet-4-20250514")
        ai_gen.generate_response("Test query")
        
        call_args = mock_client.messages.create.call_args[1]
        
        # Check API parameters
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert len(call_args["messages"]) == 1
        assert call_args["messages"][0]["role"] == "user"
        assert call_args["messages"][0]["content"] == "Test query"
        
        print("✓ API parameters correctly set")


class TestAIGeneratorConversationFlow:
    """Test conversation flow and message building in sequential tool calls"""

    @patch('ai_generator.anthropic')
    def test_conversation_messages_accumulate_correctly(self, mock_anthropic):
        """Test that conversation messages accumulate properly through rounds"""
        import copy

        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Track all API calls to verify message accumulation
        api_calls = []

        def track_calls(**kwargs):
            # Deep copy to preserve state at this moment
            api_calls.append(copy.deepcopy(kwargs))
            if len(api_calls) == 1:
                # First call - return tool use
                response = Mock()
                response.stop_reason = "tool_use"
                tool_use = Mock()
                tool_use.type = "tool_use"
                tool_use.name = "get_course_outline"
                tool_use.id = "tool_1"
                tool_use.input = {"course_title": "test"}
                response.content = [tool_use]
                return response
            elif len(api_calls) == 2:
                # Second call - return another tool use
                response = Mock()
                response.stop_reason = "tool_use"
                tool_use = Mock()
                tool_use.type = "tool_use"
                tool_use.name = "search_course_content"
                tool_use.id = "tool_2"
                tool_use.input = {"query": "APIs"}
                response.content = [tool_use]
                return response
            else:
                # Final call
                response = Mock()
                response.stop_reason = "end_turn"
                response.content = [Mock(text="Final answer")]
                return response

        mock_client.messages.create = Mock(side_effect=track_calls)

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course outline results",
            "Content search results"
        ]

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        result = ai_gen.generate_response(
            "Initial query",
            tools=[{"name": "get_course_outline"}, {"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )

        # Verify we made 3 API calls total
        assert len(api_calls) == 3

        # Check first call - should only have user query
        assert len(api_calls[0]["messages"]) == 1
        assert api_calls[0]["messages"][0]["role"] == "user"
        assert api_calls[0]["messages"][0]["content"] == "Initial query"
        assert "tools" in api_calls[0]  # Should have tools available

        # Check second call - should have accumulated messages from round 1
        assert len(api_calls[1]["messages"]) == 3
        assert api_calls[1]["messages"][0]["content"] == "Initial query"  # Original user query
        assert api_calls[1]["messages"][1]["role"] == "assistant"  # Assistant's tool use
        assert api_calls[1]["messages"][2]["role"] == "user"  # Tool results
        assert api_calls[1]["messages"][2]["content"][0]["content"] == "Course outline results"
        assert "tools" in api_calls[1]  # Should still have tools available

        # Check third call - should have all accumulated messages
        assert len(api_calls[2]["messages"]) == 5
        assert api_calls[2]["messages"][0]["content"] == "Initial query"  # Original query
        assert api_calls[2]["messages"][3]["role"] == "assistant"  # Second tool use
        assert api_calls[2]["messages"][4]["role"] == "user"  # Second tool results
        assert api_calls[2]["messages"][4]["content"][0]["content"] == "Content search results"
        assert "tools" not in api_calls[2]  # Final call should NOT have tools

        print("✓ Conversation messages accumulate correctly through rounds")

    @patch('ai_generator.anthropic')
    def test_backward_compatibility_single_round(self, mock_anthropic):
        """Test that single-round behavior is preserved for backward compatibility"""
        mock_client = Mock()
        mock_anthropic.Anthropic.return_value = mock_client

        # Single tool use followed by final response
        mock_first_response = Mock()
        mock_first_response.stop_reason = "tool_use"
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "search_course_content"
        mock_tool_use.id = "tool_1"
        mock_tool_use.input = {"query": "neural networks"}
        mock_first_response.content = [mock_tool_use]

        mock_final_response = Mock()
        mock_final_response.stop_reason = "end_turn"
        mock_final_response.content = [Mock(text="Neural networks are...")]

        mock_client.messages.create.side_effect = [
            mock_first_response,
            mock_final_response
        ]

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results about neural networks"

        ai_gen = AIGenerator("test-api-key", "claude-sonnet")

        # Call with max_rounds=1 to ensure single-round behavior
        result = ai_gen.generate_response(
            "What are neural networks?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager,
            max_rounds=1
        )

        assert result == "Neural networks are..."
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

        # Verify final call has no tools
        final_call = mock_client.messages.create.call_args_list[1][1]
        assert "tools" not in final_call

        print("✓ Backward compatibility for single-round tool calling preserved")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])