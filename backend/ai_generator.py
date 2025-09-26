from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search tools for course information.

Available Tools:
1. **Content Search Tool**: Use for questions about specific course content or detailed educational materials
2. **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews

Tool Usage Guidelines:
- Use the **course outline tool** for questions about:
  - Course outlines, structure, or lesson lists
  - "What lessons are in..." or "What's covered in..."
  - Course organization or curriculum overview
- Use the **content search tool** for questions about:
  - Specific topics within course materials
  - Detailed educational content or concepts
- **Up to 2 sequential tool calls allowed** for complex queries requiring:
  - Initial search followed by refined search based on results
  - Comparisons between different courses or lessons
  - Multi-part questions needing information from multiple sources
- Most queries only need a single tool call - use multiple rounds judiciously
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course outline questions**: Use course outline tool, then provide complete course title, course link, and numbered lesson list
- **Course content questions**: Use content search tool, then answer
- **Complex queries**: Use multiple tool calls if needed to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results" or "using the tool"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
        max_rounds: int = 2,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of sequential tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Initialize conversation with the user's query
        messages = [{"role": "user", "content": query}]
        rounds_completed = 0

        # Iterative tool calling loop - support up to max_rounds of tool usage
        while rounds_completed < max_rounds:
            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }

            # Add tools if available
            if tools and tool_manager:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            response = self.client.messages.create(**api_params)

            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and update conversation for next round
                messages = self._execute_tools_and_update_messages(
                    response, messages, tool_manager
                )
                rounds_completed += 1
            else:
                # Claude provided final response without using tools
                return response.content[0].text

        # Max rounds reached - get final response without tools
        return self._get_final_response(messages, system_content)

    def _execute_tools_and_update_messages(
        self, response, messages: List, tool_manager
    ) -> List:
        """
        Execute tools and update conversation messages for the next round.

        Args:
            response: The response containing tool use requests
            messages: Current conversation messages
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list with tool results
        """
        # Add AI's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors gracefully
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": f"Tool execution error: {str(e)}",
                        }
                    )

        # Add tool results to conversation
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages

    def _get_final_response(self, messages: List, system_content: str) -> str:
        """
        Get final response when max rounds reached or tool execution complete.

        Args:
            messages: Conversation messages including tool results
            system_content: System prompt content

        Returns:
            Final response text
        """
        # Make final API call without tools to get synthesized response
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
            # Note: No tools parameter - forcing Claude to provide final answer
        }

        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method for backward compatibility.
        Handle execution of tool calls and get follow-up response.

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Execute tools and update messages
        messages = self._execute_tools_and_update_messages(
            initial_response, messages, tool_manager
        )

        # Get final response without tools
        return self._get_final_response(messages, base_params["system"])
