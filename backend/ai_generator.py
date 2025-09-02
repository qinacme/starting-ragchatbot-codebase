from typing import Any, Dict, List, Optional

import anthropic


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive tools for course information.

Tool Usage Guidelines:
- **search_course_content**: Use for questions about specific course content, concepts, or detailed educational materials
- **get_course_outline**: Use for questions about course structure, lesson lists, course overviews, or outlines
- **Up to 2 sequential tool calling rounds per query**
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

When to Use Each Tool:
- Course outline queries: "What's the structure of [course]?", "Show me the outline", "What lessons are in [course]?"
- Content search queries: "Explain [concept] from [course]", "How does [topic] work?", "What did [instructor] say about [topic]?"

Multi-Round Tool Usage Strategies:
- **Comparative queries**: First search one course/topic, then search another for comparison
- **Structure + Content queries**: First get course outline, then search specific lesson content
- **Refinement searches**: Initial broad search, then focused search based on results
- **Cross-course analysis**: Search multiple courses to find related topics or concepts

For outline queries, always return the complete course information:
- Course title and instructor
- Course link (if available)
- Complete lesson list with lesson numbers and titles
- Lesson links (if available)

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without tools
- **Course-specific questions**: Use appropriate tool first, then answer
- **Complex queries**: Use up to 2 rounds of tool calls to gather comprehensive information
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
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
        Generate AI response with sequential tool usage support (up to 2 rounds).

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default: 2)

        Returns:
            Generated response as string
        """

        round_count = 0
        messages = [{"role": "user", "content": query}]
        system_content = self._build_system_content(conversation_history)

        # Sequential tool calling loop (up to max_rounds)
        while round_count < max_rounds:
            # Make API call with tools available
            response = self._make_api_call(messages, system_content, tools)

            # Check termination conditions
            if not self._should_continue_rounds(
                response, round_count, max_rounds, tool_manager
            ):
                return response.content[0].text

            # Execute tools and prepare for next round
            messages = self._process_tool_round(response, messages, tool_manager)
            round_count += 1

        # Final call without tools after max rounds reached
        final_response = self._make_api_call(messages, system_content, tools=None)
        return final_response.content[0].text
    
    def _process_tool_round(self, response, messages: List, tool_manager):
        """
        Process one round of tool execution and update message chain.

        Args:
            response: The response containing tool use requests
            messages: Current message chain
            tool_manager: Manager to execute tools

        Returns:
            Updated messages list with tool use and tool results
        """
        # Add AI's tool use response to message chain
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
                    error_msg = f"Tool '{content_block.name}' failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                            "is_error": True,
                        }
                    )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        return messages
    
    def _make_api_call(
        self, messages: List, system_content: str, tools: Optional[List] = None
    ):
        """
        Make a single API call to Claude with consistent parameters.

        Args:
            messages: Message chain for the conversation
            system_content: System prompt content
            tools: Available tools (optional)

        Returns:
            API response from Claude
        """
        api_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        return self.client.messages.create(**api_params)
    
    def _should_continue_rounds(
        self, response, round_count: int, max_rounds: int, tool_manager
    ) -> bool:
        """
        Determine if tool calling should continue to next round.

        Args:
            response: Current API response
            round_count: Current round number
            max_rounds: Maximum allowed rounds
            tool_manager: Tool execution manager

        Returns:
            True if should continue, False if should terminate
        """
        # Terminate if no tool use in response
        if response.stop_reason != "tool_use":
            return False

        # Terminate if no tool manager available
        if not tool_manager:
            return False

        # Continue if within round limits
        return round_count < max_rounds
    
    def _build_system_content(self, conversation_history: Optional[str] = None) -> str:
        """
        Build system content with optional conversation history.

        Args:
            conversation_history: Previous conversation context

        Returns:
            Complete system content string
        """
        return (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )
