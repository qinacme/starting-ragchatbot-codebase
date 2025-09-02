"""
Unit tests for AIGenerator tool calling functionality
Tests Claude API integration and tool orchestration
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator


class TestAIGenerator:
    """Test cases for AIGenerator"""

    def test_init(self):
        """Test AIGenerator initialization"""
        api_key = "test-key"
        model = "claude-sonnet-4-20250514"
        
        ai_gen = AIGenerator(api_key, model)
        
        assert ai_gen.model == model
        assert ai_gen.base_params["model"] == model
        assert ai_gen.base_params["temperature"] == 0
        assert ai_gen.base_params["max_tokens"] == 800

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_simple(self, mock_anthropic):
        """Test simple response generation without tools"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response("test query")
        
        # Verify client was called correctly
        mock_client.messages.create.assert_called_once()
        call_args = mock_client.messages.create.call_args
        
        # Check message structure
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "test query"
        assert call_args[1]["model"] == "test-model"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        
        assert result == "Test response"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response with history"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        history = "Previous conversation context"
        result = ai_gen.generate_response("test query", conversation_history=history)
        
        # Verify system message includes history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert history in system_content
        
        assert result == "Response with history"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tools(self, mock_anthropic, sample_tool_definitions, mock_tool_manager):
        """Test response generation with tools available"""
        # Setup mock - no tool use in this case
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Direct response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "test query", 
            tools=sample_tool_definitions, 
            tool_manager=mock_tool_manager
        )
        
        # Verify tools were included in API call
        call_args = mock_client.messages.create.call_args
        assert "tools" in call_args[1]
        assert call_args[1]["tools"] == sample_tool_definitions
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        
        assert result == "Direct response"

    @patch('ai_generator.anthropic.Anthropic')
    def test_generate_response_with_tool_use(self, mock_anthropic, sample_tool_definitions, mock_tool_manager):
        """Test response generation when Claude uses tools"""
        # Setup mock for tool use response
        mock_client = Mock()
        
        # First response with tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test-tool-id"
        tool_block.input = {"query": "test query"}
        tool_response.content = [tool_block]
        
        # Final response after tool execution
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Response after tool use"
        
        # Mock client calls
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "test query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query"
        )
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
        
        assert result == "Response after tool use"

    @patch('ai_generator.anthropic.Anthropic')
    def test_process_tool_round_flow(self, mock_anthropic, sample_tool_definitions):
        """Test the tool round processing flow"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Create mock tool use response
        response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test-tool-id"
        tool_block.input = {"query": "test query", "course_name": "Test Course"}
        response.content = [tool_block]
        
        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Search results"
        
        messages = [{"role": "user", "content": "test query"}]
        
        updated_messages = ai_gen._process_tool_round(response, messages, mock_tool_manager)
        
        # Verify tool was executed with correct parameters
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query",
            course_name="Test Course"
        )
        
        # Verify message structure
        # Should have 3 messages: original user, assistant tool use, user tool result
        assert len(updated_messages) == 3
        assert updated_messages[0]["role"] == "user"
        assert updated_messages[1]["role"] == "assistant"
        assert updated_messages[2]["role"] == "user"
        
        # Tool result should be formatted correctly
        tool_results = updated_messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "test-tool-id"
        assert tool_results[0]["content"] == "Search results"

    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls(self, mock_anthropic):
        """Test handling multiple tool calls in one response"""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Create response with multiple tool blocks
        initial_response = Mock()
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool-1"
        tool_block1.input = {"query": "query 1"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool-2"
        tool_block2.input = {"course_title": "Test Course"}
        
        initial_response.content = [tool_block1, tool_block2]
        
        # Mock final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Multi-tool response"
        mock_client.messages.create.return_value = final_response
        
        # Mock tool manager and executions
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        messages = [{"role": "user", "content": "test query"}]
        
        updated_messages = ai_gen._process_tool_round(initial_response, messages, mock_tool_manager)
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="query 1")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_title="Test Course")
        
        # Verify tool results were formatted correctly
        tool_results = updated_messages[2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool-1"
        assert tool_results[0]["content"] == "Result 1"
        assert tool_results[1]["tool_use_id"] == "tool-2"
        assert tool_results[1]["content"] == "Result 2"

    def test_system_prompt_content(self):
        """Test that system prompt has expected content"""
        ai_gen = AIGenerator("test-key", "test-model")
        
        system_prompt = ai_gen.SYSTEM_PROMPT
        
        # Check key components of the system prompt
        assert "search_course_content" in system_prompt
        assert "get_course_outline" in system_prompt
        assert "tool calling" in system_prompt.lower() or "tool" in system_prompt.lower()
        assert "course" in system_prompt.lower()

    @patch('ai_generator.anthropic.Anthropic')
    def test_api_error_handling(self, mock_anthropic):
        """Test that API errors are properly propagated"""
        # Setup mock to raise exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        with pytest.raises(Exception, match="API Error"):
            ai_gen.generate_response("test query")

    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_manager_error_handling_old_behavior(self, mock_anthropic, sample_tool_definitions):
        """Test handling of tool manager errors in generate_response"""
        # Setup mock for tool use response
        mock_client = Mock()
        
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "test-tool-id"
        tool_block.input = {"query": "test query"}
        tool_response.content = [tool_block]
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Response after tool error"
        
        mock_client.messages.create.side_effect = [tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager that raises error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Should continue with error message instead of raising
        result = ai_gen.generate_response(
            "test query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Should get the final response despite the tool error
        assert result == "Response after tool error"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic, sample_tool_definitions):
        """Test sequential tool calling with 2 rounds"""
        mock_client = Mock()
        
        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool-1"
        tool_block1.input = {"query": "first search"}
        round1_response.content = [tool_block1]
        
        # Round 2: Tool use response
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content"
        tool_block2.id = "tool-2"
        tool_block2.input = {"query": "second search"}
        round2_response.content = [tool_block2]
        
        # Final response
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final response after 2 rounds"
        
        # Mock API calls: round1, round2, final
        mock_client.messages.create.side_effect = [round1_response, round2_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager and executions
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "complex query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify 3 API calls were made (2 tool rounds + final)
        assert mock_client.messages.create.call_count == 3
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first search")
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="second search")
        
        assert result == "Final response after 2 rounds"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_terminate_after_one_round(self, mock_anthropic, sample_tool_definitions):
        """Test termination after one round when Claude doesn't use tools in round 2"""
        mock_client = Mock()
        
        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool-1"
        tool_block.input = {"query": "search query"}
        round1_response.content = [tool_block]
        
        # Round 2: No tool use, direct text response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        round2_response.content = [Mock()]
        round2_response.content[0].text = "Direct response, no more tools needed"
        
        # Mock API calls
        mock_client.messages.create.side_effect = [round1_response, round2_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager and execution
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify only 2 API calls were made (tool round + response)
        assert mock_client.messages.create.call_count == 2
        
        # Verify only one tool was executed
        assert mock_tool_manager.execute_tool.call_count == 1
        
        assert result == "Direct response, no more tools needed"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_max_rounds_exceeded(self, mock_anthropic, sample_tool_definitions):
        """Test that tool calling stops after max rounds (2) even if Claude wants to continue"""
        mock_client = Mock()
        
        # Both rounds result in tool use
        tool_response = Mock()
        tool_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool-id"
        tool_block.input = {"query": "search"}
        tool_response.content = [tool_block]
        
        # Final response without tools
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final response after max rounds"
        
        # Mock API calls: 2 tool rounds + final call without tools
        mock_client.messages.create.side_effect = [tool_response, tool_response, final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager and executions
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Verify 3 API calls (2 rounds + final without tools)
        assert mock_client.messages.create.call_count == 3
        
        # Verify 2 tool executions
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify final call was made without tools
        final_call_args = mock_client.messages.create.call_args_list[-1]
        assert "tools" not in final_call_args[1] or final_call_args[1]["tools"] is None
        
        assert result == "Final response after max rounds"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_context_preservation(self, mock_anthropic, sample_tool_definitions):
        """Test that conversation context is preserved across multiple rounds"""
        mock_client = Mock()
        
        # Round 1: Tool use
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool-1"
        tool_block1.input = {"query": "first"}
        round1_response.content = [tool_block1]
        
        # Round 2: Direct response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        round2_response.content = [Mock()]
        round2_response.content[0].text = "Response with context"
        
        mock_client.messages.create.side_effect = [round1_response, round2_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager and execution
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "original query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Check that round 2 call includes full message history
        round2_call_args = mock_client.messages.create.call_args_list[1]
        messages = round2_call_args[1]["messages"]
        
        # Should have: user query + assistant tool use + user tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "original query"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Tool results should be properly formatted
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool-1"
        assert tool_results[0]["content"] == "Tool result"

    @patch('ai_generator.anthropic.Anthropic')
    def test_sequential_tool_calling_error_handling(self, mock_anthropic, sample_tool_definitions):
        """Test error handling in sequential tool calling"""
        mock_client = Mock()
        
        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool-1"
        tool_block.input = {"query": "search"}
        round1_response.content = [tool_block]
        
        # Round 2: Direct response
        round2_response = Mock()
        round2_response.stop_reason = "end_turn"
        round2_response.content = [Mock()]
        round2_response.content[0].text = "Response despite tool error"
        
        mock_client.messages.create.side_effect = [round1_response, round2_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager that raises error
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool failed")
        
        ai_gen = AIGenerator("test-key", "test-model")
        
        result = ai_gen.generate_response(
            "query",
            tools=sample_tool_definitions,
            tool_manager=mock_tool_manager
        )
        
        # Should continue despite tool error
        assert result == "Response despite tool error"
        
        # Check that error was included in tool results
        round2_call_args = mock_client.messages.create.call_args_list[1]
        messages = round2_call_args[1]["messages"]
        tool_results = messages[2]["content"]
        
        assert tool_results[0]["type"] == "tool_result"
        assert "Tool 'search_course_content' failed: Tool failed" in tool_results[0]["content"]
        assert tool_results[0]["is_error"] == True

    def test_helper_methods(self):
        """Test the new helper methods work correctly"""
        ai_gen = AIGenerator("test-key", "test-model")
        
        # Test _build_system_content
        content_without_history = ai_gen._build_system_content()
        assert content_without_history == ai_gen.SYSTEM_PROMPT
        
        content_with_history = ai_gen._build_system_content("Previous conversation")
        assert "Previous conversation" in content_with_history
        assert ai_gen.SYSTEM_PROMPT in content_with_history
        
        # Test _should_continue_rounds
        mock_response = Mock()
        mock_tool_manager = Mock()
        
        # Should continue: tool_use and valid tool_manager
        mock_response.stop_reason = "tool_use"
        assert ai_gen._should_continue_rounds(mock_response, 0, 2, mock_tool_manager) == True
        
        # Should not continue: end_turn
        mock_response.stop_reason = "end_turn"
        assert ai_gen._should_continue_rounds(mock_response, 0, 2, mock_tool_manager) == False
        
        # Should not continue: no tool_manager
        mock_response.stop_reason = "tool_use"
        assert ai_gen._should_continue_rounds(mock_response, 0, 2, None) == False
        
        # Should not continue: max rounds reached
        assert ai_gen._should_continue_rounds(mock_response, 2, 2, mock_tool_manager) == False