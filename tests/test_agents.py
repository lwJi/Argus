import pytest
import json
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_openai import ChatOpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import (
    run_worker_agent,
    run_supervisor_agent, 
    run_synthesizer_agent,
    format_reviews_for_supervisor,
    render_worker_prompt_text,
    render_supervisor_prompt_text,
    render_synthesizer_prompt_text,
    _pick_worker_prompt,
    _ainvoke_with_retry
)
from prompts import WORKER_PROMPT_CPP, WORKER_PROMPT_PY, WORKER_PROMPT_GENERIC

class TestPromptSelection:
    def test_pick_worker_prompt_cpp(self):
        assert _pick_worker_prompt("cpp") == WORKER_PROMPT_CPP
    
    def test_pick_worker_prompt_python(self):
        assert _pick_worker_prompt("python") == WORKER_PROMPT_PY
    
    def test_pick_worker_prompt_generic(self):
        assert _pick_worker_prompt("javascript") == WORKER_PROMPT_GENERIC
        assert _pick_worker_prompt("unknown") == WORKER_PROMPT_GENERIC

class TestPromptRendering:
    def test_render_worker_prompt_text_cpp(self):
        result = render_worker_prompt_text(
            language="cpp",
            file_path="test.cpp", 
            chunk_index=1,
            total_chunks=2,
            code_with_line_numbers="L001: int main() {}"
        )
        
        assert "test.cpp" in result
        assert "1/2" in result
        assert "L001: int main() {}" in result
        assert "C++" in result
    
    def test_render_worker_prompt_text_generic(self):
        result = render_worker_prompt_text(
            language="javascript",
            file_path="test.js",
            chunk_index=1,
            total_chunks=1,
            code_with_line_numbers="L001: function test() {}"
        )
        
        assert "javascript" in result
        assert "test.js" in result
        assert "L001: function test() {}" in result
    
    def test_render_supervisor_prompt_text(self):
        reviews = "Review 1: Good\nReview 2: Better"
        result = render_supervisor_prompt_text(reviews_text_block=reviews)
        
        assert reviews in result
        assert "Staff Software Engineer" in result
    
    def test_render_synthesizer_prompt_text(self):
        summaries = '{"chunk": 1}\n{"chunk": 2}'
        result = render_synthesizer_prompt_text(chunk_summaries_jsonl=summaries)
        
        assert summaries in result
        assert "Principal Engineer" in result

class TestFormatReviews:
    def test_format_reviews_for_supervisor(self):
        reviews = [
            {"summary": "Good code", "findings": []},
            {"summary": "Has issues", "findings": [{"type": "bug"}]}
        ]
        
        result = format_reviews_for_supervisor(reviews)
        
        assert "--- Review 1 JSON ---" in result
        assert "--- Review 2 JSON ---" in result
        assert "Good code" in result
        assert "Has issues" in result

class TestRetryMechanism:
    @pytest.mark.asyncio
    async def test_ainvoke_with_retry_success(self):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.return_value = "success"
        
        result = await _ainvoke_with_retry(mock_chain, {"input": "test"})
        assert result == "success"
        mock_chain.ainvoke.assert_called_once_with({"input": "test"})
    
    @pytest.mark.asyncio
    async def test_ainvoke_with_retry_eventually_succeeds(self):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = [
            Exception("First failure"),
            Exception("Second failure"), 
            "success"
        ]
        
        result = await _ainvoke_with_retry(mock_chain, {"input": "test"}, attempts=4)
        assert result == "success"
        assert mock_chain.ainvoke.call_count == 3
    
    @pytest.mark.asyncio
    async def test_ainvoke_with_retry_exhausts_attempts(self):
        mock_chain = AsyncMock()
        mock_chain.ainvoke.side_effect = Exception("Always fails")
        
        from tenacity import RetryError
        with pytest.raises(RetryError):
            await _ainvoke_with_retry(mock_chain, {"input": "test"}, attempts=2)

class TestAgentFunctions:
    @pytest.mark.asyncio
    async def test_run_worker_agent_success(self, sample_worker_response):
        mock_llm = MagicMock(spec=ChatOpenAI)
        
        # Mock the chain behavior
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_worker_response)
            
            result = await run_worker_agent(
                mock_llm,
                language="python",
                file_path="test.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: def test(): pass"
            )
            
            assert result == sample_worker_response
            assert mock_retry.call_count == 1
    
    @pytest.mark.asyncio
    async def test_run_worker_agent_with_repair(self, sample_worker_response):
        mock_llm = MagicMock(spec=ChatOpenAI)
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # First call returns invalid JSON, second call succeeds
            mock_retry.side_effect = [
                "Invalid JSON response",  # First attempt
                json.dumps(sample_worker_response)  # Repair attempt
            ]
            
            with patch('agents.extract_json_from_text') as mock_extract:
                mock_extract.side_effect = [
                    "Invalid JSON response",  # First extraction
                    json.dumps(sample_worker_response)  # Second extraction
                ]
                
                result = await run_worker_agent(
                    mock_llm,
                    language="python", 
                    file_path="test.py",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: def test(): pass"
                )
                
                assert result == sample_worker_response
                assert mock_retry.call_count == 2  # Original + repair
    
    @pytest.mark.asyncio
    async def test_run_supervisor_agent_success(self, sample_supervisor_response):
        mock_llm = MagicMock(spec=ChatOpenAI)
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_supervisor_response)
            
            result = await run_supervisor_agent(
                mock_llm,
                reviews_text_block="Sample reviews text"
            )
            
            assert result == sample_supervisor_response
    
    @pytest.mark.asyncio
    async def test_run_synthesizer_agent_success(self):
        mock_llm = MagicMock(spec=ChatOpenAI)
        expected_markdown = "# Code Review\n\nSummary of findings..."
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = expected_markdown
            
            result = await run_synthesizer_agent(
                mock_llm,
                chunk_summaries_jsonl='{"chunk": 1}\n{"chunk": 2}'
            )
            
            assert result == expected_markdown

class TestAgentInputValidation:
    @pytest.mark.asyncio
    async def test_worker_agent_with_empty_code(self):
        mock_llm = MagicMock(spec=ChatOpenAI)
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = '{"summary": "Empty file", "findings": [], "counts": {"bug": 0, "performance": 0, "style": 0, "maintainability": 0}}'
            
            result = await run_worker_agent(
                mock_llm,
                language="python",
                file_path="empty.py", 
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers=""
            )
            
            assert "summary" in result
            assert "findings" in result
            assert "counts" in result


class TestIterativeAgents:
    """Test iterative-specific agent functionality."""
    
    @pytest.mark.asyncio
    async def test_run_iterative_worker_agent_basic(self, sample_worker_response, mock_llm):
        """Test basic iterative worker agent functionality."""
        from agents import run_iterative_worker_agent
        
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 3,
            "strategy": "worker_pool",
            "previous_iterations": []
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_worker_response)
            
            result = await run_iterative_worker_agent(
                mock_llm,
                language="python",
                file_path="test.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: def test(): pass",
                iteration_context=iteration_context,
                linus_mode=False
            )
            
            assert result == sample_worker_response
            mock_retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_run_iterative_supervisor_agent_basic(self, sample_iterative_supervisor_response, mock_llm):
        """Test basic iterative supervisor agent functionality."""
        from agents import run_iterative_supervisor_agent
        
        iteration_context = {
            "iteration": 2,
            "total_iterations_planned": 3,
            "strategy": "feedback_driven",
            "supervisor_feedback": "Previous feedback here"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_iterative_supervisor_response)
            
            result = await run_iterative_supervisor_agent(
                mock_llm,
                reviews_text_block="Sample reviews text",
                iteration_context=iteration_context
            )
            
            assert result == sample_iterative_supervisor_response
            assert "iteration_comparison" in result
            assert "feedback_for_next_iteration" in result
            mock_retry.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_iterative_worker_with_different_strategies(self, sample_worker_response, mock_llm):
        """Test iterative worker agent with different strategy contexts."""
        from agents import run_iterative_worker_agent
        
        strategies = [
            {
                "iteration": 2,
                "strategy": "worker_pool",
                "previous_iterations": [{"iteration": 1, "summary": "First iteration summary"}]
            },
            {
                "iteration": 2, 
                "strategy": "feedback_driven",
                "supervisor_feedback": "Focus on error handling improvements"
            },
            {
                "iteration": 2,
                "strategy": "consensus", 
                "peer_reviews_previous": [{"review": "peer1"}, {"review": "peer2"}]
            }
        ]
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            for strategy_context in strategies:
                mock_retry.return_value = json.dumps(sample_worker_response)
                
                result = await run_iterative_worker_agent(
                    mock_llm,
                    language="python",
                    file_path="test.py", 
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    iteration_context=strategy_context,
                    linus_mode=False
                )
                
                assert result == sample_worker_response
                
                # Verify that strategy-specific context was included in the call
                call_args = mock_retry.call_args[0] 
                chain, params = call_args
                assert params["strategy"] == strategy_context["strategy"]
    
    @pytest.mark.asyncio
    async def test_iterative_worker_with_linus_mode(self, sample_linus_response, mock_llm):
        """Test iterative worker agent in Linus mode."""
        from agents import run_iterative_worker_agent
        
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 2,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_linus_response)
            
            result = await run_iterative_worker_agent(
                mock_llm,
                language="cpp",
                file_path="test.cpp",
                chunk_index=1, 
                total_chunks=1,
                code_with_line_numbers="L001: int main() {}",
                iteration_context=iteration_context,
                linus_mode=True
            )
            
            assert result == sample_linus_response
            assert "taste_score" in result
            assert "linus_analysis" in result
            assert result["taste_score"] in ["good", "so-so", "trash"]
    
    def test_render_iterative_supervisor_prompt_text(self):
        """Test iterative supervisor prompt rendering."""
        from agents import render_iterative_supervisor_prompt_text
        
        iteration_context = {
            "iteration": 2,
            "total_iterations_planned": 3,
            "strategy": "feedback_driven",
            "supervisor_feedback": "Previous supervisor feedback here"
        }
        
        prompt = render_iterative_supervisor_prompt_text(
            reviews_text_block="Sample reviews for comparison",
            iteration_context=iteration_context
        )
        
        # Verify iteration context is included
        assert "iteration 2 of 3" in prompt
        assert "feedback_driven" in prompt
        assert "Previous supervisor feedback here" in prompt
        assert "Sample reviews for comparison" in prompt
        
        # Verify iterative schema is used - check for iterative-specific fields
        assert "iteration_comparison" in prompt
        assert "feedback_for_next_iteration" in prompt
    
    def test_render_iterative_supervisor_prompt_different_strategies(self):
        """Test iterative supervisor prompt rendering with different strategies."""
        from agents import render_iterative_supervisor_prompt_text
        
        # Test feedback_driven strategy
        feedback_context = {
            "iteration": 2,
            "strategy": "feedback_driven",
            "supervisor_feedback": "Focus on performance issues"
        }
        
        prompt = render_iterative_supervisor_prompt_text(
            reviews_text_block="reviews", 
            iteration_context=feedback_context
        )
        assert "Focus on performance issues" in prompt
        
        # Test consensus strategy
        consensus_context = {
            "iteration": 2,
            "strategy": "consensus",
            "peer_reviews_previous": [{"review": "1"}, {"review": "2"}]
        }
        
        prompt = render_iterative_supervisor_prompt_text(
            reviews_text_block="reviews",
            iteration_context=consensus_context
        )
        assert "2 anonymized reviews from previous iteration" in prompt
    
    @pytest.mark.asyncio
    async def test_iterative_agents_error_handling(self, mock_llm):
        """Test error handling in iterative agents."""
        from agents import run_iterative_worker_agent, run_iterative_supervisor_agent
        
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 2,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.side_effect = Exception("Network error")
            
            # Worker agent should propagate error
            with pytest.raises(Exception, match="Network error"):
                await run_iterative_worker_agent(
                    mock_llm,
                    language="python",
                    file_path="test.py",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    iteration_context=iteration_context
                )
            
            # Supervisor agent should propagate error
            with pytest.raises(Exception, match="Network error"):
                await run_iterative_supervisor_agent(
                    mock_llm,
                    reviews_text_block="reviews",
                    iteration_context=iteration_context
                )
    
    @pytest.mark.asyncio
    async def test_iterative_worker_json_repair(self, sample_worker_response, mock_llm):
        """Test JSON repair in iterative worker agent."""
        from agents import run_iterative_worker_agent
        
        iteration_context = {
            "iteration": 1,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # First call returns malformed JSON, second call (repair) succeeds
            mock_retry.side_effect = [
                '{"summary": "incomplete json...',  # Malformed
                json.dumps(sample_worker_response)   # Repaired
            ]
            
            with patch('agents.extract_json_from_text') as mock_extract:
                mock_extract.side_effect = [
                    '{"summary": "incomplete json...',
                    json.dumps(sample_worker_response)
                ]
                
                result = await run_iterative_worker_agent(
                    mock_llm,
                    language="python",
                    file_path="test.py",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    iteration_context=iteration_context
                )
                
                assert result == sample_worker_response
                assert mock_retry.call_count == 2  # Original + repair
    
    def test_iteration_context_validation(self):
        """Test that iteration context contains expected fields."""
        from agents import IterationController, IterationStrategy
        
        controller = IterationController(
            max_iterations=3,
            strategy=IterationStrategy.FEEDBACK_DRIVEN
        )
        
        context = controller.get_context_for_iteration(2)
        
        # Required fields
        required_fields = ["iteration", "total_iterations_planned", "strategy", "previous_iterations"]
        for field in required_fields:
            assert field in context, f"Missing required context field: {field}"
        
        assert context["iteration"] == 2
        assert context["total_iterations_planned"] == 3
        assert context["strategy"] == "feedback_driven"
        assert isinstance(context["previous_iterations"], list)
    
    @pytest.mark.asyncio
    async def test_iterative_agents_preserve_chunk_context(self, sample_worker_response, mock_llm):
        """Test that iterative agents preserve chunk-specific context."""
        from agents import run_iterative_worker_agent
        
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 2,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_worker_response)
            
            result = await run_iterative_worker_agent(
                mock_llm,
                language="javascript",
                file_path="complex/nested/file.js",
                chunk_index=3,
                total_chunks=5,
                code_with_line_numbers="L100: function complex() {}",
                iteration_context=iteration_context
            )
            
            # Verify call includes chunk context
            call_args = mock_retry.call_args[0]
            chain, params = call_args
            
            assert params["file_path"] == "complex/nested/file.js"
            assert params["chunk_index"] == 3
            assert params["total_chunks"] == 5
            assert params["language"] == "javascript"
            assert "L100: function complex() {}" in params["code_with_line_numbers"]
            
            # Verify iteration context is also included
            assert params["iteration"] == 1
            assert params["total_iterations"] == 2