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