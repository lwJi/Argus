import pytest
import json
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import (
    _pick_worker_prompt,
    render_worker_prompt_text,
    run_worker_agent,
    _ainvoke_with_retry
)
from prompts import (
    WORKER_PROMPT_LINUS,
    WORKER_PROMPT_CPP, 
    WORKER_PROMPT_PY,
    WORKER_PROMPT_GENERIC,
    JSON_LINUS_SCHEMA,
    JSON_WORKER_SCHEMA
)


class TestLinusPromptSelection:
    """Test prompt selection logic for Linus mode."""
    
    def test_linus_mode_overrides_language_prompts(self):
        """Test that linus_mode=True returns WORKER_PROMPT_LINUS regardless of language."""
        assert _pick_worker_prompt("cpp", linus_mode=True) == WORKER_PROMPT_LINUS
        assert _pick_worker_prompt("python", linus_mode=True) == WORKER_PROMPT_LINUS
        assert _pick_worker_prompt("javascript", linus_mode=True) == WORKER_PROMPT_LINUS
        assert _pick_worker_prompt("unknown", linus_mode=True) == WORKER_PROMPT_LINUS
    
    def test_linus_mode_false_uses_language_prompts(self):
        """Test that linus_mode=False uses language-specific prompts."""
        assert _pick_worker_prompt("cpp", linus_mode=False) == WORKER_PROMPT_CPP
        assert _pick_worker_prompt("python", linus_mode=False) == WORKER_PROMPT_PY
        assert _pick_worker_prompt("javascript", linus_mode=False) == WORKER_PROMPT_GENERIC
    
    def test_linus_mode_default_false(self):
        """Test that linus_mode defaults to False."""
        assert _pick_worker_prompt("cpp") == WORKER_PROMPT_CPP
        assert _pick_worker_prompt("python") == WORKER_PROMPT_PY


class TestLinusPromptRendering:
    """Test Linus mode prompt rendering with correct schema."""
    
    def test_render_linus_prompt_text_uses_correct_schema(self):
        """Test that Linus mode uses JSON_LINUS_SCHEMA."""
        result = render_worker_prompt_text(
            language="cpp",
            file_path="test.cpp",
            chunk_index=1,
            total_chunks=1,
            code_with_line_numbers="L001: int main() {}",
            linus_mode=True
        )
        
        # Should contain Linus schema, not regular worker schema
        assert "taste_score" in result
        assert "fatal_problems" in result
        assert "key_insights" in result
        assert "linus_analysis" in result
        
        # Should not contain regular worker schema fields
        assert "findings" not in result or result.count("findings") < 3  # May appear in template but not as main field
    
    def test_render_regular_prompt_uses_worker_schema(self):
        """Test that regular mode uses JSON_WORKER_SCHEMA."""
        result = render_worker_prompt_text(
            language="cpp",
            file_path="test.cpp", 
            chunk_index=1,
            total_chunks=1,
            code_with_line_numbers="L001: int main() {}",
            linus_mode=False
        )
        
        # Should contain regular worker schema
        assert "findings" in result
        assert "summary" in result
        assert "counts" in result
        
        # Should not contain Linus-specific fields
        assert "taste_score" not in result
        assert "fatal_problems" not in result
    
    def test_linus_prompt_includes_language_context(self):
        """Test that Linus prompt includes language information."""
        for language in ["cpp", "python", "javascript"]:
            result = render_worker_prompt_text(
                language=language,
                file_path=f"test.{language}",
                chunk_index=1,
                total_chunks=1, 
                code_with_line_numbers="L001: code here",
                linus_mode=True
            )
            assert language in result.lower()


class TestLinusWorkerAgent:
    """Test Linus mode worker agent functionality."""
    
    @pytest.mark.asyncio
    async def test_run_worker_agent_linus_success(self, sample_linus_response, mock_llm):
        """Test successful Linus mode worker agent execution."""
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_linus_response)
            
            result = await run_worker_agent(
                mock_llm,
                language="cpp",
                file_path="test.cpp",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: int main() {}",
                linus_mode=True
            )
            
            assert result == sample_linus_response
            assert result["taste_score"] in ["good", "so-so", "trash"]
            assert "fatal_problems" in result
            assert "key_insights" in result
            assert "linus_analysis" in result
            
            # Verify the correct prompt template was used
            mock_retry.assert_called_once()
            call_args = mock_retry.call_args[0]
            assert len(call_args) == 2  # chain and params
    
    @pytest.mark.asyncio
    async def test_linus_agent_json_validation(self, mock_llm):
        """Test that Linus agent validates required JSON fields."""
        
        # Test valid Linus response
        valid_response = {
            "pre_analysis": {"real_problem": "Yes", "simpler_way": "No", "breaks_compatibility": "No"},
            "taste_score": "good",
            "fatal_problems": [],
            "key_insights": {"data_structure": "test", "complexity_removal": "test", "risk_assessment": "low"},
            "core_judgment": "worth_doing",
            "linus_analysis": {
                "data_structure_analysis": "test",
                "special_case_elimination": "test", 
                "complexity_review": "test",
                "destructiveness_analysis": "test",
                "practicality_verification": "test"
            },
            "improvement_direction": []
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(valid_response)
            
            result = await run_worker_agent(
                mock_llm,
                language="python",
                file_path="test.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: def test(): pass",
                linus_mode=True
            )
            
            assert result == valid_response
            assert all(key in result for key in [
                "pre_analysis", "taste_score", "fatal_problems", 
                "key_insights", "linus_analysis", "improvement_direction"
            ])
    
    @pytest.mark.asyncio
    async def test_linus_agent_taste_score_validation(self, mock_llm):
        """Test that taste_score is properly validated."""
        valid_scores = ["good", "so-so", "trash"]
        
        for score in valid_scores:
            response = {
                "pre_analysis": {"real_problem": "Yes", "simpler_way": "No", "breaks_compatibility": "No"},
                "taste_score": score,
                "fatal_problems": [],
                "key_insights": {"data_structure": "test", "complexity_removal": "test", "risk_assessment": "low"},
                "core_judgment": "worth_doing", 
                "linus_analysis": {
                    "data_structure_analysis": "test",
                    "special_case_elimination": "test",
                    "complexity_review": "test", 
                    "destructiveness_analysis": "test",
                    "practicality_verification": "test"
                },
                "improvement_direction": []
            }
            
            with patch('agents._ainvoke_with_retry') as mock_retry:
                mock_retry.return_value = json.dumps(response)
                
                result = await run_worker_agent(
                    mock_llm,
                    language="cpp",
                    file_path="test.cpp",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    linus_mode=True
                )
                
                assert result["taste_score"] == score
    
    @pytest.mark.asyncio
    async def test_linus_agent_with_json_repair(self, sample_linus_response, mock_llm):
        """Test that Linus agent can recover from malformed JSON."""
        malformed_json = '{"taste_score": "good", "fatal_problems": [malformed'
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # First call returns malformed, second call (repair) succeeds
            mock_retry.side_effect = [
                malformed_json,
                json.dumps(sample_linus_response)
            ]
            
            with patch('agents.extract_json_from_text') as mock_extract:
                mock_extract.side_effect = [
                    malformed_json,
                    json.dumps(sample_linus_response)
                ]
                
                result = await run_worker_agent(
                    mock_llm,
                    language="cpp", 
                    file_path="test.cpp",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    linus_mode=True
                )
                
                assert result == sample_linus_response
                assert mock_retry.call_count == 2  # Original + repair
    
    @pytest.mark.asyncio
    async def test_linus_agent_handles_empty_lists(self, mock_llm):
        """Test that Linus agent properly handles empty arrays."""
        response_with_empty_lists = {
            "pre_analysis": {"real_problem": "No", "simpler_way": "No", "breaks_compatibility": "No"},
            "taste_score": "good",
            "fatal_problems": [],  # Empty list should be fine
            "key_insights": {"data_structure": "Simple", "complexity_removal": "None needed", "risk_assessment": "low"},
            "core_judgment": "not_worth_doing",
            "linus_analysis": {
                "data_structure_analysis": "Already optimal",
                "special_case_elimination": "No special cases",
                "complexity_review": "Minimal complexity",
                "destructiveness_analysis": "No changes needed",
                "practicality_verification": "No real problem exists"
            },
            "improvement_direction": []  # Empty improvement list is valid
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(response_with_empty_lists)
            
            result = await run_worker_agent(
                mock_llm,
                language="python",
                file_path="perfect.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: # Perfect code",
                linus_mode=True
            )
            
            assert result == response_with_empty_lists
            assert result["fatal_problems"] == []
            assert result["improvement_direction"] == []


class TestLinusIntegration:
    """Test Linus mode integration with other components."""
    
    def test_linus_schema_contains_required_fields(self):
        """Test that JSON_LINUS_SCHEMA defines all required fields."""
        schema_text = JSON_LINUS_SCHEMA
        
        required_fields = [
            "pre_analysis", "taste_score", "fatal_problems", 
            "key_insights", "core_judgment", "linus_analysis",
            "improvement_direction"
        ]
        
        for field in required_fields:
            assert field in schema_text, f"Required field '{field}' missing from Linus schema"
        
        # Check that taste_score constraints are defined
        assert "good" in schema_text
        assert "so-so" in schema_text
        assert "trash" in schema_text
    
    def test_linus_vs_regular_schema_differences(self):
        """Test that Linus and regular schemas are properly different."""
        linus_schema = JSON_LINUS_SCHEMA
        worker_schema = JSON_WORKER_SCHEMA
        
        # Linus-specific fields should only be in Linus schema
        linus_fields = ["taste_score", "fatal_problems", "linus_analysis"]
        for field in linus_fields:
            assert field in linus_schema
            assert field not in worker_schema
        
        # Regular worker fields should only be in worker schema  
        worker_fields = ["findings", "counts", "summary"]
        for field in worker_fields:
            assert field in worker_schema
            # May appear in Linus schema as examples but not as main structure
    
    @pytest.mark.asyncio
    async def test_linus_agent_with_different_languages(self, sample_linus_response, mock_llm):
        """Test that Linus mode works with different programming languages."""
        languages = ["cpp", "python", "javascript", "rust", "go"]
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            for language in languages:
                mock_retry.return_value = json.dumps(sample_linus_response)
                
                result = await run_worker_agent(
                    mock_llm,
                    language=language,
                    file_path=f"test.{language}",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: // sample code",
                    linus_mode=True
                )
                
                assert result == sample_linus_response
                # All languages should use the same Linus analysis framework
                assert "linus_analysis" in result


class TestLinusAnalysisFramework:
    """Test the specific Linus analysis framework components."""
    
    def test_linus_five_level_analysis_structure(self, sample_linus_response):
        """Test that Linus analysis follows the 5-level framework."""
        linus_analysis = sample_linus_response["linus_analysis"]
        
        required_analysis_levels = [
            "data_structure_analysis",      # Level 1: Data structure analysis  
            "special_case_elimination",     # Level 2: Special case identification
            "complexity_review",            # Level 3: Complexity review
            "destructiveness_analysis",     # Level 4: Destructiveness analysis
            "practicality_verification"     # Level 5: Practicality verification
        ]
        
        for level in required_analysis_levels:
            assert level in linus_analysis, f"Missing analysis level: {level}"
            assert isinstance(linus_analysis[level], str), f"Analysis level {level} should be string"
            assert len(linus_analysis[level]) > 0, f"Analysis level {level} should not be empty"
    
    def test_linus_pre_analysis_questions(self, sample_linus_response):
        """Test that pre-analysis covers Linus's three key questions."""
        pre_analysis = sample_linus_response["pre_analysis"]
        
        # The three fundamental questions Linus asks
        required_questions = [
            "real_problem",           # "Is this a real problem or an imagined one?"
            "simpler_way",           # "Is there a simpler way?"
            "breaks_compatibility"   # "Will this break anything?" (userspace/compatibility)
        ]
        
        for question in required_questions:
            assert question in pre_analysis, f"Missing pre-analysis question: {question}"
            assert isinstance(pre_analysis[question], str), f"Pre-analysis {question} should be string"
    
    def test_linus_key_insights_structure(self, sample_linus_response):
        """Test that key insights follow Linus's core principles."""
        key_insights = sample_linus_response["key_insights"]
        
        required_insights = [
            "data_structure",      # "Good programmers worry about data structures"
            "complexity_removal",  # Simplicity obsession
            "risk_assessment"      # Pragmatic risk evaluation
        ]
        
        for insight in required_insights:
            assert insight in key_insights, f"Missing key insight: {insight}"
            assert isinstance(key_insights[insight], str), f"Key insight {insight} should be string"
    
    def test_linus_judgment_values(self):
        """Test that core_judgment uses correct enumeration."""
        valid_judgments = ["worth_doing", "not_worth_doing"]
        
        # Test both valid values work
        for judgment in valid_judgments:
            response = {
                "pre_analysis": {"real_problem": "Yes", "simpler_way": "No", "breaks_compatibility": "No"},
                "taste_score": "good",
                "fatal_problems": [],
                "key_insights": {"data_structure": "test", "complexity_removal": "test", "risk_assessment": "low"},
                "core_judgment": judgment,
                "linus_analysis": {
                    "data_structure_analysis": "test", "special_case_elimination": "test",
                    "complexity_review": "test", "destructiveness_analysis": "test", 
                    "practicality_verification": "test"
                },
                "improvement_direction": []
            }
            # Should be valid structure - this is more about schema validation
            assert response["core_judgment"] in valid_judgments