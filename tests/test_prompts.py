import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompts import (
    WORKER_PROMPT_GENERIC,
    WORKER_PROMPT_CPP, 
    WORKER_PROMPT_PY,
    SUPERVISOR_PROMPT,
    SYNTHESIZER_PROMPT,
    JSON_WORKER_SCHEMA,
    JSON_SUPERVISOR_SCHEMA
)

class TestPromptTemplates:
    def test_worker_prompt_generic_format(self):
        formatted = WORKER_PROMPT_GENERIC.format(
            language="javascript",
            file_path="test.js",
            chunk_index=1,
            total_chunks=2,
            code_with_line_numbers="L001: function test() {}",
            json_schema=JSON_WORKER_SCHEMA
        )
        
        assert "javascript" in formatted
        assert "test.js" in formatted
        assert "1/2" in formatted
        assert "L001: function test() {}" in formatted
        assert "Return ONLY valid JSON" in formatted
    
    def test_worker_prompt_cpp_format(self):
        formatted = WORKER_PROMPT_CPP.format(
            file_path="test.cpp",
            chunk_index=2,
            total_chunks=3,
            code_with_line_numbers="L001: int main() {}",
            json_schema=JSON_WORKER_SCHEMA
        )
        
        assert "test.cpp" in formatted
        assert "2/3" in formatted
        assert "L001: int main() {}" in formatted
        assert "C++ Core Guidelines" in formatted
        assert "RAII" in formatted
    
    def test_worker_prompt_py_format(self):
        formatted = WORKER_PROMPT_PY.format(
            file_path="test.py",
            chunk_index=1,
            total_chunks=1,
            code_with_line_numbers="L001: def hello(): pass",
            json_schema=JSON_WORKER_SCHEMA
        )
        
        assert "test.py" in formatted
        assert "1/1" in formatted
        assert "L001: def hello(): pass" in formatted
        assert "PEP 8" in formatted
        assert "type hints" in formatted
    
    def test_supervisor_prompt_format(self):
        reviews = "Review 1: Good\nReview 2: Better"
        formatted = SUPERVISOR_PROMPT.format(
            reviews=reviews,
            json_schema=JSON_SUPERVISOR_SCHEMA
        )
        
        assert reviews in formatted
        assert "Staff Software Engineer" in formatted
        assert "Accuracy" in formatted
        assert "Completeness" in formatted
    
    def test_synthesizer_prompt_format(self):
        summaries = '{"chunk": 1, "summary": "test"}\n{"chunk": 2, "summary": "test2"}'
        formatted = SYNTHESIZER_PROMPT.format(
            chunk_summaries=summaries
        )
        
        assert summaries in formatted
        assert "Principal Engineer" in formatted
        assert "executive summary" in formatted
        assert "Markdown" in formatted

class TestJsonSchemas:
    def test_worker_schema_structure(self):
        schema = JSON_WORKER_SCHEMA
        
        # Check required structure elements
        assert "summary" in schema
        assert "findings" in schema
        assert "counts" in schema
        assert "type" in schema
        assert "severity" in schema
        assert "bug" in schema
        assert "performance" in schema
        assert "style" in schema
        assert "maintainability" in schema
    
    def test_supervisor_schema_structure(self):
        schema = JSON_SUPERVISOR_SCHEMA
        
        # Check required structure elements
        assert "analysis" in schema
        assert "scores" in schema
        assert "winner_index" in schema
        assert "merged_takeaways" in schema
        assert "winning_review_text" in schema
        assert "accuracy" in schema
        assert "completeness" in schema
        assert "clarity" in schema
        assert "insightfulness" in schema

class TestPromptInputVariables:
    def test_worker_generic_input_variables(self):
        expected_vars = ["language", "file_path", "chunk_index", 
                        "total_chunks", "code_with_line_numbers", "json_schema"]
        assert set(WORKER_PROMPT_GENERIC.input_variables) == set(expected_vars)
    
    def test_worker_cpp_input_variables(self):
        expected_vars = ["file_path", "chunk_index",
                        "total_chunks", "code_with_line_numbers", "json_schema"]
        assert set(WORKER_PROMPT_CPP.input_variables) == set(expected_vars)
    
    def test_worker_py_input_variables(self):
        expected_vars = ["file_path", "chunk_index",
                        "total_chunks", "code_with_line_numbers", "json_schema"]
        assert set(WORKER_PROMPT_PY.input_variables) == set(expected_vars)
    
    def test_supervisor_input_variables(self):
        expected_vars = ["reviews", "json_schema"]
        assert set(SUPERVISOR_PROMPT.input_variables) == set(expected_vars)
    
    def test_synthesizer_input_variables(self):
        expected_vars = ["chunk_summaries"]
        assert set(SYNTHESIZER_PROMPT.input_variables) == set(expected_vars)

class TestPromptContent:
    def test_worker_prompts_mention_line_numbers(self):
        # All worker prompts should mention using line numbers
        assert "L###" in WORKER_PROMPT_GENERIC.template or "line number" in WORKER_PROMPT_GENERIC.template
        assert "L###" in WORKER_PROMPT_CPP.template or "line number" in WORKER_PROMPT_CPP.template  
        assert "L###" in WORKER_PROMPT_PY.template or "line number" in WORKER_PROMPT_PY.template
    
    def test_worker_prompts_require_json(self):
        # All worker prompts should require JSON output
        assert "JSON" in WORKER_PROMPT_GENERIC.template
        assert "JSON" in WORKER_PROMPT_CPP.template
        assert "JSON" in WORKER_PROMPT_PY.template
    
    def test_cpp_prompt_mentions_specific_concepts(self):
        template = WORKER_PROMPT_CPP.template
        assert "RAII" in template
        assert "const-correctness" in template
        assert "Core Guidelines" in template
    
    def test_python_prompt_mentions_pep8(self):
        template = WORKER_PROMPT_PY.template
        assert "PEP 8" in template or "PEP8" in template
        assert "type hints" in template