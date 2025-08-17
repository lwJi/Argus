import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import run_worker_agent, run_supervisor_agent, run_synthesizer_agent, format_reviews_for_supervisor
from utils import (
    chunk_code_by_lines,
    add_line_numbers_preserve, 
    detect_language_from_extension,
    save_json,
    save_text
)

class TestEndToEndPipeline:
    @pytest.mark.asyncio
    async def test_single_chunk_workflow(self, sample_python_code, sample_worker_response, sample_supervisor_response):
        """Test the complete workflow for a single chunk."""
        mock_worker_llm = MagicMock()
        mock_supervisor_llm = MagicMock()
        mock_synthesizer_llm = MagicMock()
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # Mock responses for worker, supervisor, and synthesizer
            mock_retry.side_effect = [
                json.dumps(sample_worker_response),  # Worker 1
                json.dumps(sample_worker_response),  # Worker 2  
                json.dumps(sample_supervisor_response),  # Supervisor
                "# Final Review\n\nCode analysis complete."  # Synthesizer
            ]
            
            # Step 1: Process with workers
            language = detect_language_from_extension(".py")
            chunks = chunk_code_by_lines(sample_python_code, max_lines=100)
            chunk_content = add_line_numbers_preserve(chunks[0][1], start_line=1)
            
            worker_results = []
            for worker_llm in [mock_worker_llm, mock_worker_llm]:
                result = await run_worker_agent(
                    worker_llm,
                    language=language,
                    file_path="test.py",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers=chunk_content
                )
                worker_results.append(result)
            
            # Step 2: Supervisor selection
            reviews_text = format_reviews_for_supervisor(worker_results)
            supervisor_result = await run_supervisor_agent(
                mock_supervisor_llm,
                reviews_text_block=reviews_text
            )
            
            # Step 3: Final synthesis
            chunk_summaries = json.dumps({
                "chunk_index": 1,
                "summary": supervisor_result.get("winning_review_text", "")
            })
            
            final_report = await run_synthesizer_agent(
                mock_synthesizer_llm,
                chunk_summaries_jsonl=chunk_summaries
            )
            
            # Verify results
            assert len(worker_results) == 2
            assert supervisor_result["winner_index"] == 1
            assert "# Final Review" in final_report
            assert mock_retry.call_count == 4

    @pytest.mark.asyncio 
    async def test_multi_chunk_workflow(self, sample_cpp_code):
        """Test workflow with multiple chunks."""
        mock_llm = MagicMock()
        
        # Create code that will be split into multiple chunks
        long_code = sample_cpp_code + "\n" + sample_cpp_code + "\n" + sample_cpp_code
        chunks = chunk_code_by_lines(long_code, max_lines=10)  # Force multiple chunks
        
        assert len(chunks) > 1, "Test should create multiple chunks"
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps({
                "summary": "Test chunk analysis",
                "findings": [],
                "counts": {"bug": 0, "performance": 0, "style": 0, "maintainability": 0}
            })
            
            # Process each chunk
            chunk_results = []
            for idx, (start_line, chunk_content) in enumerate(chunks):
                code_with_lines = add_line_numbers_preserve(chunk_content, start_line)
                
                result = await run_worker_agent(
                    mock_llm,
                    language="cpp",
                    file_path="test.cpp",
                    chunk_index=idx + 1,
                    total_chunks=len(chunks),
                    code_with_line_numbers=code_with_lines
                )
                chunk_results.append(result)
            
            assert len(chunk_results) == len(chunks)
            assert all("summary" in result for result in chunk_results)

class TestFileProcessingIntegration:
    def test_file_discovery_and_chunking(self, temp_dir, sample_python_code, sample_cpp_code):
        """Test file discovery and processing preparation."""
        # Create test files
        py_file = Path(temp_dir) / "test.py"
        cpp_file = Path(temp_dir) / "test.cpp" 
        txt_file = Path(temp_dir) / "readme.txt"
        
        py_file.write_text(sample_python_code)
        cpp_file.write_text(sample_cpp_code)
        txt_file.write_text("Not source code")
        
        from utils import list_source_files
        
        # Test file discovery
        source_files = list_source_files(temp_dir, [".py", ".cpp"])
        assert len(source_files) == 2
        assert any("test.py" in f for f in source_files)
        assert any("test.cpp" in f for f in source_files)
        assert not any("readme.txt" in f for f in source_files)
        
        # Test language detection and chunking
        for file_path in source_files:
            ext = Path(file_path).suffix
            language = detect_language_from_extension(ext)
            
            with open(file_path, 'r') as f:
                content = f.read()
            
            chunks = chunk_code_by_lines(content)
            assert len(chunks) >= 1
            assert all(isinstance(start_line, int) for start_line, _ in chunks)
            assert all(isinstance(chunk_text, str) for _, chunk_text in chunks)
            
            if ext == ".py":
                assert language == "python"
            elif ext == ".cpp":
                assert language == "cpp"

class TestErrorHandling:
    @pytest.mark.asyncio
    async def test_invalid_json_response_handling(self):
        """Test handling of invalid JSON responses from LLM."""
        mock_llm = MagicMock()
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # First call returns invalid JSON, second call (repair) succeeds
            mock_retry.side_effect = [
                "This is not valid JSON at all",
                json.dumps({
                    "summary": "Repaired response",
                    "findings": [],
                    "counts": {"bug": 0, "performance": 0, "style": 0, "maintainability": 0}
                })
            ]
            
            result = await run_worker_agent(
                mock_llm,
                language="python",
                file_path="test.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: def test(): pass"
            )
            
            assert result["summary"] == "Repaired response"
            assert mock_retry.call_count == 2  # Original + repair attempt
    
    def test_malformed_file_handling(self, temp_dir):
        """Test handling of files that can't be read."""
        from utils import list_source_files
        
        # Create a file and then make it unreadable (simulate permission issues)
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("print('hello')")
        
        # This should work normally
        files = list_source_files(temp_dir, [".py"])
        assert len(files) == 1
        
        # Test reading the file works
        with open(files[0], 'r') as f:
            content = f.read()
        assert "hello" in content

class TestOutputGeneration:
    def test_save_operations(self, temp_dir):
        """Test saving JSON and text outputs."""
        # Test JSON saving
        test_data = {
            "file": "test.py",
            "language": "python", 
            "results": ["result1", "result2"]
        }
        
        json_path = os.path.join(temp_dir, "test.json")
        save_json(json_path, test_data)
        
        # Verify JSON was saved correctly
        with open(json_path, 'r') as f:
            loaded_data = json.load(f)
        assert loaded_data == test_data
        
        # Test text saving
        markdown_content = "# Test Report\n\nThis is a test report."
        text_path = os.path.join(temp_dir, "test.md")
        save_text(text_path, markdown_content)
        
        # Verify text was saved correctly
        with open(text_path, 'r') as f:
            loaded_text = f.read()
        assert loaded_text == markdown_content

class TestConfigurationIntegration:
    def test_models_config_loading(self, temp_dir, sample_models_config):
        """Test loading and using models configuration."""
        import yaml
        from utils import load_models_config
        
        # Save test config
        config_path = os.path.join(temp_dir, "test_models.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sample_models_config, f)
        
        # Load and verify
        loaded_config = load_models_config(config_path)
        assert loaded_config == sample_models_config
        assert len(loaded_config["workers"]) == 2
        assert loaded_config["supervisor"]["model"] == "gpt-4o"
    
    def test_missing_config_handling(self):
        """Test handling of missing configuration files."""
        from utils import load_models_config
        
        result = load_models_config("nonexistent.yaml")
        assert result == {}


class TestLinusModeIntegration:
    """Test Linus mode integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_linus_mode_end_to_end(self, sample_cpp_code, sample_linus_response, mock_llm):
        """Test complete Linus mode workflow."""
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # Mock Linus worker response, supervisor response, and synthesizer
            mock_retry.side_effect = [
                json.dumps(sample_linus_response),  # Linus worker
                json.dumps({  # Supervisor handling Linus output
                    "analysis": "Linus analysis provides systematic framework",
                    "scores": [{"accuracy": 0.95, "completeness": 0.9, "clarity": 0.9, "insightfulness": 0.95}],
                    "winner_index": 1,
                    "merged_takeaways": ["Apply RAII patterns", "Eliminate special cases"],
                    "winning_review_text": "Systematic Linus analysis complete"
                }),
                "# Linus Code Review\n\nSystematic analysis complete with taste assessment."  # Synthesizer
            ]
            
            # Process with Linus mode
            language = detect_language_from_extension(".cpp")
            chunks = chunk_code_by_lines(sample_cpp_code, max_lines=100)
            chunk_content = add_line_numbers_preserve(chunks[0][1], start_line=1)
            
            # Step 1: Linus worker
            worker_result = await run_worker_agent(
                mock_llm,
                language=language,
                file_path="test.cpp",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers=chunk_content,
                linus_mode=True
            )
            
            assert worker_result == sample_linus_response
            assert worker_result["taste_score"] in ["good", "so-so", "trash"]
            assert "linus_analysis" in worker_result
            
            # Step 2: Supervisor with Linus results
            supervisor_input = format_reviews_for_supervisor([worker_result])
            supervisor_result = await run_supervisor_agent(mock_llm, reviews_text_block=supervisor_input)
            
            assert supervisor_result["winner_index"] == 1
            assert "systematic" in supervisor_result["analysis"].lower()
            
            # Step 3: Synthesizer
            summary = {
                "file": "test.cpp",
                "chunk_index": 1,
                "total_chunks": 1,
                "winner_index": 1,
                "scores": supervisor_result["scores"],
                "winning_review_text": supervisor_result["winning_review_text"]
            }
            
            final_markdown = await run_synthesizer_agent(
                mock_llm,
                chunk_summaries_jsonl=json.dumps(summary)
            )
            
            assert "Linus" in final_markdown
            assert "systematic" in final_markdown.lower()
    
    def test_linus_output_structure_validation(self, sample_linus_response):
        """Test that Linus mode produces correctly structured output."""
        # Validate all required Linus fields are present
        required_fields = [
            "pre_analysis", "taste_score", "fatal_problems", 
            "key_insights", "linus_analysis", "improvement_direction"
        ]
        
        for field in required_fields:
            assert field in sample_linus_response, f"Missing required Linus field: {field}"
        
        # Validate pre_analysis structure
        pre_analysis = sample_linus_response["pre_analysis"]
        assert "real_problem" in pre_analysis
        assert "simpler_way" in pre_analysis  
        assert "breaks_compatibility" in pre_analysis
        
        # Validate linus_analysis structure (5-level framework)
        linus_analysis = sample_linus_response["linus_analysis"]
        analysis_levels = [
            "data_structure_analysis", "special_case_elimination",
            "complexity_review", "destructiveness_analysis", "practicality_verification"
        ]
        for level in analysis_levels:
            assert level in linus_analysis, f"Missing Linus analysis level: {level}"


class TestIterativeIntegration:
    """Test iterative review integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_iterative_pipeline_workflow(self, sample_complex_code, sample_worker_response, sample_iterative_supervisor_response, mock_llm):
        """Test complete iterative review workflow."""
        from agents import IterationController, IterationStrategy, run_iterative_worker_agent, run_iterative_supervisor_agent
        
        controller = IterationController(max_iterations=2, strategy=IterationStrategy.WORKER_POOL, retain_full_data=True)
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # Mock iterative responses - improving over iterations
            mock_retry.side_effect = [
                # Iteration 1: Workers
                json.dumps(sample_worker_response),
                json.dumps(sample_worker_response),
                # Iteration 1: Supervisor
                json.dumps(sample_iterative_supervisor_response),
                # Iteration 2: Workers (with iteration context)
                json.dumps({**sample_worker_response, "summary": "Improved analysis with iteration context"}),
                json.dumps({**sample_worker_response, "summary": "Enhanced findings from previous iteration"}),
                # Iteration 2: Supervisor
                json.dumps({
                    **sample_iterative_supervisor_response,
                    "iteration_comparison": {
                        "improvement_over_previous": "Significant improvement in depth and accuracy",
                        "convergence_indicators": "High similarity in core findings",
                        "quality_delta": 0.2
                    }
                })
            ]
            
            language = detect_language_from_extension(".py")
            chunks = chunk_code_by_lines(sample_complex_code, max_lines=100)
            chunk_content = add_line_numbers_preserve(chunks[0][1], start_line=1)
            
            # Simulate iterative review
            all_iterations_workers = []
            all_iterations_supervisors = []
            
            iteration_num = 1
            while controller.should_continue_iterating():
                # Get iteration context
                iteration_context = controller.get_context_for_iteration(iteration_num)
                
                # Run workers with iteration context
                worker_results = []
                for _ in range(2):  # 2 workers
                    result = await run_iterative_worker_agent(
                        mock_llm,
                        language=language,
                        file_path="complex.py",
                        chunk_index=1,
                        total_chunks=1,
                        code_with_line_numbers=chunk_content,
                        iteration_context=iteration_context
                    )
                    worker_results.append(result)
                
                # Run supervisor
                supervisor_input = format_reviews_for_supervisor(worker_results)
                supervisor_result = await run_iterative_supervisor_agent(
                    mock_llm,
                    reviews_text_block=supervisor_input,
                    iteration_context=iteration_context
                )
                
                # Record iteration
                controller.record_iteration(worker_results, supervisor_result, {})
                all_iterations_workers.append(worker_results)
                all_iterations_supervisors.append(supervisor_result)
                
                iteration_num += 1
            
            # Verify iteration progression
            assert len(all_iterations_workers) == 2
            assert len(all_iterations_supervisors) == 2
            assert controller.current_iteration == 2
            
            # Verify iteration context was used
            final_supervisor = all_iterations_supervisors[-1]
            assert "iteration_comparison" in final_supervisor
            assert "improvement_over_previous" in final_supervisor["iteration_comparison"]


class TestCombinedLinusIterative:
    """Test combining Linus mode with iterative review."""
    
    @pytest.mark.asyncio
    async def test_linus_mode_with_iterations(self, sample_cpp_code, sample_linus_response, mock_llm):
        """Test Linus mode combined with iterative review."""
        from agents import IterationController, IterationStrategy, run_iterative_worker_agent, run_iterative_supervisor_agent
        
        controller = IterationController(max_iterations=2, strategy=IterationStrategy.FEEDBACK_DRIVEN, retain_full_data=True)
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            # Mock Linus responses across iterations
            iteration_1_linus = sample_linus_response.copy()
            iteration_2_linus = {
                **sample_linus_response,
                "taste_score": "good",  # Improved from "so-so"
                "improvement_direction": [
                    "Original suggestions confirmed",
                    "Additional: Consider move semantics for performance"
                ]
            }
            
            mock_retry.side_effect = [
                # Iteration 1
                json.dumps(iteration_1_linus),  # Linus worker
                json.dumps({  # Iterative supervisor with Linus input
                    "analysis": "Linus systematic analysis provides solid foundation",
                    "scores": [{"accuracy": 0.9, "completeness": 0.85, "clarity": 0.9, "insightfulness": 0.9}],
                    "winner_index": 1,
                    "feedback_for_next_iteration": "Build on the data structure insights, focus on performance implications"
                }),
                # Iteration 2
                json.dumps(iteration_2_linus),  # Enhanced Linus worker with feedback
                json.dumps({  # Final iterative supervisor
                    "analysis": "Excellent progression in Linus analysis quality",
                    "scores": [{"accuracy": 0.95, "completeness": 0.9, "clarity": 0.9, "insightfulness": 0.95}],
                    "winner_index": 1,
                    "iteration_comparison": {
                        "improvement_over_previous": "Deeper analysis with performance considerations",
                        "quality_delta": 0.1
                    },
                    "feedback_for_next_iteration": "Analysis is comprehensive"
                })
            ]
            
            language = detect_language_from_extension(".cpp")
            chunks = chunk_code_by_lines(sample_cpp_code, max_lines=100)
            chunk_content = add_line_numbers_preserve(chunks[0][1], start_line=1)
            
            all_workers = []
            all_supervisors = []
            
            # Run iterations with Linus mode
            iteration_num = 1
            while controller.should_continue_iterating():
                context = controller.get_context_for_iteration(iteration_num)
                
                # Linus worker with iteration context
                worker_result = await run_iterative_worker_agent(
                    mock_llm,
                    language=language,
                    file_path="test.cpp",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers=chunk_content,
                    iteration_context=context,
                    linus_mode=True  # KEY: Linus mode with iterations
                )
                
                # Iterative supervisor handling Linus output
                supervisor_result = await run_iterative_supervisor_agent(
                    mock_llm,
                    reviews_text_block=json.dumps(worker_result),
                    iteration_context=context
                )
                
                controller.record_iteration([worker_result], supervisor_result, {})
                all_workers.append([worker_result])
                all_supervisors.append(supervisor_result)
                
                iteration_num += 1
            
            # Verify combined functionality
            assert len(all_workers) == 2
            assert len(all_supervisors) == 2
            
            # First iteration should have basic Linus analysis
            first_worker = all_workers[0][0]
            assert first_worker["taste_score"] == "so-so"
            assert "linus_analysis" in first_worker
            
            # Second iteration should show improvement
            second_worker = all_workers[1][0] 
            assert second_worker["taste_score"] == "good"  # Improved
            assert "move semantics" in str(second_worker["improvement_direction"])