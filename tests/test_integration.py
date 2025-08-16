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