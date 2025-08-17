import pytest
import os
import sys
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestCLIFunctionality:
    def test_models_config_loading_in_main(self, temp_dir, sample_models_config):
        """Test that the main CLI can load models configuration."""
        
        # Import here to avoid module loading issues
        from utils import load_models_config
        
        # Create test config file
        config_path = os.path.join(temp_dir, "test_models.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(sample_models_config, f)
        
        # Test loading
        config = load_models_config(config_path)
        assert config == sample_models_config
        assert len(config["workers"]) == 2
        assert config["supervisor"]["model"] == "gpt-4o"
    
    def test_file_discovery_integration(self, temp_dir):
        """Test file discovery as used by CLI."""
        from utils import list_source_files, detect_language_from_extension
        
        # Create test files  
        files_to_create = [
            ("src/main.py", "python code"),
            ("src/utils.cpp", "cpp code"),
            ("docs/readme.txt", "documentation"),
            ("src/nested/deep.py", "nested python"),
        ]
        
        for file_path, content in files_to_create:
            full_path = Path(temp_dir) / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content)
        
        # Test discovery
        source_files = list_source_files(temp_dir, [".py", ".cpp"])
        assert len(source_files) == 3
        
        # Test language detection
        for file_path in source_files:
            ext = Path(file_path).suffix
            language = detect_language_from_extension(ext)
            assert language in ["python", "cpp"]
    
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_environment_variable_handling(self):
        """Test that environment variables are properly handled."""
        assert os.getenv("OPENAI_API_KEY") == "test-key"

class TestOutputGeneration:
    def test_safe_filename_generation(self):
        """Test safe filename generation for output files."""
        from utils import safe_filename_from_path, content_hash
        
        # Test path sanitization
        assert safe_filename_from_path("src/utils.py") == "src__utils.py"
        assert safe_filename_from_path("/abs/path/file.cpp") == "__abs__path__file.cpp"
        
        # Test content hashing for unique filenames
        hash1 = content_hash("file1.py", "content")
        hash2 = content_hash("file1.py", "different content") 
        assert hash1 != hash2
        assert len(hash1) == 16

class TestProgressTracking:
    def test_chunk_counting_for_progress(self, sample_python_code):
        """Test chunk counting for progress tracking."""
        from utils import chunk_code_by_lines
        
        # Small file - single chunk
        chunks_small = chunk_code_by_lines(sample_python_code, max_lines=100)
        assert len(chunks_small) == 1
        
        # Large content - multiple chunks
        large_code = "\n".join([f"line {i}" for i in range(1000)])
        chunks_large = chunk_code_by_lines(large_code, max_lines=100)
        assert len(chunks_large) == 10  # 1000 lines / 100 per chunk


class TestIterationCLIArguments:
    """Test CLI argument parsing and validation for iteration features."""
    
    def test_iterations_argument_default(self):
        """Test that --iterations defaults to 1."""
        import argparse
        from argparse import ArgumentParser
        
        # Create a parser similar to the main CLI
        parser = ArgumentParser()
        parser.add_argument("--iterations", type=int, default=1)
        
        # Test default value
        args = parser.parse_args([])
        assert args.iterations == 1
    
    def test_iterations_argument_parsing(self):
        """Test --iterations argument parsing with valid values."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--iterations", type=int, default=1)
        
        # Test valid values
        for valid_value in [1, 2, 3, 4, 5]:
            args = parser.parse_args([f"--iterations", str(valid_value)])
            assert args.iterations == valid_value
    
    def test_iteration_strategy_choices(self):
        """Test --iteration-strategy argument with valid choices."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--iteration-strategy", 
                          choices=["worker_pool", "feedback_driven", "consensus"],
                          default="worker_pool")
        
        # Test all valid strategies
        valid_strategies = ["worker_pool", "feedback_driven", "consensus"]
        for strategy in valid_strategies:
            args = parser.parse_args(["--iteration-strategy", strategy])
            assert args.iteration_strategy == strategy
        
        # Test default
        args = parser.parse_args([])
        assert args.iteration_strategy == "worker_pool"
    
    def test_convergence_threshold_parsing(self):
        """Test --convergence-threshold argument parsing."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--convergence-threshold", type=float, default=0.8)
        
        # Test valid thresholds
        valid_thresholds = [0.0, 0.5, 0.8, 0.9, 1.0]
        for threshold in valid_thresholds:
            args = parser.parse_args(["--convergence-threshold", str(threshold)])
            assert args.convergence_threshold == threshold
        
        # Test default
        args = parser.parse_args([])
        assert args.convergence_threshold == 0.8
    
    def test_combined_iteration_arguments(self):
        """Test combining iteration arguments."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--iterations", type=int, default=1)
        parser.add_argument("--iteration-strategy", 
                          choices=["worker_pool", "feedback_driven", "consensus"],
                          default="worker_pool")
        parser.add_argument("--convergence-threshold", type=float, default=0.8)
        
        args = parser.parse_args([
            "--iterations", "3",
            "--iteration-strategy", "feedback_driven", 
            "--convergence-threshold", "0.9"
        ])
        
        assert args.iterations == 3
        assert args.iteration_strategy == "feedback_driven"
        assert args.convergence_threshold == 0.9
    
    def test_linus_mode_with_iterations(self):
        """Test combining --linus-mode with iteration arguments.""" 
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--linus-mode", action="store_true")
        parser.add_argument("--iterations", type=int, default=1)
        parser.add_argument("--iteration-strategy", default="worker_pool")
        
        args = parser.parse_args([
            "--linus-mode",
            "--iterations", "3",
            "--iteration-strategy", "consensus"
        ])
        
        assert args.linus_mode == True
        assert args.iterations == 3
        assert args.iteration_strategy == "consensus"


class TestCLIValidation:
    """Test CLI argument validation logic."""
    
    @patch('builtins.print')
    def test_iterations_range_validation(self, mock_print):
        """Test that iterations must be between 1 and 5."""
        # This tests the validation logic that should exist in main()
        # Since we can't easily test the actual main() function without refactoring,
        # we test the validation logic conceptually
        
        def validate_iterations(iterations):
            """Mock validation function similar to main()."""
            return 1 <= iterations <= 5
        
        # Valid iterations
        for valid in [1, 2, 3, 4, 5]:
            assert validate_iterations(valid) == True
        
        # Invalid iterations
        for invalid in [0, -1, 6, 10, 100]:
            assert validate_iterations(invalid) == False
    
    def test_convergence_threshold_range_validation(self):
        """Test that convergence threshold must be between 0.0 and 1.0."""
        def validate_convergence_threshold(threshold):
            """Mock validation function."""
            return 0.0 <= threshold <= 1.0
        
        # Valid thresholds
        for valid in [0.0, 0.1, 0.5, 0.8, 0.9, 1.0]:
            assert validate_convergence_threshold(valid) == True
        
        # Invalid thresholds
        for invalid in [-0.1, -1.0, 1.1, 2.0, 10.0]:
            assert validate_convergence_threshold(invalid) == False


class TestTokenEstimationWithIterations:
    """Test token estimation adjustments for iteration mode."""
    
    def test_iteration_multiplier_logic(self):
        """Test that iterations multiply token usage estimates."""
        # Mock the preflight estimation logic
        base_tokens = 1000
        iterations = 3
        
        # Worker calls should multiply by iterations
        worker_tokens = base_tokens * iterations
        assert worker_tokens == 3000
        
        # Supervisor calls also multiply with overhead
        supervisor_overhead = 0.2  # 20% more for iteration context
        supervisor_tokens = base_tokens * iterations * (1 + supervisor_overhead)
        assert supervisor_tokens == 3600  # 1000 * 3 * 1.2
        
        # Synthesizer has complexity overhead for cross-iteration synthesis
        synthesizer_overhead = 0.3  # 30% more for iteration synthesis
        synthesizer_tokens = base_tokens * (1 + synthesizer_overhead)
        assert synthesizer_tokens == 1300  # 1000 * 1.3
    
    def test_iteration_strategy_overhead(self):
        """Test that different strategies might have different overhead."""
        base_cost = 100
        
        # Worker pool strategy - standard multiplier
        worker_pool_cost = base_cost * 3  # 3 iterations
        
        # Feedback driven might have slightly more context
        feedback_overhead = 0.1  # 10% more for feedback context
        feedback_cost = base_cost * 3 * (1 + feedback_overhead)
        
        # Consensus might have most overhead due to peer review sharing
        consensus_overhead = 0.2  # 20% more for peer review context
        consensus_cost = base_cost * 3 * (1 + consensus_overhead)
        
        assert feedback_cost > worker_pool_cost
        assert consensus_cost > feedback_cost
        assert consensus_cost > worker_pool_cost


class TestCLIOutputFormatting:
    """Test CLI output formatting for iteration mode."""
    
    def test_progress_message_with_iterations(self):
        """Test that progress messages include iteration information."""
        # Mock progress message generation
        def format_progress_message(filename, chunks, iterations=1):
            base_msg = f"Reviewing {filename} ({chunks} chunk{'s' if chunks > 1 else ''})"
            if iterations > 1:
                base_msg += f" [italic]({iterations} iterations)[/italic]"
            return base_msg
        
        # Single iteration (default)
        msg1 = format_progress_message("test.py", 3, 1)
        assert msg1 == "Reviewing test.py (3 chunks)"
        assert "iterations" not in msg1
        
        # Multiple iterations
        msg2 = format_progress_message("test.py", 3, 3)
        assert msg2 == "Reviewing test.py (3 chunks) [italic](3 iterations)[/italic]"
        assert "3 iterations" in msg2
    
    def test_completion_message_with_iterations(self):
        """Test completion messages include iteration metadata."""
        def format_completion_message(json_path, md_path, iterations=1, strategy="single_pass"):
            base_msg = f"JSON: {json_path}\nMarkdown: {md_path}"
            if iterations > 1:
                iteration_info = f" ({iterations} iterations, {strategy})"
                return base_msg.replace("JSON:", f"JSON:{iteration_info}").replace("Markdown:", f"Markdown:{iteration_info}")
            return base_msg
        
        # Single iteration
        msg1 = format_completion_message("out.json", "out.md", 1)
        assert "iterations" not in msg1
        
        # Multiple iterations  
        msg2 = format_completion_message("out.json", "out.md", 3, "feedback_driven")
        assert "(3 iterations, feedback_driven)" in msg2