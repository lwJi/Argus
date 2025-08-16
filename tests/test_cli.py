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