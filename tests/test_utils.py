import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import (
    add_line_numbers_preserve,
    chunk_code_by_lines,
    detect_language_from_extension,
    extract_json_from_text,
    safe_filename_from_path,
    content_hash,
    list_source_files,
    load_models_config,
    save_json,
    save_text,
    count_tokens_text,
    get_model_name
)

class TestAddLineNumbers:
    def test_basic_numbering(self):
        code = "line1\nline2\nline3"
        result = add_line_numbers_preserve(code)
        expected = "L0001: line1\nL0002: line2\nL0003: line3"
        assert result == expected
    
    def test_custom_start_line(self):
        code = "line1\nline2"
        result = add_line_numbers_preserve(code, start_line=10)
        expected = "L0010: line1\nL0011: line2"
        assert result == expected
    
    def test_empty_code(self):
        result = add_line_numbers_preserve("")
        assert result == ""

class TestChunkCodeByLines:
    def test_small_code_no_chunking(self):
        code = "line1\nline2\nline3"
        chunks = chunk_code_by_lines(code, max_lines=10)
        assert len(chunks) == 1
        assert chunks[0] == (1, code)
    
    def test_large_code_chunking(self):
        lines = [f"line{i}" for i in range(1, 6)]
        code = "\n".join(lines)
        chunks = chunk_code_by_lines(code, max_lines=2)
        
        assert len(chunks) == 3
        assert chunks[0] == (1, "line1\nline2")
        assert chunks[1] == (3, "line3\nline4")
        assert chunks[2] == (5, "line5")
    
    def test_exact_chunk_boundary(self):
        code = "line1\nline2\nline3\nline4"
        chunks = chunk_code_by_lines(code, max_lines=2)
        assert len(chunks) == 2
        assert chunks[0] == (1, "line1\nline2")
        assert chunks[1] == (3, "line3\nline4")

class TestLanguageDetection:
    def test_cpp_extensions(self):
        assert detect_language_from_extension(".cpp") == "cpp"
        assert detect_language_from_extension(".hpp") == "cpp"
        assert detect_language_from_extension(".cxx") == "cpp"
        assert detect_language_from_extension(".h") == "cpp"
        assert detect_language_from_extension(".cc") == "cpp"
    
    def test_python_extensions(self):
        assert detect_language_from_extension(".py") == "python"
    
    def test_javascript_extensions(self):
        assert detect_language_from_extension(".js") == "javascript"
        assert detect_language_from_extension(".ts") == "javascript"
        assert detect_language_from_extension(".jsx") == "javascript"
        assert detect_language_from_extension(".tsx") == "javascript"
    
    def test_java_extensions(self):
        assert detect_language_from_extension(".java") == "java"
    
    def test_unknown_extension(self):
        assert detect_language_from_extension(".xyz") == "generic"
        assert detect_language_from_extension("") == "generic"

class TestJsonExtraction:
    def test_simple_json(self):
        text = '{"key": "value", "number": 42}'
        result = extract_json_from_text(text)
        assert result == text
    
    def test_json_with_prose(self):
        text = 'Here is the analysis: {"summary": "good", "issues": []} and that is all.'
        result = extract_json_from_text(text)
        assert result == '{"summary": "good", "issues": []}'
    
    def test_json_in_code_fence(self):
        text = '''The result is:
```json
{"status": "success", "data": {"count": 5}}
```
End of response.'''
        result = extract_json_from_text(text)
        assert result == '{"status": "success", "data": {"count": 5}}'
    
    def test_json_in_plain_fence(self):
        text = '''```
{"simple": "object"}
```'''
        result = extract_json_from_text(text)
        assert result == '{"simple": "object"}'
    
    def test_no_json_found(self):
        text = "No JSON here at all"
        result = extract_json_from_text(text)
        assert result == text

class TestFilenameUtils:
    def test_safe_filename_from_path(self):
        assert safe_filename_from_path("src/utils.py") == "src__utils.py"
        assert safe_filename_from_path("/absolute/path/file.cpp") == "__absolute__path__file.cpp"
        assert safe_filename_from_path("simple.py") == "simple.py"
    
    def test_content_hash(self):
        hash1 = content_hash("file1", "content1")
        hash2 = content_hash("file1", "content1")
        hash3 = content_hash("file1", "content2")
        
        assert hash1 == hash2  # Same inputs = same hash
        assert hash1 != hash3  # Different inputs = different hash
        assert len(hash1) == 16  # Expected length

class TestFileOperations:
    def test_list_source_files(self, temp_dir):
        # Create test files
        (Path(temp_dir) / "test.py").write_text("python code")
        (Path(temp_dir) / "test.cpp").write_text("cpp code")
        (Path(temp_dir) / "test.txt").write_text("text file")
        (Path(temp_dir) / "subdir").mkdir()
        (Path(temp_dir) / "subdir" / "nested.py").write_text("nested python")
        
        files = list_source_files(temp_dir, [".py", ".cpp"])
        
        assert len(files) == 3
        assert any("test.py" in f for f in files)
        assert any("test.cpp" in f for f in files)
        assert any("nested.py" in f for f in files)
        assert not any("test.txt" in f for f in files)
    
    def test_save_and_load_json(self, temp_dir):
        data = {"test": "data", "number": 42}
        file_path = os.path.join(temp_dir, "test.json")
        
        save_json(file_path, data)
        
        with open(file_path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == data
    
    def test_save_text(self, temp_dir):
        text = "Hello\nWorld\nTest"
        file_path = os.path.join(temp_dir, "test.txt")
        
        save_text(file_path, text)
        
        with open(file_path, 'r') as f:
            loaded = f.read()
        
        assert loaded == text
    
    def test_load_models_config_missing_file(self):
        result = load_models_config("nonexistent.yaml")
        assert result == {}
    
    def test_load_models_config_valid_file(self, temp_dir):
        config_data = {
            "workers": [{"model": "gpt-4o", "temperature": 0.3}],
            "supervisor": {"model": "gpt-4o", "temperature": 0.1}
        }
        
        import yaml
        config_path = os.path.join(temp_dir, "test_models.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        result = load_models_config(config_path)
        assert result == config_data

class TestTokenCounting:
    def test_count_tokens_text_without_tiktoken(self):
        """Test token counting when tiktoken is not available."""
        with patch('utils._get_encoding_for_model') as mock_get_encoding:
            mock_encoding = MagicMock()
            mock_encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            mock_get_encoding.return_value = mock_encoding
            
            result = count_tokens_text("gpt-4o", "test text")
            assert result == 5
    
    def test_get_model_name_caching(self):
        """Test that model name extraction uses caching."""
        from utils import _get_encoding_for_model
        
        # Test that the LRU cache decorator is working
        assert hasattr(_get_encoding_for_model, '__wrapped__')
        assert _get_encoding_for_model.cache_info is not None

class TestGetModelName:
    def test_get_model_name_with_model_attr(self):
        llm = MagicMock()
        llm.model = "gpt-4o"
        
        result = get_model_name(llm)
        assert result == "gpt-4o"
    
    def test_get_model_name_with_model_name_attr(self):
        llm = MagicMock()
        llm.model = None
        llm.model_name = "gpt-4o-mini"
        
        result = get_model_name(llm)
        assert result == "gpt-4o-mini"
    
    def test_get_model_name_fallback(self):
        llm = MagicMock()
        llm.model = None
        llm.model_name = None
        llm.model_id = None
        
        result = get_model_name(llm, default="fallback-model")
        assert result == "fallback-model"
    
    def test_get_model_name_from_kwargs(self):
        llm = MagicMock()
        llm.model = None
        llm.model_name = None
        llm.kwargs = {"model": "gpt-4o-from-kwargs"}
        
        result = get_model_name(llm)
        assert result == "gpt-4o-from-kwargs"