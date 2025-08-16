import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock
from langchain_openai import ChatOpenAI

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''def hello_world():
    print("Hello, World!")
    return True

class Calculator:
    def add(self, a, b):
        return a + b
    
    def divide(self, a, b):
        return a / b  # Potential division by zero
'''

@pytest.fixture
def sample_cpp_code():
    """Sample C++ code for testing."""
    return '''#include <iostream>
#include <vector>

class Example {
private:
    int* ptr;
public:
    Example() {
        ptr = new int(42);
    }
    
    ~Example() {
        delete ptr;
    }
    
    void process(std::vector<int> data) {
        for (int i = 0; i <= data.size(); ++i) {
            std::cout << data[i] << std::endl;
        }
    }
};
'''

@pytest.fixture
def mock_llm():
    """Mock ChatOpenAI instance for testing."""
    llm = MagicMock(spec=ChatOpenAI)
    llm.model = "gpt-4o-mini"
    llm.temperature = 0.3
    return llm

@pytest.fixture
def sample_models_config():
    """Sample models configuration."""
    return {
        "workers": [
            {"model": "gpt-4o-mini", "temperature": 0.3},
            {"model": "gpt-4o-mini", "temperature": 0.7}
        ],
        "supervisor": {
            "model": "gpt-4o", 
            "temperature": 0.1
        }
    }

@pytest.fixture
def sample_worker_response():
    """Sample worker agent JSON response."""
    return {
        "summary": "Code has potential bugs and style issues",
        "findings": [
            {
                "type": "bug",
                "title": "Potential division by zero",
                "severity": "high",
                "lines": [8],
                "snippet": "return a / b",
                "explanation": "No check for zero divisor",
                "suggestion": "Add zero check before division",
                "diff": ""
            }
        ],
        "counts": {"bug": 1, "performance": 0, "style": 0, "maintainability": 0}
    }

@pytest.fixture
def sample_supervisor_response():
    """Sample supervisor agent JSON response."""
    return {
        "analysis": "Review 1 provides better error detection",
        "scores": [
            {
                "review_index": 1,
                "accuracy": 0.9,
                "completeness": 0.8,
                "clarity": 0.9,
                "insightfulness": 0.7,
                "notes": "Good bug detection"
            }
        ],
        "winner_index": 1,
        "merged_takeaways": ["Add input validation"],
        "winning_review_text": "Sample winning review"
    }