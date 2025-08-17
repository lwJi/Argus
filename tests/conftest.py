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

@pytest.fixture
def sample_linus_response():
    """Sample Linus mode worker agent JSON response."""
    return {
        "pre_analysis": {
            "real_problem": "Yes, potential memory safety issues",
            "simpler_way": "Use smart pointers instead of raw pointers",
            "breaks_compatibility": "No, this is internal implementation"
        },
        "taste_score": "so-so",
        "fatal_problems": [
            "Raw pointer without proper RAII management",
            "Potential buffer overflow in loop bounds"
        ],
        "key_insights": {
            "data_structure": "Vector iteration with manual bounds checking",
            "complexity_removal": "Replace manual loop with range-based for",
            "risk_assessment": "High - memory safety violations likely"
        },
        "core_judgment": "worth_doing",
        "linus_analysis": {
            "data_structure_analysis": "The core data is a vector, but we're accessing it with manual indexing",
            "special_case_elimination": "The <= instead of < creates a special boundary case",
            "complexity_review": "This 5-line loop can become a 1-line range-based for",
            "destructiveness_analysis": "Changes are internal, no API breakage",
            "practicality_verification": "Real bug - will crash in production"
        },
        "improvement_direction": [
            "Replace manual loop with range-based for iteration",
            "Use smart pointers for automatic memory management",
            "Add const-correctness to prevent accidental modification"
        ]
    }

@pytest.fixture
def sample_iterative_supervisor_response():
    """Sample iterative supervisor agent JSON response."""
    return {
        "analysis": "Review quality improved from iteration 1, convergence detected",
        "scores": [
            {
                "review_index": 1,
                "accuracy": 0.95,
                "completeness": 0.9,
                "clarity": 0.9,
                "insightfulness": 0.8,
                "notes": "Excellent bug detection with context from previous iteration"
            }
        ],
        "winner_index": 1,
        "merged_takeaways": ["Use RAII patterns", "Add bounds checking"],
        "winning_review_text": "Comprehensive review building on previous findings",
        "iteration_comparison": {
            "improvement_over_previous": "Better context understanding and more specific suggestions",
            "convergence_indicators": "Similar findings with higher confidence",
            "quality_delta": 0.15
        },
        "feedback_for_next_iteration": "Focus on performance implications of suggested changes"
    }

@pytest.fixture
def sample_iteration_metadata():
    """Sample iteration metadata for testing."""
    return {
        "total_iterations": 3,
        "convergence_achieved": True,
        "strategy_used": "worker_pool",
        "improvement_trajectory": {
            "trend": "improving",
            "trajectories": {
                "quality_scores": [0.7, 0.82, 0.85],
                "finding_counts": [3, 4, 4],
                "critical_counts": [1, 2, 2]
            },
            "final_quality": 0.85
        },
        "best_iteration": {
            "iteration": 3,
            "reason": "highest_quality_score"
        }
    }

@pytest.fixture
def sample_complex_code():
    """Sample complex code that would benefit from iteration."""
    return '''class NetworkManager:
    def __init__(self):
        self.connections = []
        self.timeout = None
        
    def connect(self, host, port):
        # Multiple potential issues for iterative discovery
        if host == None:  # Should use 'is None'
            return False
        if port < 0 or port > 65535:
            raise Exception("Invalid port")  # Should be more specific
        
        conn = self.create_connection(host, port)
        self.connections.append(conn)
        return True
        
    def create_connection(self, host, port):
        # Resource leak potential
        import socket
        s = socket.socket()
        s.settimeout(self.timeout or 30)
        s.connect((host, port))  # No error handling
        return s
        
    def disconnect_all(self):
        # Potential double-close issue
        for conn in self.connections:
            conn.close()
        del self.connections[:]  # Pythonic: self.connections.clear()
'''