# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Argus is an intelligent code review system powered by AI that provides automated code analysis for various programming languages including Python, C++, and JavaScript.

## Architecture

The system uses a multi-agent approach with three main components:

1. **Worker Agents** (`agents.py:62-126`) - Perform initial code analysis on code chunks
   - Language-specific prompts for Python and C++ with fallback to generic
   - Return structured JSON analysis following a defined schema
   - Handle retry logic with exponential backoff

2. **Supervisor Agent** (`agents.py:128-151`) - Evaluates multiple worker reviews
   - Compares worker outputs for accuracy, completeness, clarity, and insightfulness
   - Selects the best review and provides consolidated feedback
   - Returns structured analysis with scoring

3. **Synthesizer Agent** (`agents.py:153-163`) - Creates final file-level review
   - Merges chunk-level analyses into a comprehensive markdown report
   - Provides executive summary, categorized findings, and action items

## Key Components

### Core Files
- `src/agents.py` - Main agent orchestration and execution logic
- `src/prompts.py` - Prompt templates for different programming languages and agent types
- `src/utils.py` - Utility functions for file handling, tokenization, and configuration
- `src/models.example.yaml` - Model configuration template

### Dependencies
The project uses LangChain for LLM orchestration with these key dependencies:
- `langchain==0.2.1` and `langchain-openai==0.1.7` for LLM integration
- `tenacity==8.2.3` for retry mechanisms
- `rich==13.7.1` for terminal output formatting
- `tiktoken==0.7.0` for token counting
- `PyYAML==6.0.2` for configuration management

## Development Commands

### Installation
```bash
pip install -r src/requirements.txt
```

### Configuration
Copy `src/models.example.yaml` to `src/models.yaml` and configure your model settings:
- Worker models: Multiple models can be configured with different temperatures
- Supervisor model: Single model for review evaluation

## Code Analysis Flow

1. **Code Chunking** (`utils.py:30-40`) - Split large files into manageable chunks (400 lines default)
2. **Worker Analysis** - Each chunk analyzed by multiple worker agents in parallel
3. **Supervision** - Best worker review selected for each chunk
4. **Synthesis** - Final markdown report generated from winning chunk reviews

## Language Support

- **Python**: Specialized prompts focusing on PEP 8/20, type hints, performance
- **C++**: C++17/20 focus with Core Guidelines, RAII, const-correctness
- **Generic**: Fallback for JavaScript, Java, and other languages

## Key Utilities

- `utils.py:22-27` - Add line numbers to code for precise issue reporting
- `utils.py:91-105` - Extract JSON from LLM responses with fallback parsing
- `utils.py:128-134` - Token counting for cost estimation and chunking decisions
- `utils.py:82-88` - Source file discovery with extension filtering