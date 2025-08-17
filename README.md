# Argus
Intelligent Code Review, Powered by AI

Argus is an advanced AI-powered code review tool that uses a multi-agent pipeline to provide comprehensive, insightful analysis of your codebase. Built with a sophisticated three-tier architecture, Argus delivers professional-grade code reviews with specialized modes including the unique "Linus Mode" inspired by Linux kernel development practices.

## Features

- **Multi-Agent Architecture**: Multiple AI models analyze code independently, with a supervisor selecting the best insights
- **Language-Specific Analysis**: Tailored prompts for C++, Python, and other languages
- **Linus Review Mode**: Systematic analysis framework inspired by Linux kernel development practices
- **Comprehensive Output**: Both structured JSON and human-readable Markdown reports
- **Scalable Processing**: Intelligent chunking handles large codebases efficiently
- **Token Management**: Built-in estimation prevents expensive API overruns

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r src/requirements.txt
```

### Environment Setup

Argus requires `OPENAI_API_KEY`. It will look for environment variables in:
1. Project `.env` file (nearest to current directory)
2. `~/.argus.env`
3. `~/.env` 
4. `~/.config/argus/env`

### Basic Usage

```bash
# Basic usage - review all C++/Python files in a directory
python src/Argus <directory>

# With custom extensions and output directory
python src/Argus <directory> --extensions .cpp .py .js --save-dir reviews

# Skip token estimation prompt (for automation)
python src/Argus <directory> --skip-preflight

# Use Linus Torvalds review style with systematic analysis
python src/Argus <directory> --linus-mode

# Adjust chunking for large files
python src/Argus <directory> --chunk-size 300 --chunk-threshold 400
```

## Architecture Overview

### Multi-Agent Pipeline Pattern

Argus implements a sophisticated three-tier agent architecture:

```
File → Chunks → Workers → Supervisor → Synthesizer → Final Report
             (parallel)   (consensus)  (synthesis)
```

**Core Components:**
- **Worker Agents**: Multiple AI models analyze code chunks independently using language-specific prompts
- **Supervisor Agent**: Evaluates worker outputs and selects the best review based on accuracy, completeness, clarity, and insightfulness  
- **Synthesizer Agent**: Combines winning chunk reviews into final markdown report

### Key Data Flow

1. **File Processing**: Files are chunked by lines (default 400 lines/chunk) to respect LLM context limits
2. **Worker Execution**: Each chunk processed by multiple worker agents in parallel with different temperatures for diversity
3. **Supervisor Decision**: Best worker review selected using multi-dimensional scoring
4. **Final Synthesis**: Chunk winners merged into comprehensive file-level review

## Review Modes

### Standard Language-Specific Review
- **C++ prompts**: Focus on RAII, const-correctness, memory safety, C++ Core Guidelines
- **Python prompts**: Emphasize PEP 8/20, type hints, performance patterns
- **Generic prompts**: General code quality patterns for other languages

### Linus Review Mode (`--linus-mode`)

Applies Linus Torvalds' systematic code analysis framework with 30+ years of kernel experience:

**Core Philosophy:**
- **"Good taste"**: Eliminate special cases, make the normal case handle everything
- **"Never break userspace"**: Backward compatibility is sacred
- **Pragmatism over theory**: Solve real problems, reject overengineering
- **Simplicity obsession**: If you need >3 levels of indentation, redesign it

**5-Level Analysis Framework:**
1. **Pre-analysis**: Real problem? Simpler way? Backward compatibility impact?
2. **Data structure analysis**: "Good programmers worry about data structures"
3. **Special case identification**: Eliminate if/else branches through better design
4. **Complexity review**: Can concepts be halved, then halved again?
5. **Practicality verification**: Does this exist in production? How many affected?

**Structured Output:**
- **Taste Score**: Good/So-so/Trash technical assessment
- **Fatal Problems**: Critical issues with direct technical criticism  
- **Key Insights**: Data structure, complexity, and risk analysis
- **Improvement Direction**: Specific actionable recommendations

## Configuration

### Model Configuration
- **models.yaml**: Define worker models, temperatures, and supervisor settings
- **Linus Mode Optimization**: Higher-capability models recommended for complex analytical framework
- **Extensible**: Add new languages by creating new prompt templates

### Token Management
Built-in preflight estimation prevents expensive API surprises on large codebases.

## Output Structure

Each review produces:

### Standard Output
- **Structured JSON**: Complete audit trail with worker outputs, supervisor decisions, metadata
- **Human-readable Markdown**: Final synthesized report for developers
- **Categorized Findings**: Issues classified as bug/performance/style/maintainability with severity levels

### Linus Mode Additional Fields
- `pre_analysis`: Three fundamental questions (real problem? simpler way? compatibility?)
- `taste_score`: Technical quality assessment (good/so-so/trash)
- `fatal_problems`: Critical issues with direct technical criticism
- `key_insights`: Data structure analysis, complexity removal opportunities, risk assessment
- `linus_analysis`: Complete 5-level systematic breakdown

## Error Handling

- **Resilient Processing**: Pipeline continues if individual workers fail
- **JSON Repair**: Uses AI-powered repair for malformed model outputs
- **Exponential Backoff**: Automatic retry with jitter for API reliability

## Development

### Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_agents.py::TestClassName::test_method_name

# Run with coverage
pytest --cov=src tests/
```

### Key Files

- `src/Argus`: Main CLI orchestrator with async processing pipeline
- `src/agents.py`: Core agent implementations and retry/repair logic  
- `src/prompts.py`: Language-specific prompt templates and JSON schemas
- `src/utils.py`: File processing, chunking, token estimation utilities
- `models.yaml`: LLM configuration (models, temperatures per agent role)

## Contributing

Argus is designed to be extensible. You can:
- Add new programming languages by creating prompt templates
- Implement new review modes with custom analysis frameworks
- Extend the configuration system for different AI models
- Contribute to the core pipeline architecture

## License

[License information to be added]