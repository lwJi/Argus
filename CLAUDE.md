# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Running Argus
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

### Development Commands
```bash
# Install dependencies
pip install -r src/requirements.txt

# Run tests
pytest

# Run specific test
pytest tests/test_agents.py::TestClassName::test_method_name

# Run with coverage
pytest --cov=src tests/
```

### Environment Setup
Argus requires `OPENAI_API_KEY`. It will look for environment variables in:
1. Project `.env` file (nearest to current directory)
2. `~/.argus.env`
3. `~/.env` 
4. `~/.config/argus/env`

## Architecture Overview

### Multi-Agent Pipeline Pattern
Argus implements a three-tier agent architecture for code review:

```
File → Chunks → Workers → Supervisor → Synthesizer → Final Report
             (parallel)   (consensus)  (synthesis)
```

**Core Components:**
- **Worker Agents** (`src/agents.py:84-128`): Multiple AI models analyze code chunks independently using language-specific prompts
- **Supervisor Agent** (`src/agents.py:130-154`): Evaluates worker outputs and selects the best review based on accuracy, completeness, clarity, and insightfulness  
- **Synthesizer Agent** (`src/agents.py:156-166`): Combines winning chunk reviews into final markdown report

### Key Data Flow
1. **File Processing** (`src/Argus:178-255`): Files are chunked by lines (default 400 lines/chunk) to respect LLM context limits
2. **Worker Execution**: Each chunk processed by N worker agents in parallel with different temperatures for diversity
3. **Supervisor Decision**: Best worker review selected using multi-dimensional scoring
4. **Final Synthesis**: Chunk winners merged into comprehensive file-level review

### Review Modes
The system supports multiple review approaches via Strategy pattern (`src/prompts.py`):

#### Standard Language-Specific Review
- **C++ prompts**: Focus on RAII, const-correctness, memory safety, C++ Core Guidelines
- **Python prompts**: Emphasize PEP 8/20, type hints, performance patterns
- **Generic prompts**: General code quality patterns for other languages

#### Linus Review Mode (`--linus-mode`)
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

### Configuration System
- **Model Configuration** (`models.yaml`): Define worker models, temperatures, and supervisor settings
- **Linus Mode Optimization**: Higher-capability models recommended for complex analytical framework
- **Extensible**: Add new languages by creating new prompt templates in `src/prompts.py`
- **Token Management**: Built-in preflight estimation prevents expensive API surprises

### Error Handling Patterns
- **Resilient Processing**: Pipeline continues if individual workers fail
- **JSON Repair**: Uses AI-powered repair for malformed model outputs (`REPAIR_JSON_PROMPT`)
- **Exponential Backoff**: Automatic retry with jitter for API reliability

### Output Structure
Each review produces:
- **Structured JSON**: Complete audit trail with worker outputs, supervisor decisions, metadata
- **Human-readable Markdown**: Final synthesized report for developers
- **Categorized Findings**: Issues classified as bug/performance/style/maintainability with severity levels

**Linus Mode Additional Fields:**
- `pre_analysis`: Three fundamental questions (real problem? simpler way? compatibility?)
- `taste_score`: Technical quality assessment (good/so-so/trash)
- `fatal_problems`: Critical issues with direct technical criticism
- `key_insights`: Data structure analysis, complexity removal opportunities, risk assessment
- `linus_analysis`: Complete 5-level systematic breakdown

## Key Files

- `src/Argus`: Main CLI orchestrator with async processing pipeline
- `src/agents.py`: Core agent implementations and retry/repair logic  
- `src/prompts.py`: Language-specific prompt templates and JSON schemas
- `src/utils.py`: File processing, chunking, token estimation utilities
- `models.yaml`: LLM configuration (models, temperatures per agent role)

## Development Notes

### Adding New Languages
1. Create new prompt template in `src/prompts.py` following existing patterns
2. Add language detection in `utils.py:detect_language_from_extension()`
3. Update `agents.py:_pick_worker_prompt()` to return new prompt

### Modifying JSON Schemas
Worker and Supervisor schemas are defined in `src/prompts.py`. Changes require:
1. Update schema constants (`JSON_WORKER_SCHEMA`, `JSON_SUPERVISOR_SCHEMA`, `JSON_LINUS_SCHEMA`)
2. Ensure prompt templates reference correct schema format
3. Test with various AI models as schema parsing can be brittle

### Extending Review Modes
To add new systematic review approaches:
1. Create new prompt template in `src/prompts.py` following `WORKER_PROMPT_LINUS` pattern
2. Add corresponding JSON schema for structured output format
3. Update `agents.py:_pick_worker_prompt()` to include new mode selection
4. Add CLI flag in `src/Argus` for enabling the new mode
5. Update configuration examples in `models.yaml`

### Token Management
The system includes sophisticated token estimation (`src/utils.py:128-151`) using tiktoken. Preflight calculations help avoid expensive API overruns on large codebases.