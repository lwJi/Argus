# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

### Commands
```bash
# Basic usage
python src/Argus.py <directory>

# Development
pytest                                    # Run tests
pytest tests/test_agents.py::TestClass  # Run specific test
```

### Architecture Quick Reference

**Core Pipeline:**
```
File → Chunks → Workers → Supervisor → Synthesizer → Final Report
             (parallel)   (consensus)  (synthesis)
```

**Key Implementation Points:**
- **Worker Agents** (`src/agents.py:84-128`): Language-specific analysis with different temperatures
- **Supervisor Agent** (`src/agents.py:130-154`): Multi-dimensional scoring for best review selection
- **Synthesizer Agent** (`src/agents.py:156-166`): Combines winning reviews into markdown
- **File Processing** (`src/Argus.py:178-255`): Chunking logic (400 lines/chunk default)
- **Token Management** (`src/utils.py:128-151`): Uses tiktoken for preflight estimation

## Key Files & Functions

- `src/Argus.py`: Main CLI orchestrator with async processing pipeline
- `src/agents.py`: Core agent implementations and retry/repair logic  
  - `WorkerAgent._pick_worker_prompt()`: Language-specific prompt selection
  - `JSON_REPAIR_PROMPT`: AI-powered malformed JSON repair
- `src/prompts.py`: Language-specific prompt templates and JSON schemas
  - `JSON_WORKER_SCHEMA`, `JSON_SUPERVISOR_SCHEMA`, `JSON_LINUS_SCHEMA`: Output schemas
- `src/utils.py`: File processing, chunking, token estimation utilities
  - `detect_language_from_extension()`: File type detection
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
4. Add CLI flag in `src/Argus.py` for enabling the new mode
5. Update configuration examples in `models.yaml`

### Token Management
The system includes sophisticated token estimation (`src/utils.py:128-151`) using tiktoken. Preflight calculations help avoid expensive API overruns on large codebases.
