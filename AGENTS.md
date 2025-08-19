# Repository Guidelines

## Project Structure & Modules
- `src/Argus.py`: CLI entrypoint and pipeline orchestrator.
- `src/agents.py`, `src/prompts.py`, `src/rate_limiting.py`, `src/utils.py`: Core logic, prompt templates, rate limiting, and helpers.
- `src/models.yaml` (example at `src/models.example.yaml`): Model configuration.
- `tests/`: Pytest suite (`test_*.py`), fixtures, and integration tests.
- `examples/`: Sample inputs/usage.
- CI: `.github/workflows/ci.yml` runs tests on Python 3.8–3.11.

## Build, Test, and Development
- Install deps: `pip install -r src/requirements.txt`
- Run all tests: `pytest`
- Coverage (XML + terminal): `pytest --cov=src --cov-report=xml --cov-report=term-missing`
- Run locally: `python src/Argus.py <directory> [--linus-mode] [--iterations N] [--skip-preflight]`
- Env: set `OPENAI_API_KEY` (supports `.env`, `~/.argus.env`, `~/.env`, `~/.config/argus/env`).

## Coding Style & Naming
- Python 3.8+; prefer type hints and docstrings.
- Indentation: 4 spaces; line length ~100 where practical.
- Naming: modules `snake_case.py`; classes `CamelCase`; functions/vars `snake_case`.
- Imports: standard lib, third-party, local (grouped, alphabetized).
- Be mindful of token usage; reuse utilities in `utils.py` for chunking/estimation.

## Testing Guidelines
- Frameworks: `pytest`, `pytest-asyncio`.
- Test layout: `tests/test_*.py`; classes start with `Test*`; functions `test_*` (configured in `pytest.ini`).
- Write fast, deterministic tests; prefer fixtures in `tests/fixtures/` or `conftest.py`.
- Aim for meaningful coverage on new logic; keep CI green.

## Commit & Pull Requests
- Commit style: Conventional Commits (`feat:`, `fix:`, `docs:`, etc.) as seen in history.
- PRs should include: clear description, linked issues, reproduction steps, before/after behavior, and test updates.
- Update docs when adding flags, config, or modes (README, examples, configs).
- Keep changes focused; separate refactors from features when possible.

## Security & Configuration
- Do not commit secrets; use env files or OS keychain.
- Models and rate limits: update `src/models.yaml` and `rate_limits.example.yaml`; keep examples generic.
- Local runs can be expensive—use `--skip-preflight` intentionally and prefer small test directories during development.
