# prompts.py
from langchain.prompts import PromptTemplate

# ---------- Common JSON guidance ----------
JSON_WORKER_SCHEMA = """
Return ONLY valid JSON with this exact object shape (no prose outside JSON). All keys must be present.
Allowed values:
- type ∈ {"bug","performance","style","maintainability"}
- severity ∈ {"low","medium","high","critical"}

{
  "summary": "one-paragraph overview of key issues and themes",
  "findings": [
    {
      "type": "bug",
      "title": "short title",
      "severity": "medium",
      "lines": [12, 13],
      "snippet": "small relevant code excerpt (<= 10 lines)",
      "explanation": "why this is an issue",
      "suggestion": "actionable fix or improvement",
      "diff": ""
    }
  ],
  "counts": {"bug": 0, "performance": 0, "style": 0, "maintainability": 0}
}
"""

JSON_SUPERVISOR_SCHEMA = """
Return ONLY valid JSON with this structure (no prose outside JSON):

{
  "analysis": "brief comparison across reviews",
  "scores": [
    {
      "review_index": 1,
      "accuracy": 0.0,
      "completeness": 0.0,
      "clarity": 0.0,
      "insightfulness": 0.0,
      "notes": "brief justification"
    }
  ],
  "winner_index": 1,
  "merged_takeaways": [
    "concise bullet capturing the best, non-duplicated insights across reviews"
  ],
  "winning_review_text": "the full text of the winning review"
}
"""

# ---------- Worker Prompts (Language-aware) ----------
WORKER_PROMPT_GENERIC_TEMPLATE = """
You are an expert AI code reviewer.

Context:
- Language: {language}
- File: {file_path}
- Chunk: {chunk_index}/{total_chunks}

Your task is to analyze the following code for issues with:
1) Bugs/Errors, 2) Performance, 3) Style/Readability, 4) Maintainability/Best Practices.

IMPORTANT:
- Use the provided line numbers (the code is prefixed with L###).
- Be specific and actionable.
- Follow the JSON schema strictly.

{json_schema}

--- CODE START ---
{code_with_line_numbers}
--- CODE END ---
"""

WORKER_PROMPT_CPP_TEMPLATE = """
You are an expert C++ reviewer (C++17/20). Apply C++ Core Guidelines, RAII, const-correctness,
exception safety, performance (allocations, copies, move semantics), and readability (Google style acceptable).

Context:
- Language: C++
- File: {file_path}
- Chunk: {chunk_index}/{total_chunks}

Focus additionally on:
- Correctness (UB, lifetime, thread-safety, iterator invalidation)
- API design, value categories, noexcept, inline vs ODR, headers hygiene

Use the provided line numbers (L###). Return STRICT JSON.

{json_schema}

--- CODE START ---
{code_with_line_numbers}
--- CODE END ---
"""

WORKER_PROMPT_PY_TEMPLATE = """
You are an expert Python reviewer. Apply PEP 8/20, type hints, error handling, performance (avoid N^2, eager I/O),
and maintainability.

Context:
- Language: Python
- File: {file_path}
- Chunk: {chunk_index}/{total_chunks}

Use the provided line numbers (L###). Return STRICT JSON.

{json_schema}

--- CODE START ---
{code_with_line_numbers}
--- CODE END ---
"""

WORKER_PROMPT_GENERIC = PromptTemplate(
    input_variables=["language", "file_path", "chunk_index",
                     "total_chunks", "code_with_line_numbers", "json_schema"],
    template=WORKER_PROMPT_GENERIC_TEMPLATE
)

WORKER_PROMPT_CPP = PromptTemplate(
    input_variables=["file_path", "chunk_index",
                     "total_chunks", "code_with_line_numbers", "json_schema"],
    template=WORKER_PROMPT_CPP_TEMPLATE
)

WORKER_PROMPT_PY = PromptTemplate(
    input_variables=["file_path", "chunk_index",
                     "total_chunks", "code_with_line_numbers", "json_schema"],
    template=WORKER_PROMPT_PY_TEMPLATE
)

# ---------- Supervisor Prompt ----------
SUPERVISOR_PROMPT_TEMPLATE = """
You are a Staff Software Engineer evaluating multiple AI code reviews for the SAME code chunk.
Pick the best review and synthesize cross-review takeaways.

Original code (with line numbers) is available to you only implicitly; judge based on reviews' internal consistency,
specificity, and plausibility.

Criteria:
- Accuracy
- Completeness (bugs, performance, style)
- Clarity
- Insightfulness

Return STRICT JSON as per schema.

{json_schema}

--- REVIEWS START ---
{reviews}
--- REVIEWS END ---
"""

SUPERVISOR_PROMPT = PromptTemplate(
    input_variables=["reviews", "json_schema"],
    template=SUPERVISOR_PROMPT_TEMPLATE
)

# ---------- File-level Synthesizer (merges chunk winners) ----------
SYNTHESIZER_PROMPT_TEMPLATE = """
You are a Principal Engineer creating a final, human-readable review for a whole file by merging
the BEST per-chunk reviews and their takeaways.

Provide:
1) A concise executive summary.
2) A categorized list of findings (bugs, performance, style, maintainability) with line references.
3) A short prioritized action list (most impactful first).
4) (Optional) A code block with a minimal diff if appropriate.

Write Markdown for humans.

--- CHUNK SUMMARIES (JSON blobs, one per chunk) ---
{chunk_summaries}
--- END ---
"""

SYNTHESIZER_PROMPT = PromptTemplate(
    input_variables=["chunk_summaries"],
    template=SYNTHESIZER_PROMPT_TEMPLATE
)
