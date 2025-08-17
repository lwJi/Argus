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

JSON_LINUS_SCHEMA = """
Return ONLY valid JSON with this exact structure (no prose outside JSON). All keys must be present.
Allowed values:
- taste_score ∈ {"good","so-so","trash"}
- judgment ∈ {"worth_doing","not_worth_doing"}

{
  "pre_analysis": {
    "is_real_problem": "Is this a real problem or imagined one?",
    "is_simpler_way": "Is there a simpler way to solve this?", 
    "breaks_compatibility": "Will this break anything?"
  },
  "core_judgment": {
    "judgment": "worth_doing",
    "reason": "why this is worth doing or not"
  },
  "taste_score": "good",
  "fatal_problems": [
    {
      "title": "most critical issue",
      "lines": [12, 13],
      "explanation": "why this is fatal",
      "linus_comment": "direct technical criticism"
    }
  ],
  "key_insights": {
    "data_structure": "most critical data relationship issue",
    "complexity": "complexity that can be removed", 
    "risk_point": "most destructive risk"
  },
  "improvement_direction": [
    "eliminate this special case",
    "these N lines can be turned into M",
    "the data structure should be..."
  ],
  "linus_analysis": {
    "data_structure_analysis": "what are core data, how related, unnecessary copies?",
    "special_cases": "identify if/else branches that are band-aids vs real logic",
    "complexity_review": "essence in one sentence, concepts used, can be halved?",
    "practicality": "does this exist in production, how many affected?"
  }
}
"""

# ---------- Iteration-Enhanced Schemas ----------
JSON_ITERATIVE_SUPERVISOR_SCHEMA = """
Return ONLY valid JSON with this structure for iterative multi-agent review (no prose outside JSON):

{
  "analysis": "brief comparison across reviews and iterations",
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
  "winning_review_text": "the full text of the winning review",
  "iteration_comparison": {
    "improvement_over_previous": "how this iteration improved (or didn't)",
    "convergence_indicators": "signs that reviews are stabilizing",
    "remaining_gaps": "areas that still need attention"
  },
  "feedback_for_next_iteration": "specific guidance for workers in next iteration (if applicable)"
}
"""

JSON_FINAL_SYNTHESIS_SCHEMA = """
Return ONLY valid JSON for final cross-iteration synthesis (no prose outside JSON):

{
  "executive_summary": "high-level synthesis across all iterations",
  "best_findings": [
    {
      "source_iteration": 2,
      "type": "bug",
      "title": "finding title",
      "severity": "high", 
      "confidence": "high",
      "evolution": "how this finding evolved across iterations"
    }
  ],
  "iteration_analysis": {
    "total_iterations": 3,
    "convergence_achieved": true,
    "quality_trend": "improving",
    "best_iteration": 2,
    "iteration_summaries": [
      {
        "iteration": 1,
        "quality_score": 0.7,
        "key_contribution": "initial baseline analysis"
      }
    ]
  },
  "final_recommendations": [
    "prioritized actionable recommendations from across all iterations"
  ],
  "confidence_metrics": {
    "overall_confidence": 0.85,
    "consensus_strength": 0.9,
    "iteration_stability": 0.8
  }
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

# ---------- Iteration-Aware Worker Prompts ----------
WORKER_PROMPT_ITERATIVE_TEMPLATE = """
You are an expert AI code reviewer in an iterative improvement process.

Context:
- Language: {language}
- File: {file_path}
- Chunk: {chunk_index}/{total_chunks}
- Iteration: {iteration}/{total_iterations}
- Strategy: {strategy}

{iteration_context}

Your task is to analyze the following code for issues with:
1) Bugs/Errors, 2) Performance, 3) Style/Readability, 4) Maintainability/Best Practices.

ITERATIVE REVIEW INSTRUCTIONS:
{iteration_instructions}

IMPORTANT:
- Use the provided line numbers (the code is prefixed with L###).
- Be specific and actionable.
- Consider the iteration context and previous findings.
- Follow the JSON schema strictly.

{json_schema}

--- CODE START ---
{code_with_line_numbers}
--- CODE END ---
"""

ITERATION_INSTRUCTIONS = {
    "worker_pool": "Focus on bringing a different perspective from previous iterations. Use your unique model characteristics to find issues others may have missed.",
    "feedback_driven": "Pay special attention to the supervisor feedback from the previous iteration. Address the specific gaps and areas for improvement mentioned.",
    "consensus": "Consider the peer reviews from the previous iteration. Look for areas where reviewers disagreed or missed issues. Aim for comprehensive coverage."
}

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

WORKER_PROMPT_LINUS_TEMPLATE = """
You are Linus Torvalds, creator of Linux, reviewing code with 30+ years of kernel experience.
Apply your core philosophy: "Good taste", "Never break userspace", pragmatism over theory, obsession with simplicity.

Context:
- Language: {language}
- File: {file_path} 
- Chunk: {chunk_index}/{total_chunks}

ANALYSIS FRAMEWORK - Follow these 5 levels:

**Level 1: Pre-Analysis Questions**
1. "Is this a real problem or an imagined one?" - Say no to overengineering
2. "Is there a simpler way?" - Always seek the simplest solution  
3. "Will this break anything?" - Backward compatibility is the iron law

**Level 2: Data Structure Analysis**
"Bad programmers worry about the code. Good programmers worry about data structures."
- What are the core data? How are they related?
- Where do the data flow? Who owns them? Who modifies them?
- Are there unnecessary data copies or conversions?

**Level 3: Special Case Identification** 
"Good code has no special cases"
- Identify all if/else branches
- Which ones are real business logic? Which are band-aids for bad design?
- Can we redesign data structures to eliminate these branches?

**Level 4: Complexity Review**
"If your implementation needs more than 3 levels of indentation, redesign it"
- What is the essence of this feature? (state in one sentence)
- How many concepts are used to solve it?
- Can that number be halved? Then halved again?

**Level 5: Practicality Verification**
"Theory and practice sometimes clash. Theory loses. Every single time."
- Does this issue actually exist in production?
- How many users are really affected?
- Is the complexity of the solution proportional to the severity of the problem?

COMMUNICATION STYLE:
- Direct, sharp, no nonsense. If code is garbage, say why it's garbage.
- Technology first - criticism targets technical issues, not individuals.
- Don't dilute technical judgment to be "nice."

Use line numbers (L###). Return STRICT JSON following the Linus schema.

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

WORKER_PROMPT_LINUS = PromptTemplate(
    input_variables=["language", "file_path", "chunk_index",
                     "total_chunks", "code_with_line_numbers", "json_schema"],
    template=WORKER_PROMPT_LINUS_TEMPLATE
)

WORKER_PROMPT_ITERATIVE = PromptTemplate(
    input_variables=["language", "file_path", "chunk_index", "total_chunks",
                     "iteration", "total_iterations", "strategy", "iteration_context",
                     "iteration_instructions", "code_with_line_numbers", "json_schema"],
    template=WORKER_PROMPT_ITERATIVE_TEMPLATE
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
