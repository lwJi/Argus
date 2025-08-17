# agents.py
import json
from typing import List, Dict, Any, Optional, Tuple
import copy
from enum import Enum
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential_jitter
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from prompts import (
    WORKER_PROMPT_GENERIC,
    WORKER_PROMPT_CPP,
    WORKER_PROMPT_PY,
    WORKER_PROMPT_LINUS,
    WORKER_PROMPT_ITERATIVE,
    SUPERVISOR_PROMPT,
    SYNTHESIZER_PROMPT,
    JSON_WORKER_SCHEMA,
    JSON_SUPERVISOR_SCHEMA,
    JSON_LINUS_SCHEMA,
    JSON_ITERATIVE_SUPERVISOR_SCHEMA,
    JSON_FINAL_SYNTHESIS_SCHEMA,
    ITERATION_INSTRUCTIONS,
)
from utils import extract_json_from_text

# A minimal, strict JSON repair prompt used when a model wraps JSON in prose or emits invalid JSON.
REPAIR_JSON_PROMPT = PromptTemplate(
    input_variables=["malformed", "json_schema"],
    template="""
You are a strict JSON normalizer. Given a possibly malformed model output,
produce ONE valid JSON object that conforms to the schema below.

{json_schema}

Rules:
- Output ONLY the JSON object (no code fences, no prose, no extra text).
- Preserve content faithfully; if a required key is missing, include it with a sensible empty value.
- Do not invent findings beyond what is present.

--- MALFORMED START ---
{malformed}
--- MALFORMED END ---
""".strip()
)


async def _ainvoke_with_retry(chain, inputs: dict, attempts: int = 4) -> str:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=1, max=8)
    ):
        with attempt:
            return await chain.ainvoke(inputs)
    raise RuntimeError("Exhausted retries")


def render_worker_prompt_text(
    *, language: str, file_path: str, chunk_index: int, total_chunks: int, code_with_line_numbers: str, linus_mode: bool = False
) -> str:
    worker_prompt = _pick_worker_prompt(language, linus_mode=linus_mode)
    json_schema = JSON_LINUS_SCHEMA if linus_mode else JSON_WORKER_SCHEMA
    return worker_prompt.format(
        **(
            {"language": language} if worker_prompt in (WORKER_PROMPT_GENERIC, WORKER_PROMPT_LINUS) else {}
        ),
        file_path=file_path,
        chunk_index=chunk_index,
        total_chunks=total_chunks,
        code_with_line_numbers=code_with_line_numbers,
        json_schema=json_schema,
    )


def render_supervisor_prompt_text(*, reviews_text_block: str) -> str:
    return SUPERVISOR_PROMPT.format(reviews=reviews_text_block, json_schema=JSON_SUPERVISOR_SCHEMA)


def render_synthesizer_prompt_text(*, chunk_summaries_jsonl: str) -> str:
    return SYNTHESIZER_PROMPT.format(chunk_summaries=chunk_summaries_jsonl)


def _pick_worker_prompt(language: str, linus_mode: bool = False):
    if linus_mode:
        return WORKER_PROMPT_LINUS
    if language == "cpp":
        return WORKER_PROMPT_CPP
    if language == "python":
        return WORKER_PROMPT_PY
    return WORKER_PROMPT_GENERIC


async def run_worker_agent(
    llm: ChatOpenAI,
    *,
    language: str,
    file_path: str,
    chunk_index: int,
    total_chunks: int,
    code_with_line_numbers: str,
    linus_mode: bool = False
) -> Dict[str, Any]:
    """
    Returns parsed JSON dict following the worker schema (or Linus schema if linus_mode=True).
    """
    worker_prompt = _pick_worker_prompt(language, linus_mode=linus_mode)
    json_schema = JSON_LINUS_SCHEMA if linus_mode else JSON_WORKER_SCHEMA
    chain = worker_prompt | llm | StrOutputParser()
    
    # Build parameters based on prompt type
    if worker_prompt in (WORKER_PROMPT_GENERIC, WORKER_PROMPT_LINUS):
        params = {
            "language": language,
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "code_with_line_numbers": code_with_line_numbers,
            "json_schema": json_schema,
        }
    else:
        params = {
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "code_with_line_numbers": code_with_line_numbers,
            "json_schema": json_schema,
        }
    
    rendered = await _ainvoke_with_retry(chain, params)
    json_text = extract_json_from_text(rendered)
    try:
        return json.loads(json_text)
    except Exception:
        # Proper repair: feed the malformed output to a strict JSON normalizer that emits JSON only.
        repair_chain = (REPAIR_JSON_PROMPT | llm | StrOutputParser())
        repaired = await _ainvoke_with_retry(
            repair_chain,
            {"malformed": rendered, "json_schema": json_schema},
        )
        return json.loads(extract_json_from_text(repaired))


async def run_supervisor_agent(
    llm: ChatOpenAI,
    *,
    reviews_text_block: str
) -> Dict[str, Any]:
    """
    Returns parsed JSON dict following the supervisor schema.
    """
    chain = SUPERVISOR_PROMPT | llm | StrOutputParser()
    rendered = await _ainvoke_with_retry(
        chain,
        {"reviews": reviews_text_block, "json_schema": JSON_SUPERVISOR_SCHEMA}
    )
    json_text = extract_json_from_text(rendered)
    try:
        return json.loads(json_text)
    except Exception:
        # Proper repair: normalize to strict JSON using the malformed output.
        repair_chain = (REPAIR_JSON_PROMPT | llm | StrOutputParser())
        repaired = await _ainvoke_with_retry(
            repair_chain,
            {"malformed": rendered, "json_schema": JSON_SUPERVISOR_SCHEMA},
        )
        return json.loads(extract_json_from_text(repaired))


async def run_iterative_worker_agent(
    llm: ChatOpenAI,
    *,
    language: str,
    file_path: str,
    chunk_index: int,
    total_chunks: int,
    code_with_line_numbers: str,
    iteration_context: Dict[str, Any],
    linus_mode: bool = False
) -> Dict[str, Any]:
    """
    Iterative worker agent that uses iteration context for improved reviews.
    """
    iteration_num = iteration_context.get("iteration", 1)
    total_iterations = iteration_context.get("total_iterations_planned", 1)
    strategy = iteration_context.get("strategy", "worker_pool")
    
    # Build iteration context string
    context_parts = []
    if iteration_context.get("supervisor_feedback"):
        context_parts.append(f"SUPERVISOR FEEDBACK: {iteration_context['supervisor_feedback']}")
        
    if iteration_context.get("peer_reviews_previous"):
        peer_count = len(iteration_context["peer_reviews_previous"])
        context_parts.append(f"PREVIOUS PEER REVIEWS: {peer_count} reviews from previous iteration")
        # Could include actual peer review summaries here
        
    if iteration_context.get("previous_iterations"):
        prev_summaries = []
        for prev in iteration_context["previous_iterations"]:
            prev_summaries.append(f"Iteration {prev['iteration']}: {prev['summary']}")
        context_parts.append(f"HISTORY: {'; '.join(prev_summaries)}")
    
    context_str = "\n".join(context_parts) if context_parts else "First iteration - no previous context."
    
    # Get strategy-specific instructions
    instructions = ITERATION_INSTRUCTIONS.get(strategy, ITERATION_INSTRUCTIONS["worker_pool"])
    
    # Use iterative prompt or fall back to regular prompts
    if linus_mode:
        # For Linus mode, use regular Linus prompt (could enhance later)
        worker_prompt = WORKER_PROMPT_LINUS
        json_schema = JSON_LINUS_SCHEMA
        params = {
            "language": language,
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "code_with_line_numbers": code_with_line_numbers,
            "json_schema": json_schema,
        }
    else:
        # Use iterative prompt
        worker_prompt = WORKER_PROMPT_ITERATIVE
        json_schema = JSON_WORKER_SCHEMA
        params = {
            "language": language,
            "file_path": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "iteration": iteration_num,
            "total_iterations": total_iterations,
            "strategy": strategy,
            "iteration_context": context_str,
            "iteration_instructions": instructions,
            "code_with_line_numbers": code_with_line_numbers,
            "json_schema": json_schema,
        }
    
    chain = worker_prompt | llm | StrOutputParser()
    rendered = await _ainvoke_with_retry(chain, params)
    json_text = extract_json_from_text(rendered)
    
    try:
        return json.loads(json_text)
    except Exception:
        # Repair JSON if needed
        repair_chain = (REPAIR_JSON_PROMPT | llm | StrOutputParser())
        repaired = await _ainvoke_with_retry(
            repair_chain,
            {"malformed": rendered, "json_schema": json_schema},
        )
        return json.loads(extract_json_from_text(repaired))


async def run_iterative_supervisor_agent(
    llm: ChatOpenAI,
    *,
    reviews_text_block: str,
    iteration_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Enhanced supervisor agent that handles iteration-aware review comparison.
    Returns parsed JSON dict following the iterative supervisor schema.
    """
    # Create iteration-aware prompt that includes context from previous iterations
    prompt_text = render_iterative_supervisor_prompt_text(
        reviews_text_block=reviews_text_block,
        iteration_context=iteration_context
    )
    
    chain = PromptTemplate(
        input_variables=["prompt_text"],
        template="{prompt_text}"
    ) | llm | StrOutputParser()
    
    rendered = await _ainvoke_with_retry(chain, {"prompt_text": prompt_text})
    json_text = extract_json_from_text(rendered)
    
    try:
        return json.loads(json_text)
    except Exception:
        # Proper repair: feed the malformed output to a strict JSON normalizer
        repair_chain = (REPAIR_JSON_PROMPT | llm | StrOutputParser())
        repaired = await _ainvoke_with_retry(
            repair_chain,
            {"malformed": rendered, "json_schema": JSON_ITERATIVE_SUPERVISOR_SCHEMA},
        )
        return json.loads(extract_json_from_text(repaired))


async def run_synthesizer_agent(
    llm: ChatOpenAI,
    *,
    chunk_summaries_jsonl: str
) -> str:
    """
    Returns final Markdown string synthesizing chunk winners for the whole file.
    """
    chain = SYNTHESIZER_PROMPT | llm | StrOutputParser()
    md = await _ainvoke_with_retry(chain, {"chunk_summaries": chunk_summaries_jsonl})
    return md


def render_iterative_supervisor_prompt_text(*, reviews_text_block: str, iteration_context: Dict[str, Any]) -> str:
    """
    Render supervisor prompt with iteration context for iterative review improvement.
    """
    # Build context section based on iteration strategy and history
    context_parts = []
    
    # Add current iteration info
    iteration_num = iteration_context.get("iteration", 1)
    total_planned = iteration_context.get("total_iterations_planned", 1)
    strategy = iteration_context.get("strategy", "worker_pool")
    
    context_parts.append(f"ITERATION CONTEXT: This is iteration {iteration_num} of {total_planned} using {strategy} strategy.")
    
    # Add strategy-specific context
    if iteration_context.get("supervisor_feedback"):
        context_parts.append(f"PREVIOUS FEEDBACK: {iteration_context['supervisor_feedback']}")
        
    if iteration_context.get("peer_reviews_previous"):
        peer_count = len(iteration_context["peer_reviews_previous"])
        context_parts.append(f"PREVIOUS PEER REVIEWS: {peer_count} anonymized reviews from previous iteration available for comparison.")
        
    # Add previous iteration summaries
    if iteration_context.get("previous_iterations"):
        prev_summaries = []
        for prev in iteration_context["previous_iterations"]:
            prev_summaries.append(f"Iteration {prev['iteration']}: {prev['summary']}")
        context_parts.append(f"ITERATION HISTORY: {'; '.join(prev_summaries)}")
    
    context_block = "\n".join(context_parts) if context_parts else "This is the first iteration."
    
    prompt_template = f"""
You are an expert supervisor evaluating multiple AI code reviews in an iterative improvement process.

{context_block}

TASK: Compare the reviews below and select the best one. For iterative reviews, also:
1. Compare quality against previous iterations (if any)
2. Identify convergence indicators (are reviews becoming similar?)
3. Note remaining gaps that future iterations should address
4. Provide specific feedback for next iteration (if more are planned)

REVIEWS TO EVALUATE:
{reviews_text_block}

Provide your analysis using this JSON schema:
{JSON_ITERATIVE_SUPERVISOR_SCHEMA}
"""
    
    return prompt_template.strip()

def _clip_str(s: Any, n: int) -> Any:
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else (s[:n] + "…")

def _compact_review_for_supervisor(review: dict, max_field_chars: int = 800) -> dict:
    """
    Return a deep-copied review with long string fields clipped to reduce tokens.
    Preserves structure and keys so supervisor can still compare fairly.
    """
    out = json.loads(json.dumps(review))  # cheap deep copy that drops non-serializable bits
    # Top-level
    if "summary" in out:
        out["summary"] = _clip_str(out["summary"], max_field_chars)
    # Findings
    if isinstance(out.get("findings"), list):
        for f in out["findings"]:
            if isinstance(f, dict):
                for k in ("title", "snippet", "explanation", "suggestion", "diff"):
                    if k in f:
                        f[k] = _clip_str(f[k], max_field_chars)
    return out

def format_reviews_for_supervisor_compact(
    worker_jsons: List[dict],
    *,
    max_field_chars: int = 800,
    header_prefix: str = "#R"
) -> str:
    """
    Compact, delimiter-light formatter for supervisor input:
    - Minifies JSON (no pretty indent) to reduce tokens
    - Clips verbose fields (snippet/diff/etc.) to keep size bounded
    - Adds tiny headers (#R1, #R2, …) for separation
    """
    blocks = []
    for i, j in enumerate(worker_jsons, start=1):
        compact = _compact_review_for_supervisor(j, max_field_chars=max_field_chars)
        blocks.append(f"{header_prefix}{i}\n" + json.dumps(compact, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(blocks)


def format_reviews_for_supervisor(worker_jsons: List[dict]) -> str:
    """
    Convert worker JSON dicts back into readable, labeled text blocks for the supervisor.
    """
    blocks = []
    for i, j in enumerate(worker_jsons, start=1):
        # Compact rendering, but keep original JSON to preserve fidelity
        blocks.append(
            f"--- Review {i} JSON ---\n{json.dumps(j, ensure_ascii=False, indent=2)}")
    return "\n\n".join(blocks)


class IterationStrategy(Enum):
    """Defines different strategies for iterative review improvement."""
    WORKER_POOL = "worker_pool"          # Different models/temperatures each iteration
    FEEDBACK_DRIVEN = "feedback_driven"  # Supervisor feedback guides next iteration
    CONSENSUS = "consensus"              # Workers see peer reviews from previous iteration


class IterationController:
    """
    Manages iterative multi-agent review process with convergence detection and strategy control.
    """
    
    def __init__(self, max_iterations: int = 3, strategy: IterationStrategy = IterationStrategy.WORKER_POOL, 
                 convergence_threshold: float = 0.8, max_history_size: int = 10, retain_full_data: bool = False):
        self.max_iterations = max_iterations
        self.strategy = strategy
        self.convergence_threshold = convergence_threshold
        self.max_history_size = max_history_size
        self.retain_full_data = retain_full_data
        self.iteration_history: List[Dict[str, Any]] = []
        self.current_iteration = 0
        
    def should_continue_iterating(self) -> bool:
        """Determine if another iteration should be performed based on convergence and limits."""
        if self.current_iteration >= self.max_iterations:
            return False
            
        if self.current_iteration < 2:  # Need at least 2 iterations to compare
            return True
            
        # Check convergence based on review similarity
        if len(self.iteration_history) >= 2:
            current_iteration = self.iteration_history[-1]
            previous_iteration = self.iteration_history[-2]
            
            if self.retain_full_data:
                # Use original convergence logic with full review data
                current_reviews = current_iteration.get("worker_reviews", [])
                previous_reviews = previous_iteration.get("worker_reviews", [])
                convergence_score = self._calculate_convergence(current_reviews, previous_reviews)
            else:
                # Use compressed data for convergence analysis
                convergence_score = self._calculate_convergence_compressed(current_iteration, previous_iteration)
                
            return convergence_score < self.convergence_threshold
            
        return True
        
    def record_iteration(self, worker_reviews: List[Dict], supervisor_result: Dict, 
                        metadata: Dict[str, Any] = None) -> None:
        """Record the results of an iteration for convergence analysis and history tracking."""
        if self.retain_full_data:
            # Legacy behavior for debugging/analysis mode
            iteration_data = {
                "iteration": self.current_iteration + 1,
                "worker_reviews": copy.deepcopy(worker_reviews),
                "supervisor_result": copy.deepcopy(supervisor_result),
                "metadata": metadata or {},
                "timestamp": self._get_timestamp()
            }
        else:
            # Use compressed data to prevent memory leaks
            iteration_data = self._compress_iteration_data(worker_reviews, supervisor_result, metadata)
            
        self.iteration_history.append(iteration_data)
        self.current_iteration += 1
        
        # Enforce sliding window to prevent unbounded growth
        self._cleanup_old_iterations()
        
    def get_context_for_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """Get context data for the specified iteration based on strategy."""
        context = {
            "iteration": iteration_num,
            "total_iterations_planned": self.max_iterations,
            "strategy": self.strategy.value,
            "previous_iterations": []
        }
        
        if self.strategy == IterationStrategy.FEEDBACK_DRIVEN and self.iteration_history:
            # Include supervisor feedback from previous iteration
            last_iteration = self.iteration_history[-1]
            if self.retain_full_data:
                context["supervisor_feedback"] = last_iteration["supervisor_result"].get("feedback_for_next_iteration", "")
            else:
                context["supervisor_feedback"] = last_iteration.get("feedback_summary", "")
            
        elif self.strategy == IterationStrategy.CONSENSUS and self.iteration_history:
            # Include anonymized peer reviews from previous iteration
            last_iteration = self.iteration_history[-1]
            if self.retain_full_data:
                anonymized_reviews = self._anonymize_reviews(last_iteration["worker_reviews"])
                context["peer_reviews_previous"] = anonymized_reviews
            else:
                # For compressed data, provide summary instead of full reviews
                context["peer_reviews_previous"] = [{"summary": "Compressed review data available"}]
            
        # Always include summary of previous iterations for context
        for hist in self.iteration_history:
            context["previous_iterations"].append({
                "iteration": hist["iteration"],
                "summary": self._summarize_iteration(hist),
                "convergence_indicators": self._get_convergence_indicators(hist)
            })
            
        return context
        
    def get_final_synthesis_data(self) -> Dict[str, Any]:
        """Prepare data for final cross-iteration synthesis."""
        return {
            "total_iterations": len(self.iteration_history),
            "strategy_used": self.strategy.value,
            "convergence_achieved": len(self.iteration_history) < self.max_iterations,
            "iteration_history": self.iteration_history,
            "improvement_trajectory": self._analyze_improvement_trajectory(),
            "best_iteration": self._identify_best_iteration()
        }
        
    def _calculate_convergence(self, current: List[Dict], previous: List[Dict]) -> float:
        """Calculate convergence score between two sets of reviews (0.0 = different, 1.0 = identical)."""
        if not current or not previous:
            return 0.0
            
        # Simple convergence based on finding counts and severity distributions
        current_summary = self._extract_review_summary(current)
        previous_summary = self._extract_review_summary(previous)
        
        # Compare key metrics
        metrics = ["total_findings", "critical_count", "high_count", "bug_count", "performance_count"]
        similarities = []
        
        for metric in metrics:
            curr_val = current_summary.get(metric, 0)
            prev_val = previous_summary.get(metric, 0)
            
            if curr_val == prev_val == 0:
                similarities.append(1.0)
            elif curr_val == 0 or prev_val == 0:
                similarities.append(0.0)
            else:
                # Normalize difference
                similarity = 1.0 - abs(curr_val - prev_val) / max(curr_val, prev_val)
                similarities.append(similarity)
                
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _calculate_convergence_compressed(self, current_iteration: Dict[str, Any], 
                                        previous_iteration: Dict[str, Any]) -> float:
        """Calculate convergence score using compressed iteration data."""
        current_summary = current_iteration.get("review_summary", {})
        previous_summary = previous_iteration.get("review_summary", {})
        
        if not current_summary or not previous_summary:
            return 0.0
            
        # Compare key metrics from compressed summaries
        metrics = ["total_findings", "critical_count", "high_count", "bug_count", "performance_count"]
        similarities = []
        
        for metric in metrics:
            curr_val = current_summary.get(metric, 0)
            prev_val = previous_summary.get(metric, 0)
            
            if curr_val == prev_val == 0:
                similarities.append(1.0)
            elif curr_val == 0 or prev_val == 0:
                similarities.append(0.0)
            else:
                # Normalize difference
                similarity = 1.0 - abs(curr_val - prev_val) / max(curr_val, prev_val)
                similarities.append(similarity)
                
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    def _extract_review_summary(self, reviews: List[Dict]) -> Dict[str, int]:
        """Extract key metrics from a set of reviews for convergence analysis."""
        summary = {
            "total_findings": 0,
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "low_count": 0,
            "bug_count": 0,
            "performance_count": 0,
            "style_count": 0,
            "maintainability_count": 0
        }
        
        for review in reviews:
            findings = review.get("findings", [])
            summary["total_findings"] += len(findings)
            
            for finding in findings:
                severity = finding.get("severity", "").lower()
                if severity in summary:
                    summary[f"{severity}_count"] += 1
                    
                finding_type = finding.get("type", "").lower()
                if finding_type in ["bug", "performance", "style", "maintainability"]:
                    summary[f"{finding_type}_count"] += 1
                    
        return summary
        
    def _anonymize_reviews(self, reviews: List[Dict]) -> List[Dict]:
        """Create anonymized versions of reviews for consensus iteration."""
        anonymized = []
        for i, review in enumerate(reviews):
            anon_review = copy.deepcopy(review)
            # Remove model-specific metadata while keeping content
            anon_review.pop("model_info", None)
            anon_review.pop("temperature", None)
            anon_review["review_id"] = f"peer_{i+1}"
            anonymized.append(anon_review)
        return anonymized
        
    def _summarize_iteration(self, iteration_data: Dict) -> str:
        """Create a brief summary of an iteration for context."""
        if "worker_reviews" in iteration_data:
            # Full data format
            worker_count = len(iteration_data.get("worker_reviews", []))
            supervisor = iteration_data.get("supervisor_result", {})
            winner_idx = supervisor.get("winner_index")
        else:
            # Compressed data format
            supervisor_summary = iteration_data.get("supervisor_summary", {})
            worker_count = supervisor_summary.get("scores_count", 0)
            winner_idx = supervisor_summary.get("winner_index")
        
        return f"Iteration {iteration_data['iteration']}: {worker_count} workers, winner={winner_idx}"
        
    def _get_convergence_indicators(self, iteration_data: Dict) -> Dict[str, Any]:
        """Extract convergence indicators from iteration data."""
        if "worker_reviews" in iteration_data:
            # Full data format
            reviews = iteration_data.get("worker_reviews", [])
            summary = self._extract_review_summary(reviews)
            return {
                "finding_count": summary["total_findings"],
                "critical_issues": summary["critical_count"],
                "consensus_strength": len(reviews)
            }
        else:
            # Compressed data format
            review_summary = iteration_data.get("review_summary", {})
            supervisor_summary = iteration_data.get("supervisor_summary", {})
            return {
                "finding_count": review_summary.get("total_findings", 0),
                "critical_issues": review_summary.get("critical_count", 0),
                "consensus_strength": supervisor_summary.get("scores_count", 0)
            }
        
    def _analyze_improvement_trajectory(self) -> Dict[str, Any]:
        """Analyze how reviews improved across iterations."""
        if len(self.iteration_history) < 2:
            return {"trend": "insufficient_data"}
            
        trajectories = {
            "finding_counts": [],
            "critical_counts": [],
            "quality_scores": []
        }
        
        for iteration in self.iteration_history:
            if "worker_reviews" in iteration:
                # Full data format
                reviews = iteration.get("worker_reviews", [])
                summary = self._extract_review_summary(reviews)
                
                trajectories["finding_counts"].append(summary["total_findings"])
                trajectories["critical_counts"].append(summary["critical_count"])
                
                # Quality score based on supervisor analysis
                supervisor = iteration.get("supervisor_result", {})
                scores = supervisor.get("scores", [])
                if scores:
                    avg_score = sum(s.get("accuracy", 0) + s.get("completeness", 0) + 
                                    s.get("clarity", 0) + s.get("insightfulness", 0) 
                                    for s in scores) / (len(scores) * 4)
                    trajectories["quality_scores"].append(avg_score)
                else:
                    trajectories["quality_scores"].append(0.0)
            else:
                # Compressed data format
                review_summary = iteration.get("review_summary", {})
                supervisor_summary = iteration.get("supervisor_summary", {})
                
                trajectories["finding_counts"].append(review_summary.get("total_findings", 0))
                trajectories["critical_counts"].append(review_summary.get("critical_count", 0))
                trajectories["quality_scores"].append(supervisor_summary.get("avg_quality", 0.0))
                
        return {
            "trend": self._determine_trend(trajectories["quality_scores"]),
            "trajectories": trajectories,
            "final_quality": trajectories["quality_scores"][-1] if trajectories["quality_scores"] else 0.0
        }
        
    def _determine_trend(self, values: List[float]) -> str:
        """Determine if values show improving, declining, or stable trend."""
        if len(values) < 2:
            return "insufficient_data"
            
        if values[-1] > values[0] * 1.1:
            return "improving"
        elif values[-1] < values[0] * 0.9:
            return "declining"
        else:
            return "stable"
            
    def _identify_best_iteration(self) -> Dict[str, Any]:
        """Identify which iteration produced the best results."""
        if not self.iteration_history:
            return {"iteration": 0, "reason": "no_iterations"}
            
        best_iteration = 1
        best_score = 0.0
        
        for i, iteration in enumerate(self.iteration_history, 1):
            if "supervisor_result" in iteration:
                # Full data format
                supervisor = iteration.get("supervisor_result", {})
                scores = supervisor.get("scores", [])
                
                if scores:
                    # Calculate average quality score
                    total_score = sum(
                        s.get("accuracy", 0) + s.get("completeness", 0) + 
                        s.get("clarity", 0) + s.get("insightfulness", 0) 
                        for s in scores
                    )
                    avg_score = total_score / (len(scores) * 4) if scores else 0.0
                else:
                    avg_score = 0.0
            else:
                # Compressed data format
                supervisor_summary = iteration.get("supervisor_summary", {})
                avg_score = supervisor_summary.get("avg_quality", 0.0)
                
            if avg_score > best_score:
                best_score = avg_score
                best_iteration = i
                    
        return {
            "iteration": best_iteration,
            "score": best_score,
            "reason": "highest_quality_score"
        }
        
    def _compress_iteration_data(self, worker_reviews: List[Dict], supervisor_result: Dict, 
                               metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a compressed summary of iteration data to reduce memory usage.
        Extracts only essential metrics needed for convergence analysis and final synthesis.
        """
        compressed = {
            "iteration": self.current_iteration + 1,
            "timestamp": self._get_timestamp(),
            "metadata": metadata or {},
        }
        
        # Extract essential review metrics
        review_summary = self._extract_review_summary(worker_reviews)
        compressed["review_summary"] = review_summary
        
        # Extract supervisor decision data
        compressed["supervisor_summary"] = {
            "winner_index": supervisor_result.get("winner_index"),
            "convergence_indicators": supervisor_result.get("iteration_comparison", {}).get("convergence_indicators", ""),
            "improvement_noted": bool(supervisor_result.get("iteration_comparison", {}).get("improvement_over_previous")),
            "scores_count": len(supervisor_result.get("scores", [])),
            "avg_quality": self._calculate_average_quality(supervisor_result.get("scores", []))
        }
        
        # Keep essential context for strategies
        if self.strategy == IterationStrategy.FEEDBACK_DRIVEN:
            compressed["feedback_summary"] = supervisor_result.get("feedback_for_next_iteration", "")[:200]  # Truncate
            
        return compressed
    
    def _calculate_average_quality(self, scores: List[Dict[str, Any]]) -> float:
        """Calculate average quality score from supervisor scores."""
        if not scores:
            return 0.0
            
        total_score = sum(
            s.get("accuracy", 0) + s.get("completeness", 0) + 
            s.get("clarity", 0) + s.get("insightfulness", 0) 
            for s in scores
        )
        return total_score / (len(scores) * 4) if scores else 0.0
    
    def _cleanup_old_iterations(self) -> None:
        """Remove old iterations beyond max_history_size to prevent memory leaks."""
        if len(self.iteration_history) > self.max_history_size:
            # Keep most recent iterations
            excess = len(self.iteration_history) - self.max_history_size
            self.iteration_history = self.iteration_history[excess:]
    
    def clear_history(self) -> None:
        """Manually clear iteration history to free memory."""
        self.iteration_history.clear()
        
    def _get_timestamp(self) -> str:
        """Get current timestamp for iteration tracking."""
        import datetime
        return datetime.datetime.now().isoformat()
