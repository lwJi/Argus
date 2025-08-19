#!/usr/bin/env python

# main.py
import os
import argparse
import asyncio
import json
from typing import List, Dict, Tuple, Any
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils import (
    list_source_files, detect_language_from_extension, chunk_code_by_lines,
    add_line_numbers_preserve, safe_filename_from_path,
    save_json, save_text, load_models_config, ensure_dir, content_hash,
    count_tokens_text, get_model_name
)
from agents import (
    run_worker_agent, run_supervisor_agent, run_synthesizer_agent, format_reviews_for_supervisor,
    render_worker_prompt_text, render_supervisor_prompt_text, render_synthesizer_prompt_text,
    format_reviews_for_supervisor_compact, IterationController, IterationStrategy,
    run_iterative_worker_agent, run_iterative_supervisor_agent, _ainvoke_with_retry,
    set_global_rate_limiter
)
from rate_limiting import APIRateLimiter, RateLimitMode

console = Console()

# --- Non-blocking console input for async contexts ---
async def _async_input(prompt: str = "") -> str:
    """
    Read from stdin without blocking the event loop by offloading to a thread.
    Returns empty string on EOF.
    """
    loop = asyncio.get_running_loop()
    def _read():
        try: return input(prompt)
        except EOFError: return ""
    return await loop.run_in_executor(None, _read)

DEFAULT_MAX_LINES = 500  # chunking threshold
DEFAULT_CHUNK_SIZE = 400


def build_llms(models_cfg: dict) -> Dict[str, ChatOpenAI]:
    """
    models.yaml (optional) example:
    workers:
      - model: gpt-4o
        temperature: 0.2
      - model: gpt-4o
        temperature: 0.7
    supervisor:
      model: gpt-4o
      temperature: 0.1
    """
    workers = []
    for w in (models_cfg.get("workers") or []):
        workers.append(ChatOpenAI(
            model=w["model"], temperature=w.get("temperature", 0.3)))
    if not workers:
        # Fall back if no config
        workers = [
            ChatOpenAI(model="gpt-4o-mini", temperature=0.7),
            ChatOpenAI(model="gpt-4o", temperature=0.2),
        ]
    sup_cfg = models_cfg.get("supervisor") or {
        "model": "gpt-4o", "temperature": 0.1}
    supervisor = ChatOpenAI(
        model=sup_cfg["model"], temperature=sup_cfg.get("temperature", 0.1))
    synthesizer = supervisor  # reuse for now
    return {"workers": workers, "supervisor": supervisor, "synthesizer": synthesizer}


async def review_chunk_for_file(
    *,
    llms: Dict[str, ChatOpenAI],
    file_path: str,
    language: str,
    chunk_index: int,
    total_chunks: int,
    code_chunk: str,
    start_line: int,
    linus_mode: bool = False,
):
    # Prepare numbered code using the actual chunk start line
    numbered = add_line_numbers_preserve(code_chunk, start_line=start_line)

    # Run workers
    tasks = [
        run_worker_agent(
            llm=w,
            language=language,
            file_path=file_path,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            code_with_line_numbers=numbered,
            linus_mode=linus_mode,
        )
        for w in llms["workers"]
    ]
    # Be resilient: don't cancel all on first failure
    results = await asyncio.gather(*tasks, return_exceptions=True)
    worker_jsons = []
    failed = []
    for i, r in enumerate(results, start=1):
        if isinstance(r, Exception):
            failed.append((i, r))
        else:
            worker_jsons.append(r)

    # Log any worker failures but continue if we have at least one success
    if failed:
        for idx, err in failed:
            console.print(
                f"[yellow]⚠ Worker {idx} failed for "
                f"{os.path.basename(file_path)} (chunk {chunk_index}/{total_chunks}): {err}[/yellow]"
            )

    # If all workers failed, skip supervisor; return a minimal summary so the pipeline stays alive
    if not worker_jsons:
        console.print(
            f"[red]All worker agents failed for {os.path.basename(file_path)} "
            f"(chunk {chunk_index}/{total_chunks}). Skipping supervisor.[/red]"
        )
        sup_json = {
            "analysis": "All worker agents failed",
            "scores": [],
            "winner_index": None,
            "merged_takeaways": [],
            "winning_review_text": ""
        }
        summary = {
            "file": file_path,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "winner_index": None,
            "scores": [],
            "merged_takeaways": [],
            "winning_review_text": ""
        }
        return summary, [], sup_json

    # Supervisor decision
    # Prefer compact, token-friendly formatting
    reviews_for_sup = format_reviews_for_supervisor_compact(worker_jsons, max_field_chars=800)

    # Light token-budget guard: if the full supervisor prompt looks too large, clip harder
    try:
        sup_model = get_model_name(llms["supervisor"])
        prompt_for_count = render_supervisor_prompt_text(reviews_text_block=reviews_for_sup)
        toks = count_tokens_text(sup_model, prompt_for_count)
        # Heuristic threshold—kept conservative to avoid hitting context limits after adding model overhead
        if toks > 6000:
            console.print(
                f"[yellow]⚠ Supervisor input ~{toks} tokens; clipping long fields further for {os.path.basename(file_path)} "
                f"(chunk {chunk_index}/{total_chunks}).[/yellow]"
            )
            reviews_for_sup = format_reviews_for_supervisor_compact(worker_jsons, max_field_chars=300)
    except Exception:
        # Counting is best-effort only; continue even if tiktoken not available
        pass
    sup_json = await run_supervisor_agent(llms["supervisor"], reviews_text_block=reviews_for_sup)

    # Compose a JSON summary to feed to the file-level synthesizer later
    summary = {
        "file": file_path,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "winner_index": sup_json.get("winner_index"),
        "scores": sup_json.get("scores", []),
        "merged_takeaways": sup_json.get("merged_takeaways", []),
        "winning_review_text": sup_json.get("winning_review_text", ""),
    }
    return summary, worker_jsons, sup_json


async def review_chunk_iterative(
    *,
    llms: Dict[str, ChatOpenAI],
    file_path: str,
    language: str,
    chunk_index: int,
    total_chunks: int,
    code_chunk: str,
    start_line: int,
    iteration_controller: IterationController,
    linus_mode: bool = False,
) -> Tuple[Dict[str, Any], List[List[Dict]], List[Dict]]:
    """
    Iterative version of review_chunk_for_file that performs multiple review iterations
    with convergence detection and strategy-based improvement.
    
    Returns:
        - summary: Final chunk summary with iteration metadata
        - all_iterations_workers: List of worker results for each iteration
        - all_iterations_supervisors: List of supervisor results for each iteration
    """
    numbered = add_line_numbers_preserve(code_chunk, start_line=start_line)
    
    all_iterations_workers = []
    all_iterations_supervisors = []
    
    iteration_num = 1
    while iteration_controller.should_continue_iterating():
        console.print(f"[dim]  → Iteration {iteration_num}/{iteration_controller.max_iterations}[/dim]")
        
        # Get iteration context for strategy-specific behavior
        iteration_context = iteration_controller.get_context_for_iteration(iteration_num)
        
        # Run worker agents with iteration context
        worker_tasks = []
        for w in llms["workers"]:
            # Use iterative worker agent with context
            task = run_iterative_worker_agent(
                llm=w,
                language=language,
                file_path=file_path,
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                code_with_line_numbers=numbered,
                iteration_context=iteration_context,
                linus_mode=linus_mode,
            )
            worker_tasks.append(task)
            
        # Be resilient: don't cancel all on first failure
        results = await asyncio.gather(*worker_tasks, return_exceptions=True)
        worker_jsons = []
        failed = []
        
        for i, r in enumerate(results, start=1):
            if isinstance(r, Exception):
                failed.append((i, r))
            else:
                worker_jsons.append(r)

        # Log any worker failures
        if failed:
            for idx, err in failed:
                console.print(
                    f"[yellow]⚠ Worker {idx} failed in iteration {iteration_num} for "
                    f"{os.path.basename(file_path)} (chunk {chunk_index}/{total_chunks}): {err}[/yellow]"
                )

        # If all workers failed, break iteration loop
        if not worker_jsons:
            console.print(
                f"[red]All worker agents failed in iteration {iteration_num} for {os.path.basename(file_path)} "
                f"(chunk {chunk_index}/{total_chunks}). Ending iterations early.[/red]"
            )
            break
            
        # Supervisor decision with iteration awareness
        reviews_for_sup = format_reviews_for_supervisor_compact(worker_jsons, max_field_chars=800)
        
        # Use iterative supervisor schema for better iteration handling
        sup_json = await run_iterative_supervisor_agent(
            llms["supervisor"], 
            reviews_text_block=reviews_for_sup,
            iteration_context=iteration_context
        )
        
        # Record this iteration's results
        all_iterations_workers.append(worker_jsons)
        all_iterations_supervisors.append(sup_json)
        
        # Record iteration in controller for convergence analysis
        iteration_metadata = {
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "file_path": file_path,
            "iteration_num": iteration_num
        }
        iteration_controller.record_iteration(worker_jsons, sup_json, iteration_metadata)
        
        iteration_num += 1
        
        # Check convergence - controller will decide if we should continue
        if not iteration_controller.should_continue_iterating():
            console.print(f"[dim]  ✓ Converged after {iteration_num-1} iterations[/dim]")
            break
    
    # Generate final synthesis across all iterations
    final_synthesis_data = iteration_controller.get_final_synthesis_data()
    best_iteration_idx = final_synthesis_data["best_iteration"]["iteration"] - 1  # Convert to 0-based index
    
    # Use results from best iteration, but include iteration metadata
    if all_iterations_supervisors and best_iteration_idx < len(all_iterations_supervisors):
        best_supervisor_result = all_iterations_supervisors[best_iteration_idx]
    else:
        best_supervisor_result = all_iterations_supervisors[-1] if all_iterations_supervisors else {}
    
    summary = {
        "file": file_path,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "winner_index": best_supervisor_result.get("winner_index"),
        "scores": best_supervisor_result.get("scores", []),
        "merged_takeaways": best_supervisor_result.get("merged_takeaways", []),
        "winning_review_text": best_supervisor_result.get("winning_review_text", ""),
        # Iteration metadata
        "iteration_metadata": {
            "total_iterations": len(all_iterations_supervisors),
            "best_iteration": final_synthesis_data["best_iteration"],
            "convergence_achieved": final_synthesis_data["convergence_achieved"],
            "strategy_used": final_synthesis_data["strategy_used"],
            "improvement_trend": final_synthesis_data.get("improvement_trajectory", {}).get("trend", "unknown")
        }
    }
    
    return summary, all_iterations_workers, all_iterations_supervisors


async def review_single_file(
    *,
    llms: Dict[str, ChatOpenAI],
    file_path: str,
    save_dir: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_lines_before_chunk: int = DEFAULT_MAX_LINES,
    linus_mode: bool = False,
    iterations: int = 1,
    iteration_strategy: str = "worker_pool",
    convergence_threshold: float = 0.8,
    retain_full_iteration_data: bool = False
):
    # Read file
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        console.print(f"[red]Error reading {file_path}: {e}[/red]")
        return

    language = detect_language_from_extension(os.path.splitext(file_path)[1])
    lines = code.splitlines()
    if len(lines) > max_lines_before_chunk:
        chunks = chunk_code_by_lines(code, max_lines=chunk_size)
    else:
        chunks = [(1, code)]

    total_chunks = len(chunks)
    summaries = []
    all_workers_json = []
    all_sup_json = []
    
    # Decide whether to use iterative processing
    use_iterations = iterations > 1
    
    if use_iterations:
        console.print(f"[dim]Using iterative processing: {iterations} iterations, {iteration_strategy} strategy[/dim]")

    # Progress per file
    progress_desc = f"Reviewing {os.path.basename(file_path)} ({total_chunks} chunk{'s' if total_chunks > 1 else ''})"
    if use_iterations:
        progress_desc += f" [italic]({iterations} iterations)[/italic]"
        
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task(progress_desc, total=total_chunks)
        
        for idx, (_start, chunk_text) in enumerate(chunks, start=1):
            if use_iterations:
                # Create iteration controller for this chunk
                strategy_enum = IterationStrategy(iteration_strategy)
                iteration_controller = IterationController(
                    max_iterations=iterations,
                    strategy=strategy_enum,
                    convergence_threshold=convergence_threshold,
                    retain_full_data=retain_full_iteration_data
                )
                
                summary, iter_workers_json, iter_sup_json = await review_chunk_iterative(
                    llms=llms,
                    file_path=file_path,
                    language=language,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    code_chunk=chunk_text,
                    start_line=_start,
                    iteration_controller=iteration_controller,
                    linus_mode=linus_mode,
                )
                
                summaries.append(summary)
                all_workers_json.append(iter_workers_json)  # List of lists for each iteration
                all_sup_json.append(iter_sup_json)  # List of supervisor results for each iteration
            else:
                # Use original single-pass processing
                summary, worker_jsons, sup_json = await review_chunk_for_file(
                    llms=llms,
                    file_path=file_path,
                    language=language,
                    chunk_index=idx,
                    total_chunks=total_chunks,
                    code_chunk=chunk_text,
                    start_line=_start,
                    linus_mode=linus_mode,
                )
                summaries.append(summary)
                all_workers_json.append(worker_jsons)
                all_sup_json.append(sup_json)
                
            progress.update(task, advance=1)

    # Synthesize final Markdown across chunks
    chunk_summaries_jsonl = "\n".join(
        [json.dumps(s, ensure_ascii=False) for s in summaries])
    
    if use_iterations:
        # Enhanced synthesis for iterative results
        final_markdown = await run_iterative_synthesizer_agent(
            llms["synthesizer"], 
            chunk_summaries_jsonl=chunk_summaries_jsonl,
            iteration_metadata={"use_iterations": True, "iterations": iterations, "strategy": iteration_strategy}
        )
    else:
        final_markdown = await run_synthesizer_agent(llms["synthesizer"], chunk_summaries_jsonl=chunk_summaries_jsonl)

    # Persist outputs
    ensure_dir(save_dir)
    base = safe_filename_from_path(os.path.relpath(file_path))
    run_id = content_hash(file_path, "".join(lines))

    json_path = os.path.join(save_dir, f"{base}.{run_id}.reviews.json")
    md_path = os.path.join(save_dir, f"{base}.{run_id}.review.md")

    # Enhanced payload with iteration metadata
    payload = {
        "file": file_path,
        "language": language,
        "chunks": summaries,
        "workers_raw": all_workers_json,
        "supervisor_raw": all_sup_json,
        "final_markdown_path": md_path,
        # Iteration metadata
        "iteration_metadata": {
            "enabled": use_iterations,
            "iterations": iterations if use_iterations else 1,
            "strategy": iteration_strategy if use_iterations else "single_pass",
            "convergence_threshold": convergence_threshold if use_iterations else None,
        }
    }
    
    # Add iteration-specific analytics if applicable
    if use_iterations and summaries:
        iteration_analytics = analyze_iteration_results(summaries, all_workers_json, all_sup_json)
        payload["iteration_analytics"] = iteration_analytics
    
    save_json(json_path, payload)
    save_text(md_path, final_markdown)

    # Enhanced output messages
    if use_iterations:
        iteration_info = f" ({iterations} iterations, {iteration_strategy})"
        console.print(f"\n[bold green]✓ Saved[/bold green] JSON: {json_path}{iteration_info}")
        console.print(f"[bold green]✓ Saved[/bold green] Markdown: {md_path}{iteration_info}\n")
    else:
        console.print(f"\n[bold green]✓ Saved[/bold green] JSON: {json_path}")
        console.print(f"[bold green]✓ Saved[/bold green] Markdown: {md_path}\n")


async def run_iterative_synthesizer_agent(
    llm: ChatOpenAI,
    *,
    chunk_summaries_jsonl: str,
    iteration_metadata: Dict[str, Any]
) -> str:
    """
    Enhanced synthesizer for iterative review results that highlights 
    iteration evolution and convergence patterns.
    """
    # Create enhanced prompt that considers iteration data
    prompt_text = f"""
You are a senior technical writer synthesizing code review results from an iterative multi-agent analysis.

ITERATION CONTEXT:
- Total iterations: {iteration_metadata.get('iterations', 1)}
- Strategy used: {iteration_metadata.get('strategy', 'single_pass')}

Your task is to create a comprehensive markdown report that:
1. Synthesizes findings from all chunks
2. Highlights how reviews evolved across iterations (if applicable)
3. Shows convergence patterns and quality improvements
4. Provides actionable recommendations prioritized by confidence level

For iterative reviews, include:
- Iteration summary showing how analysis improved over time
- Confidence metrics based on convergence
- Best findings from across all iterations with their evolution

CHUNK DATA (JSON Lines):
{chunk_summaries_jsonl}

Create a well-structured markdown report with clear sections and actionable insights.
"""
    
    chain = PromptTemplate(
        input_variables=["prompt_text"],
        template="{prompt_text}"
    ) | llm | StrOutputParser()
    
    return await _ainvoke_with_retry(chain, {"prompt_text": prompt_text}, llm=llm)


def analyze_iteration_results(summaries: List[Dict], all_workers_json: List, all_sup_json: List) -> Dict[str, Any]:
    """
    Analyze iteration results to provide insights into convergence and quality evolution.
    """
    analytics = {
        "total_chunks": len(summaries),
        "chunks_with_iterations": 0,
        "convergence_summary": {},
        "quality_trends": {},
        "iteration_distribution": {}
    }
    
    for summary in summaries:
        iteration_meta = summary.get("iteration_metadata", {})
        if iteration_meta.get("total_iterations", 1) > 1:
            analytics["chunks_with_iterations"] += 1
            
            # Track iteration counts
            total_iters = iteration_meta.get("total_iterations", 1)
            if total_iters not in analytics["iteration_distribution"]:
                analytics["iteration_distribution"][total_iters] = 0
            analytics["iteration_distribution"][total_iters] += 1
            
            # Track convergence
            converged = iteration_meta.get("convergence_achieved", False)
            if converged not in analytics["convergence_summary"]:
                analytics["convergence_summary"][converged] = 0
            analytics["convergence_summary"][converged] += 1
            
            # Track quality trends
            trend = iteration_meta.get("improvement_trend", "unknown")
            if trend not in analytics["quality_trends"]:
                analytics["quality_trends"][trend] = 0
            analytics["quality_trends"][trend] += 1
    
    # Calculate percentages
    if analytics["chunks_with_iterations"] > 0:
        analytics["convergence_rate"] = analytics["convergence_summary"].get(True, 0) / analytics["chunks_with_iterations"]
        analytics["improvement_rate"] = analytics["quality_trends"].get("improving", 0) / analytics["chunks_with_iterations"]
    else:
        analytics["convergence_rate"] = 0.0
        analytics["improvement_rate"] = 0.0
    
    return analytics


def preflight_estimate(
    *,
    llms: Dict[str, ChatOpenAI],
    files: List[str],
    chunk_size: int,
    chunk_threshold: int,
    assume_worker_out: int,
    assume_supervisor_out: int,
    assume_synthesizer_out: int,
    linus_mode: bool = False,
    iterations: int = 1,
    iteration_strategy: str = "worker_pool",
) -> Dict[str, int]:
    """
    Build prompts (without API calls), estimate tokens before running.
    Returns a dict of totals accounting for iterations.
    """
    total_workers_calls = 0
    total_supervisor_calls = 0
    total_synthesizer_calls = 0
    total_in = total_out = 0

    worker_models = [get_model_name(w) for w in llms["workers"]]
    supervisor_model = get_model_name(llms["supervisor"])
    synthesizer_model = get_model_name(llms["synthesizer"])
    
    # Iteration multipliers for different strategies
    if iterations > 1:
        # Worker calls multiply by iterations (may vary by strategy)
        worker_iteration_multiplier = iterations
        
        # Supervisor calls - slightly more complex prompt for iterative mode
        supervisor_iteration_multiplier = iterations
        supervisor_token_overhead = 0.2  # 20% more tokens for iteration context
        
        # Synthesizer - more complex for cross-iteration synthesis
        synthesizer_token_overhead = 0.3 if iterations > 1 else 0.0
    else:
        worker_iteration_multiplier = 1
        supervisor_iteration_multiplier = 1
        supervisor_token_overhead = 0.0
        synthesizer_token_overhead = 0.0

    for fp in files:
        # Read code and chunk like the real run
        try:
            with open(fp, "r", encoding="utf-8") as f:
                code = f.read()
        except Exception:
            continue
        language = detect_language_from_extension(os.path.splitext(fp)[1])
        lines = code.splitlines()
        chunks = chunk_code_by_lines(code, max_lines=chunk_size) if len(
            lines) > chunk_threshold else [(1, code)]
        total_chunks = len(chunks)

        # Workers & supervisor per chunk (with iteration multipliers)
        for idx, (start, chunk_text) in enumerate(chunks, start=1):
            numbered = add_line_numbers_preserve(chunk_text, start_line=start)
            
            # Workers (multiply by iterations)
            for wm in worker_models:
                prompt_text = render_worker_prompt_text(
                    language=language, file_path=fp, chunk_index=idx, total_chunks=total_chunks,
                    code_with_line_numbers=numbered, linus_mode=linus_mode
                )
                worker_tokens_in = count_tokens_text(wm, prompt_text)
                
                # Account for iterations
                total_in += worker_tokens_in * worker_iteration_multiplier
                total_out += assume_worker_out * worker_iteration_multiplier
                total_workers_calls += worker_iteration_multiplier
                
            # Supervisor (with iteration overhead)
            sup_static = count_tokens_text(
                supervisor_model, render_supervisor_prompt_text(reviews_text_block=""))
            sup_base_in = sup_static + sum([assume_worker_out for _ in worker_models])
            
            # Apply iteration multiplier and overhead
            sup_in = sup_base_in * supervisor_iteration_multiplier * (1 + supervisor_token_overhead)
            total_in += int(sup_in)
            total_out += assume_supervisor_out * supervisor_iteration_multiplier
            total_supervisor_calls += supervisor_iteration_multiplier

        # Synthesizer per file (with iteration synthesis complexity)
        synth_static = count_tokens_text(
            synthesizer_model, render_synthesizer_prompt_text(chunk_summaries_jsonl=""))
        # Conservative per-chunk summary JSONL size (larger for iterations)
        base_chunk_size = 150
        if iterations > 1:
            # More complex synthesis across iterations
            base_chunk_size = int(base_chunk_size * (1.5 + 0.2 * iterations))
            
        synth_in = synth_static + total_chunks * base_chunk_size
        synth_in = int(synth_in * (1 + synthesizer_token_overhead))
        
        total_in += synth_in
        total_out += int(assume_synthesizer_out * (1 + synthesizer_token_overhead))
        total_synthesizer_calls += 1

    return dict(
        worker_calls=total_workers_calls,
        supervisor_calls=total_supervisor_calls,
        synthesizer_calls=total_synthesizer_calls,
        tokens_in=total_in,
        tokens_out=total_out,
        tokens_total=total_in + total_out,
    )


async def main():
    parser = argparse.ArgumentParser(
        description="AI Multi-Agent Code Review Tool (Improved)")
    parser.add_argument("directory", type=str, help="Root directory to review")
    parser.add_argument("--extensions", nargs="+",
                        default=[".cpp", ".hpp", ".h", ".cxx", ".hxx", ".cc", ".c++", ".h++", ".py"], help="File extensions to include")
    parser.add_argument("--save-dir", type=str,
                        default="reviews", help="Directory to save results")
    parser.add_argument("--models", type=str, default="models.yaml",
                        help="Optional models config file")
    parser.add_argument("--chunk-size", type=int,
                        default=DEFAULT_CHUNK_SIZE, help="Max lines per chunk")
    parser.add_argument("--chunk-threshold", type=int, default=DEFAULT_MAX_LINES,
                        help="If file exceeds this many lines, chunk it")
    parser.add_argument("--assume-worker-out", type=int, default=900,
                        help="Assumed max output tokens per worker review")
    parser.add_argument("--assume-supervisor-out", type=int, default=400,
                        help="Assumed max output tokens per supervisor decision")
    parser.add_argument("--assume-synthesizer-out", type=int, default=700,
                        help="Assumed max output tokens per synthesizer result")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Run without token preflight/approval")
    parser.add_argument("--linus-mode", action="store_true",
                        help="Use Linus Torvalds review style with systematic analysis framework")
    parser.add_argument("--iterations", type=int, default=1, 
                        help="Number of iterations for multi-agent review (1-5, default: 1)")
    parser.add_argument("--iteration-strategy", choices=["worker_pool", "feedback_driven", "consensus"],
                        default="worker_pool",
                        help="Strategy for iterative improvement: worker_pool (different models), feedback_driven (supervisor feedback), or consensus (peer review sharing)")
    parser.add_argument("--convergence-threshold", type=float, default=0.8,
                        help="Convergence threshold for early stopping (0.0-1.0, default: 0.8)")
    parser.add_argument("--retain-full-iteration-data", action="store_true",
                        help="Retain full iteration data (increases memory usage, useful for debugging)")
    
    # Rate limiting arguments
    parser.add_argument("--rate-limit-mode", choices=["conservative", "balanced", "aggressive", "custom"], 
                        default="balanced",
                        help="Rate limiting mode: conservative (safe defaults), balanced (optimal), aggressive (high throughput), or custom (use config file)")
    parser.add_argument("--rate-limit-config", type=str,
                        help="Path to custom rate limit configuration file (YAML)")
    parser.add_argument("--max-concurrent", type=int,
                        help="Maximum concurrent requests (overrides mode defaults)")
    parser.add_argument("--disable-rate-limiting", action="store_true",
                        help="Disable rate limiting (not recommended for production)")

    args = parser.parse_args()

    # Validate iteration arguments
    if args.iterations < 1 or args.iterations > 5:
        console.print("[red]Error: --iterations must be between 1 and 5[/red]")
        return
    
    if args.convergence_threshold < 0.0 or args.convergence_threshold > 1.0:
        console.print("[red]Error: --convergence-threshold must be between 0.0 and 1.0[/red]")
        return

    console.print(
        "[bold cyan]🤖 Initializing Improved Argus...[/bold cyan]")

    # Env: load project .env (nearest), then Home-level fallbacks (do not override existing)
    load_dotenv(find_dotenv(usecwd=True))
    for p in [Path.home()/".argus.env", Path.home()/".env", Path.home()/".config/argus/env"]:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        console.print(
            "[red]Error: OPENAI_API_KEY not found. Set it in your .env.[/red]")
        return

    # Initialize rate limiter
    if not args.disable_rate_limiting:
        try:
            if args.rate_limit_mode == "custom" and args.rate_limit_config:
                # Load custom rate limit configuration
                import yaml
                with open(args.rate_limit_config, "r") as f:
                    custom_config = yaml.safe_load(f)
                    from rate_limiting import RateLimitConfig
                    # Convert dict to RateLimitConfig objects
                    config_objects = {}
                    for model_name, config_dict in custom_config.get("rate_limits", {}).items():
                        config_objects[model_name] = RateLimitConfig(**config_dict)
                    rate_limiter = APIRateLimiter(mode=RateLimitMode.CUSTOM, custom_config=config_objects)
            else:
                mode = RateLimitMode(args.rate_limit_mode)
                rate_limiter = APIRateLimiter(mode=mode)
            
            set_global_rate_limiter(rate_limiter)
            console.print(f"[dim]Rate limiting enabled: {args.rate_limit_mode} mode[/dim]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to initialize rate limiter: {e}[/yellow]")
            console.print("[yellow]Continuing without rate limiting...[/yellow]")
    else:
        console.print("[dim]Rate limiting disabled[/dim]")

    # LLMs
    models_cfg = load_models_config(args.models)
    llms = build_llms(models_cfg)

    # Files to review
    files = list_source_files(args.directory, args.extensions)
    if not files:
        console.print(
            f"[yellow]No files with {args.extensions} under '{args.directory}'.[/yellow]"
        )
        return

    # ----- Preflight estimation & approval -----
    if not args.skip_preflight:
        est = preflight_estimate(
            llms=llms,
            files=files,
            chunk_size=args.chunk_size,
            chunk_threshold=args.chunk_threshold,
            assume_worker_out=args.assume_worker_out,
            assume_supervisor_out=args.assume_supervisor_out,
            assume_synthesizer_out=args.assume_synthesizer_out,
            linus_mode=args.linus_mode,
            iterations=args.iterations,
            iteration_strategy=args.iteration_strategy,
        )
        # Show iteration info if applicable
        title = "Token Preflight (approx.)"
        if args.iterations > 1:
            title += f" - {args.iterations} iterations ({args.iteration_strategy})"
            
        table = Table(title=title)
        table.add_column("Stage")
        table.add_column("Calls", justify="right")
        table.add_column("Input toks", justify="right")
        table.add_column("Output toks", justify="right")
        table.add_row("Workers", str(
            est["worker_calls"]), "-", str(args.assume_worker_out * est["worker_calls"]))
        table.add_row("Supervisor", str(
            est["supervisor_calls"]), "-", str(args.assume_supervisor_out * est["supervisor_calls"]))
        table.add_row("Synthesizer", str(est["synthesizer_calls"]), "-", str(
            args.assume_synthesizer_out * est["synthesizer_calls"]))
        table.add_row("—", "—", "—", "—")
        table.add_row("Totals", "", str(
            est["tokens_in"]), str(est["tokens_out"]))
        console.print(table)
        
        note = "[dim]Note: counts approximate; small chat overhead not included. Adjust --assume-* flags if needed."
        if args.iterations > 1:
            note += f" Iterations multiply token usage approximately {args.iterations}x."
        note += "[/dim]"
        console.print(note)
        
        # Show rate limiting information
        if not args.disable_rate_limiting:
            from agents import get_global_rate_limiter
            rate_limiter = get_global_rate_limiter()
            if rate_limiter:
                console.print("\n[bold cyan]Rate Limiting Status:[/bold cyan]")
                status = rate_limiter.get_status()
                for model_name, model_status in status.items():
                    requests_limit = rate_limiter._get_model_config(model_name).requests_per_minute
                    tokens_limit = rate_limiter._get_model_config(model_name).tokens_per_minute
                    max_concurrent = rate_limiter._get_model_config(model_name).max_concurrent
                    
                    console.print(f"  [dim]{model_name}:[/dim] {requests_limit} RPM, {tokens_limit:,} TPM, {max_concurrent} concurrent")
                    
                console.print(f"[dim]Mode: {args.rate_limit_mode} | Circuit breakers active | Automatic backoff enabled[/dim]")

        # Ask for approval
        console.print("\n[bold]Proceed with API calls?[/bold] [y/N]: ", end="")
        answer = (await _async_input()).strip().lower()
        if answer not in ("y", "yes"):
            console.print(
                "[yellow]Aborted by user before any API calls.[/yellow]")
            return

    console.print(f"Found {len(files)} file(s).")

    for fp in files:
        await review_single_file(
            llms=llms,
            file_path=fp,
            save_dir=args.save_dir,
            chunk_size=args.chunk_size,
            max_lines_before_chunk=args.chunk_threshold,
            linus_mode=args.linus_mode,
            iterations=args.iterations,
            iteration_strategy=args.iteration_strategy,
            convergence_threshold=args.convergence_threshold,
            retain_full_iteration_data=args.retain_full_iteration_data
        )
    
    # Show final rate limiting statistics
    if not args.disable_rate_limiting:
        from agents import get_global_rate_limiter
        rate_limiter = get_global_rate_limiter()
        if rate_limiter:
            console.print("\n[bold cyan]Rate Limiting Summary:[/bold cyan]")
            stats = rate_limiter.get_stats()
            
            total_requests = sum(stat.total_requests for stat in stats.values())
            total_successful = sum(stat.successful_requests for stat in stats.values())
            total_rate_limited = sum(stat.rate_limited_requests for stat in stats.values())
            
            console.print(f"  Total API calls: {total_requests}")
            console.print(f"  Successful: {total_successful}")
            console.print(f"  Rate limited: {total_rate_limited}")
            
            if total_requests > 0:
                success_rate = (total_successful / total_requests) * 100
                console.print(f"  Success rate: {success_rate:.1f}%")
                
                if total_rate_limited > 0:
                    console.print(f"[yellow]  {total_rate_limited} requests were rate limited and retried[/yellow]")
            
            # Show per-model statistics if multiple models used
            if len(stats) > 1:
                for model_name, stat in stats.items():
                    if stat.total_requests > 0:
                        avg_time = stat.average_response_time
                        console.print(f"  [dim]{model_name}: {stat.successful_requests}/{stat.total_requests} calls, avg {avg_time:.1f}s[/dim]")

if __name__ == "__main__":
    asyncio.run(main())
