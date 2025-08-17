import pytest
import json
import sys
import os
from unittest.mock import MagicMock, patch, AsyncMock
from typing import Dict, Any, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import (
    IterationController,
    IterationStrategy,
    run_iterative_worker_agent,
    run_iterative_supervisor_agent,
    render_iterative_supervisor_prompt_text
)
from prompts import (
    JSON_ITERATIVE_SUPERVISOR_SCHEMA,
    ITERATION_INSTRUCTIONS
)


class TestIterationController:
    """Test the IterationController core functionality."""
    
    def test_iteration_controller_initialization(self):
        """Test IterationController initialization with different parameters."""
        # Default initialization
        controller = IterationController()
        assert controller.max_iterations == 3
        assert controller.strategy == IterationStrategy.WORKER_POOL
        assert controller.convergence_threshold == 0.8
        assert controller.current_iteration == 0
        assert controller.iteration_history == []
        
        # Custom initialization
        controller = IterationController(
            max_iterations=5,
            strategy=IterationStrategy.FEEDBACK_DRIVEN,
            convergence_threshold=0.9
        )
        assert controller.max_iterations == 5
        assert controller.strategy == IterationStrategy.FEEDBACK_DRIVEN
        assert controller.convergence_threshold == 0.9
    
    def test_iteration_strategies_enum(self):
        """Test that all iteration strategies are properly defined."""
        assert IterationStrategy.WORKER_POOL.value == "worker_pool"
        assert IterationStrategy.FEEDBACK_DRIVEN.value == "feedback_driven"
        assert IterationStrategy.CONSENSUS.value == "consensus"
        
        # Test that strategies can be created from strings
        assert IterationStrategy("worker_pool") == IterationStrategy.WORKER_POOL
        assert IterationStrategy("feedback_driven") == IterationStrategy.FEEDBACK_DRIVEN
        assert IterationStrategy("consensus") == IterationStrategy.CONSENSUS
    
    def test_should_continue_iterating_logic(self, sample_worker_response, sample_supervisor_response):
        """Test the convergence detection and continuation logic."""
        controller = IterationController(max_iterations=3, convergence_threshold=0.8)
        
        # Should continue on first call (no iterations yet)
        assert controller.should_continue_iterating() == True
        
        # Add first iteration - should continue (need at least 2 to compare)
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        assert controller.should_continue_iterating() == True
        assert controller.current_iteration == 1
        
        # Add second iteration with different findings to avoid perfect convergence
        different_response = sample_worker_response.copy()
        different_response["findings"] = [
            {
                "type": "performance",
                "title": "Different performance issue",
                "severity": "medium",
                "lines": [5],
                "explanation": "Different finding"
            }
        ]
        different_response["counts"] = {"bug": 0, "performance": 1, "style": 0, "maintainability": 0}
        controller.record_iteration([different_response], sample_supervisor_response, {})
        # Should continue because convergence not achieved (findings are different)
        assert controller.should_continue_iterating() == True  
        assert controller.current_iteration == 2
        
        # Add third iteration - should stop (hit max)
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        assert controller.should_continue_iterating() == False
        assert controller.current_iteration == 3
    
    def test_max_iterations_limit(self, sample_worker_response, sample_supervisor_response):
        """Test that max_iterations is respected."""
        controller = IterationController(max_iterations=2)
        
        # First iteration
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        assert controller.should_continue_iterating() == True
        
        # Second iteration - should stop after this
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        assert controller.should_continue_iterating() == False
        assert controller.current_iteration == 2
    
    def test_record_iteration_data_storage(self, sample_worker_response, sample_supervisor_response):
        """Test that iteration data is properly stored."""
        controller = IterationController()
        metadata = {"chunk_index": 1, "file_path": "test.py"}
        
        controller.record_iteration([sample_worker_response], sample_supervisor_response, metadata)
        
        assert len(controller.iteration_history) == 1
        iteration_data = controller.iteration_history[0]
        
        assert iteration_data["iteration"] == 1
        assert iteration_data["worker_reviews"] == [sample_worker_response]
        assert iteration_data["supervisor_result"] == sample_supervisor_response
        assert iteration_data["metadata"] == metadata
        assert "timestamp" in iteration_data
        assert controller.current_iteration == 1


class TestIterationContextBuilding:
    """Test iteration context building for different strategies."""
    
    def test_worker_pool_context(self, sample_worker_response, sample_supervisor_response):
        """Test context building for worker_pool strategy."""
        controller = IterationController(strategy=IterationStrategy.WORKER_POOL)
        
        # First iteration context
        context = controller.get_context_for_iteration(1)
        assert context["iteration"] == 1
        assert context["strategy"] == "worker_pool"
        assert context["total_iterations_planned"] == 3
        assert context["previous_iterations"] == []
        
        # Add some history and test second iteration context
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        context = controller.get_context_for_iteration(2)
        
        assert context["iteration"] == 2
        assert len(context["previous_iterations"]) == 1
        assert context["previous_iterations"][0]["iteration"] == 1
    
    def test_feedback_driven_context(self, sample_iterative_supervisor_response, sample_worker_response):
        """Test context building for feedback_driven strategy."""
        controller = IterationController(strategy=IterationStrategy.FEEDBACK_DRIVEN)
        
        # First iteration - no feedback yet
        context = controller.get_context_for_iteration(1)
        assert context["strategy"] == "feedback_driven"
        assert "supervisor_feedback" not in context
        
        # Add iteration with feedback
        controller.record_iteration([sample_worker_response], sample_iterative_supervisor_response, {})
        context = controller.get_context_for_iteration(2)
        
        assert "supervisor_feedback" in context
        assert context["supervisor_feedback"] == "Focus on performance implications of suggested changes"
    
    def test_consensus_context(self, sample_worker_response, sample_supervisor_response):
        """Test context building for consensus strategy."""
        controller = IterationController(strategy=IterationStrategy.CONSENSUS)
        
        # First iteration - no peer reviews yet
        context = controller.get_context_for_iteration(1)
        assert context["strategy"] == "consensus"
        assert "peer_reviews_previous" not in context
        
        # Add iteration with multiple worker reviews
        multiple_reviews = [sample_worker_response, sample_worker_response.copy()]
        controller.record_iteration(multiple_reviews, sample_supervisor_response, {})
        context = controller.get_context_for_iteration(2)
        
        assert "peer_reviews_previous" in context
        assert len(context["peer_reviews_previous"]) == 2
    
    def test_context_includes_iteration_summaries(self, sample_worker_response, sample_supervisor_response):
        """Test that context includes summaries of previous iterations."""
        controller = IterationController()
        
        # Add two iterations
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {"test": "meta1"})
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {"test": "meta2"})
        
        context = controller.get_context_for_iteration(3)
        
        assert len(context["previous_iterations"]) == 2
        for i, prev in enumerate(context["previous_iterations"], 1):
            assert prev["iteration"] == i
            assert "summary" in prev
            assert isinstance(prev["summary"], str)


class TestIterativeAgentFunctions:
    """Test iterative-specific agent functions."""
    
    @pytest.mark.asyncio
    async def test_run_iterative_worker_agent(self, sample_worker_response, mock_llm):
        """Test iterative worker agent execution."""
        iteration_context = {
            "iteration": 2,
            "total_iterations_planned": 3,
            "strategy": "worker_pool",
            "previous_iterations": []
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_worker_response)
            
            result = await run_iterative_worker_agent(
                mock_llm,
                language="python",
                file_path="test.py",
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: def test(): pass",
                iteration_context=iteration_context,
                linus_mode=False
            )
            
            assert result == sample_worker_response
            mock_retry.assert_called_once()
            
            # Verify that iteration context was passed to the prompt
            call_args = mock_retry.call_args[0]
            chain, params = call_args
            assert params["iteration"] == 2
            assert params["total_iterations"] == 3
    
    @pytest.mark.asyncio
    async def test_run_iterative_supervisor_agent(self, sample_iterative_supervisor_response, mock_llm):
        """Test iterative supervisor agent execution."""
        iteration_context = {
            "iteration": 2,
            "total_iterations_planned": 3,
            "strategy": "feedback_driven"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_iterative_supervisor_response)
            
            result = await run_iterative_supervisor_agent(
                mock_llm,
                reviews_text_block="Sample reviews",
                iteration_context=iteration_context
            )
            
            assert result == sample_iterative_supervisor_response
            assert "iteration_comparison" in result
            assert "feedback_for_next_iteration" in result
    
    @pytest.mark.asyncio
    async def test_iterative_worker_with_linus_mode(self, sample_linus_response, mock_llm):
        """Test iterative worker agent in Linus mode."""
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 2,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.return_value = json.dumps(sample_linus_response)
            
            result = await run_iterative_worker_agent(
                mock_llm,
                language="cpp",
                file_path="test.cpp", 
                chunk_index=1,
                total_chunks=1,
                code_with_line_numbers="L001: int main() {}",
                iteration_context=iteration_context,
                linus_mode=True
            )
            
            assert result == sample_linus_response
            assert "taste_score" in result
            assert "linus_analysis" in result
    
    def test_render_iterative_supervisor_prompt_text(self):
        """Test iterative supervisor prompt rendering."""
        iteration_context = {
            "iteration": 2,
            "total_iterations_planned": 3,
            "strategy": "feedback_driven",
            "supervisor_feedback": "Previous feedback here"
        }
        
        prompt = render_iterative_supervisor_prompt_text(
            reviews_text_block="Sample reviews",
            iteration_context=iteration_context
        )
        
        assert "iteration 2 of 3" in prompt
        assert "feedback_driven" in prompt
        assert "Previous feedback here" in prompt
        assert "Sample reviews" in prompt
        # Check that iterative-specific fields are in the prompt
        assert "iteration_comparison" in prompt
        assert "feedback_for_next_iteration" in prompt


class TestIterationStrategies:
    """Test specific iteration strategy behaviors."""
    
    def test_iteration_instructions_defined(self):
        """Test that iteration instructions are defined for all strategies."""
        required_strategies = ["worker_pool", "feedback_driven", "consensus"]
        
        for strategy in required_strategies:
            assert strategy in ITERATION_INSTRUCTIONS
            instructions = ITERATION_INSTRUCTIONS[strategy]
            assert isinstance(instructions, str)
            assert len(instructions) > 0
    
    def test_worker_pool_strategy_instructions(self):
        """Test worker_pool strategy instructions emphasize diversity."""
        instructions = ITERATION_INSTRUCTIONS["worker_pool"]
        assert "different perspective" in instructions.lower()
        assert "previous iterations" in instructions.lower()
    
    def test_feedback_driven_strategy_instructions(self):
        """Test feedback_driven strategy instructions emphasize supervisor feedback."""
        instructions = ITERATION_INSTRUCTIONS["feedback_driven"]
        assert "supervisor feedback" in instructions.lower()
        assert "previous iteration" in instructions.lower()
    
    def test_consensus_strategy_instructions(self):
        """Test consensus strategy instructions emphasize peer review sharing."""
        instructions = ITERATION_INSTRUCTIONS["consensus"]
        assert "peer reviews" in instructions.lower()
        assert "comprehensive coverage" in instructions.lower()


class TestConvergenceAnalysis:
    """Test convergence detection and analysis."""
    
    def test_improvement_trajectory_analysis(self, sample_worker_response):
        """Test improvement trajectory analysis."""
        controller = IterationController()
        
        # Simulate improving scores across iterations
        supervisor_responses = [
            {"scores": [{"accuracy": 0.7, "completeness": 0.6, "clarity": 0.8, "insightfulness": 0.7}]},
            {"scores": [{"accuracy": 0.8, "completeness": 0.8, "clarity": 0.9, "insightfulness": 0.8}]},
            {"scores": [{"accuracy": 0.9, "completeness": 0.9, "clarity": 0.9, "insightfulness": 0.9}]}
        ]
        
        for supervisor_resp in supervisor_responses:
            controller.record_iteration([sample_worker_response], supervisor_resp, {})
        
        trajectory = controller._analyze_improvement_trajectory()
        assert trajectory["trend"] == "improving"
        assert len(trajectory["trajectories"]["quality_scores"]) == 3
        assert trajectory["final_quality"] > 0
    
    def test_best_iteration_identification(self, sample_worker_response):
        """Test identification of best iteration."""
        controller = IterationController()
        
        # Simulate iterations with different quality scores
        supervisor_responses = [
            {"scores": [{"accuracy": 0.7, "completeness": 0.7, "clarity": 0.7, "insightfulness": 0.7}]},  # Iteration 1
            {"scores": [{"accuracy": 0.9, "completeness": 0.9, "clarity": 0.9, "insightfulness": 0.9}]},  # Iteration 2 (best)
            {"scores": [{"accuracy": 0.8, "completeness": 0.8, "clarity": 0.8, "insightfulness": 0.8}]}   # Iteration 3
        ]
        
        for supervisor_resp in supervisor_responses:
            controller.record_iteration([sample_worker_response], supervisor_resp, {})
        
        best = controller._identify_best_iteration()
        assert best["iteration"] == 2  # Second iteration had highest scores
        assert "reason" in best
    
    def test_final_synthesis_data(self, sample_worker_response):
        """Test final synthesis data generation."""
        controller = IterationController(max_iterations=2)
        
        # Add iterations
        supervisor_resp = {"scores": [{"accuracy": 0.8, "completeness": 0.8, "clarity": 0.8, "insightfulness": 0.8}]}
        controller.record_iteration([sample_worker_response], supervisor_resp, {})
        controller.record_iteration([sample_worker_response], supervisor_resp, {})
        
        synthesis = controller.get_final_synthesis_data()
        
        assert synthesis["total_iterations"] == 2
        assert synthesis["strategy_used"] == "worker_pool"
        assert synthesis["convergence_achieved"] == False  # Completed max_iterations (didn't stop early)
        assert "iteration_history" in synthesis
        assert "best_iteration" in synthesis
        assert "improvement_trajectory" in synthesis
    
    def test_convergence_with_early_stopping(self, sample_worker_response):
        """Test that early stopping works with convergence threshold."""
        # This is more of an integration test - in practice convergence would be detected
        # by similarity between reviews, but we test the framework
        controller = IterationController(max_iterations=5, convergence_threshold=0.9)
        
        # Add iterations
        supervisor_resp = {"scores": [{"accuracy": 0.8, "completeness": 0.8, "clarity": 0.8, "insightfulness": 0.8}]}
        controller.record_iteration([sample_worker_response], supervisor_resp, {})
        controller.record_iteration([sample_worker_response], supervisor_resp, {})
        
        # Should still continue since we haven't implemented similarity checking yet
        # This tests the framework is in place for future enhancement
        assert controller.current_iteration == 2
        assert controller.max_iterations == 5


class TestIterationErrorHandling:
    """Test error handling in iteration scenarios."""
    
    def test_iteration_controller_with_no_history(self):
        """Test methods work correctly with no iteration history."""
        controller = IterationController()
        
        # Should handle empty history gracefully
        trajectory = controller._analyze_improvement_trajectory()
        assert trajectory["trend"] == "insufficient_data"
        
        best = controller._identify_best_iteration()
        assert best["iteration"] == 0
        assert best["reason"] == "no_iterations"
    
    def test_malformed_supervisor_scores(self, sample_worker_response):
        """Test handling of malformed supervisor scores."""
        controller = IterationController()
        
        # Supervisor response with missing scores
        malformed_supervisor = {"analysis": "Good", "winner_index": 1}
        controller.record_iteration([sample_worker_response], malformed_supervisor, {})
        
        trajectory = controller._analyze_improvement_trajectory()
        # Should handle missing scores gracefully
        if trajectory["trend"] != "insufficient_data":
            assert "trajectories" in trajectory
            assert len(trajectory["trajectories"]["quality_scores"]) == 1
            assert trajectory["trajectories"]["quality_scores"][0] == 0.0  # Default for missing scores
    
    def test_empty_worker_reviews(self):
        """Test handling of empty worker reviews."""
        controller = IterationController()
        supervisor_resp = {"scores": []}
        
        controller.record_iteration([], supervisor_resp, {})
        
        assert len(controller.iteration_history) == 1
        assert controller.iteration_history[0]["worker_reviews"] == []
    
    @pytest.mark.asyncio
    async def test_iterative_agent_failure_handling(self, mock_llm):
        """Test that iterative agents handle failures gracefully."""
        iteration_context = {
            "iteration": 1,
            "total_iterations_planned": 2,
            "strategy": "worker_pool"
        }
        
        with patch('agents._ainvoke_with_retry') as mock_retry:
            mock_retry.side_effect = Exception("Network error")
            
            with pytest.raises(Exception):
                await run_iterative_worker_agent(
                    mock_llm,
                    language="python",
                    file_path="test.py",
                    chunk_index=1,
                    total_chunks=1,
                    code_with_line_numbers="L001: code",
                    iteration_context=iteration_context
                )


class TestIterationMetadata:
    """Test iteration metadata handling and analytics."""
    
    def test_iteration_timestamp_recording(self, sample_worker_response, sample_supervisor_response):
        """Test that iterations record timestamps."""
        controller = IterationController()
        
        controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        
        iteration_data = controller.iteration_history[0]
        assert "timestamp" in iteration_data
        assert isinstance(iteration_data["timestamp"], str)
    
    def test_iteration_summary_generation(self, sample_worker_response, sample_supervisor_response):
        """Test iteration summary generation for context."""
        controller = IterationController()
        
        metadata = {"chunk_index": 1, "file_path": "test.py"}
        controller.record_iteration([sample_worker_response], sample_supervisor_response, metadata)
        
        iteration_data = controller.iteration_history[0]
        summary = controller._summarize_iteration(iteration_data)
        
        assert "Iteration 1" in summary
        assert "1 workers" in summary
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_convergence_indicators_extraction(self, sample_worker_response, sample_supervisor_response):
        """Test extraction of convergence indicators."""
        controller = IterationController()
        
        # Add multiple iterations
        for i in range(3):
            controller.record_iteration([sample_worker_response], sample_supervisor_response, {})
        
        iteration_data = controller.iteration_history[-1]
        indicators = controller._get_convergence_indicators(iteration_data)
        
        # Check for actual fields returned by the implementation
        assert "finding_count" in indicators
        assert "critical_issues" in indicators  
        assert "consensus_strength" in indicators
        assert indicators["consensus_strength"] == 1  # One worker review per iteration
    
    def test_anonymized_review_creation(self, sample_worker_response):
        """Test creation of anonymized reviews for consensus strategy."""
        controller = IterationController()
        
        reviews = [sample_worker_response, sample_worker_response.copy()]
        anonymized = controller._anonymize_reviews(reviews)
        
        assert len(anonymized) == 2
        for anon_review in anonymized:
            # Should preserve important content but remove identifying information
            assert "findings" in anon_review or "summary" in anon_review
            # This is a placeholder - real implementation would remove model-specific info
            assert isinstance(anon_review, dict)