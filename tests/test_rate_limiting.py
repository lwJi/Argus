import pytest
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from rate_limiting import (
    APIRateLimiter, RateLimitMode, RateLimitConfig, TokenBucket, CircuitBreaker
)


class TestTokenBucket:
    """Test the token bucket algorithm implementation."""
    
    def test_token_bucket_initialization(self):
        """Test token bucket is initialized correctly."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        assert bucket.capacity == 10
        assert bucket.tokens == 10  # Should start full
        assert bucket.refill_rate == 5.0
    
    def test_token_consumption(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(capacity=10, refill_rate=5.0)
        
        # Should be able to consume available tokens
        assert bucket.consume(5) == True
        assert bucket.consume(5) == True
        
        # Should not be able to consume more than available
        assert bucket.consume(1) == False
    
    def test_token_refill(self):
        """Test token bucket refills over time."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens per second
        
        # Consume all tokens
        assert bucket.consume(10) == True
        assert bucket.consume(1) == False
        
        # Wait and check refill (simulated by adjusting last_refill)
        bucket.last_refill -= 0.5  # Simulate 0.5 seconds ago
        assert bucket.consume(5) == True  # Should have ~5 tokens available
    
    def test_available_tokens(self):
        """Test checking available tokens without consuming."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)
        
        # Initially full
        assert bucket.available_tokens() == 10
        
        # After consuming some
        bucket.consume(3)
        available = bucket.available_tokens()
        assert abs(available - 7) < 0.1  # Allow for small floating point differences
    
    def test_time_until_tokens(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=10.0)  # 10 tokens/sec
        
        # Consume all tokens
        bucket.consume(10)
        
        # Should need to wait ~0.5 seconds for 5 tokens
        wait_time = bucket.time_until_tokens(5)
        assert 0.4 <= wait_time <= 0.6


class TestCircuitBreaker:
    """Test the circuit breaker implementation."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60.0)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        assert cb.can_execute() == True
    
    def test_circuit_breaker_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        # Record failures up to threshold
        cb.call_failed()
        assert cb.state == "CLOSED"
        cb.call_failed()
        assert cb.state == "CLOSED"
        cb.call_failed()
        assert cb.state == "OPEN"
        assert cb.can_execute() == False
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)  # Short timeout for testing
        
        # Trip the circuit breaker
        cb.call_failed()
        cb.call_failed()
        assert cb.state == "OPEN"
        
        # Wait for timeout
        time.sleep(0.2)
        assert cb.can_execute() == True
        assert cb.state == "HALF_OPEN"
        
        # Success should close it
        cb.call_succeeded()
        assert cb.state == "CLOSED"
    
    def test_circuit_breaker_success_reset(self):
        """Test successful calls reset failure count."""
        cb = CircuitBreaker(failure_threshold=3, timeout=60.0)
        
        # Record some failures
        cb.call_failed()
        cb.call_failed()
        assert cb.failure_count == 2
        
        # Success should reset
        cb.call_succeeded()
        assert cb.failure_count == 0


class TestAPIRateLimiter:
    """Test the main API rate limiter functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes with correct modes."""
        # Test different modes
        conservative = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        assert conservative.mode == RateLimitMode.CONSERVATIVE
        
        balanced = APIRateLimiter(mode=RateLimitMode.BALANCED)
        assert balanced.mode == RateLimitMode.BALANCED
        
        aggressive = APIRateLimiter(mode=RateLimitMode.AGGRESSIVE)
        assert aggressive.mode == RateLimitMode.AGGRESSIVE
    
    def test_custom_configuration(self):
        """Test rate limiter with custom configuration."""
        custom_config = {
            "test-model": RateLimitConfig(
                requests_per_minute=100,
                tokens_per_minute=10000,
                max_concurrent=5
            )
        }
        
        limiter = APIRateLimiter(mode=RateLimitMode.CUSTOM, custom_config=custom_config)
        assert limiter.mode == RateLimitMode.CUSTOM
        assert limiter.custom_config == custom_config
    
    @pytest.mark.asyncio
    async def test_acquire_permission(self):
        """Test acquiring permission for API calls."""
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        
        # Should be able to acquire initially
        can_proceed, wait_time = await limiter.acquire(mock_llm, estimated_tokens=100)
        assert can_proceed == True
        assert wait_time is None
    
    @pytest.mark.asyncio
    async def test_semaphore_acquisition(self):
        """Test semaphore acquisition for concurrency control."""
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        
        # Should be able to acquire semaphore
        semaphore = await limiter.acquire_with_semaphore(mock_llm, estimated_tokens=100)
        assert semaphore is not None
        
        # Clean up
        semaphore.release()
    
    def test_record_success(self):
        """Test recording successful API calls."""
        limiter = APIRateLimiter(mode=RateLimitMode.BALANCED)
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        
        # Record success
        limiter.record_success(mock_llm, response_time=1.5, actual_tokens=150)
        
        # Check statistics
        stats = limiter.get_stats("gpt-4o")
        assert stats["gpt-4o"].successful_requests == 1
        assert stats["gpt-4o"].average_response_time == 1.5
    
    def test_record_failure(self):
        """Test recording failed API calls."""
        limiter = APIRateLimiter(mode=RateLimitMode.BALANCED)
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        
        # Record failure
        error = Exception("Rate limit exceeded (429)")
        limiter.record_failure(mock_llm, error)
        
        # Check statistics
        stats = limiter.get_stats("gpt-4o")
        assert stats["gpt-4o"].rate_limited_requests == 1
    
    def test_get_status(self):
        """Test getting detailed rate limiter status."""
        limiter = APIRateLimiter(mode=RateLimitMode.BALANCED)
        
        # Initialize for a model
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        asyncio.run(limiter.acquire(mock_llm))
        
        status = limiter.get_status()
        assert "gpt-4o-mini" in status
        assert "requests_available" in status["gpt-4o-mini"]
        assert "tokens_available" in status["gpt-4o-mini"]
        assert "circuit_breaker_state" in status["gpt-4o-mini"]
    
    @pytest.mark.asyncio
    async def test_wait_for_capacity(self):
        """Test waiting for sufficient capacity."""
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        
        # Should be able to get capacity immediately when not rate limited
        result = await limiter.wait_for_capacity(mock_llm, estimated_tokens=100, max_wait=1.0)
        assert result == True
    
    def test_model_name_extraction(self):
        """Test extracting model names from LLM instances."""
        limiter = APIRateLimiter(mode=RateLimitMode.BALANCED)
        
        # Test different attribute names
        mock_llm1 = MagicMock()
        mock_llm1.model = "gpt-4o"
        assert limiter._get_model_name(mock_llm1) == "gpt-4o"
        
        mock_llm2 = MagicMock()
        mock_llm2.model = None
        mock_llm2.model_name = "gpt-4o-mini"
        assert limiter._get_model_name(mock_llm2) == "gpt-4o-mini"
        
        # Test fallback to default
        mock_llm3 = MagicMock()
        mock_llm3.model = None
        mock_llm3.model_name = None
        mock_llm3.model_id = None
        assert limiter._get_model_name(mock_llm3) == "default"


class TestRateLimitingIntegration:
    """Test rate limiting integration with agents."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_mock_llm(self):
        """Test rate limiting integration with mock LLM calls."""
        from agents import set_global_rate_limiter, _ainvoke_with_retry
        
        # Set up rate limiter
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        set_global_rate_limiter(limiter)
        
        # Mock chain and LLM
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value="Test response")
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o-mini"
        
        # Should work with rate limiting
        result = await _ainvoke_with_retry(
            mock_chain, 
            {"prompt_text": "Test prompt"}, 
            llm=mock_llm
        )
        
        assert result == "Test response"
        mock_chain.ainvoke.assert_called_once()
        
        # Check rate limiter recorded the success
        stats = limiter.get_stats("gpt-4o-mini")
        assert stats["gpt-4o-mini"].successful_requests == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting_with_failures(self):
        """Test rate limiting handles API failures correctly."""
        from agents import set_global_rate_limiter, _ainvoke_with_retry
        
        # Set up rate limiter
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        set_global_rate_limiter(limiter)
        
        # Mock chain that fails with rate limit error
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(side_effect=Exception("Rate limit exceeded (429)"))
        
        mock_llm = MagicMock()
        mock_llm.model = "gpt-4o"
        
        # Should retry and eventually fail
        with pytest.raises(Exception):
            await _ainvoke_with_retry(
                mock_chain, 
                {"prompt_text": "Test prompt"}, 
                attempts=2,
                llm=mock_llm
            )
        
        # Should have recorded failures
        stats = limiter.get_stats("gpt-4o")
        assert stats["gpt-4o"].total_requests > 0
    
    def test_rate_limiting_without_global_limiter(self):
        """Test graceful fallback when no rate limiter is set."""
        from agents import set_global_rate_limiter, get_global_rate_limiter
        
        # Clear global rate limiter
        set_global_rate_limiter(None)
        assert get_global_rate_limiter() is None
        
        # Should not crash - will use original retry logic


class TestRateLimitConfiguration:
    """Test rate limit configuration and validation."""
    
    def test_rate_limit_config_creation(self):
        """Test creating rate limit configuration objects."""
        config = RateLimitConfig(
            requests_per_minute=100,
            tokens_per_minute=10000,
            max_concurrent=5,
            burst_allowance=1.2,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=30.0
        )
        
        assert config.requests_per_minute == 100
        assert config.tokens_per_minute == 10000
        assert config.max_concurrent == 5
        assert config.burst_allowance == 1.2
        assert config.circuit_breaker_threshold == 3
        assert config.circuit_breaker_timeout == 30.0
    
    def test_default_rate_limits(self):
        """Test that default rate limits are reasonable."""
        limiter = APIRateLimiter(mode=RateLimitMode.CONSERVATIVE)
        
        # Should have defaults for common models
        conservative_limits = limiter.DEFAULT_LIMITS[RateLimitMode.CONSERVATIVE]
        assert "gpt-4o" in conservative_limits
        assert "gpt-4o-mini" in conservative_limits
        assert "default" in conservative_limits
        
        # Conservative should be more restrictive than aggressive
        aggressive_limits = limiter.DEFAULT_LIMITS[RateLimitMode.AGGRESSIVE]
        assert conservative_limits["gpt-4o"].requests_per_minute <= aggressive_limits["gpt-4o"].requests_per_minute
        assert conservative_limits["gpt-4o"].max_concurrent <= aggressive_limits["gpt-4o"].max_concurrent
    
    def test_rate_limit_modes(self):
        """Test different rate limiting modes have appropriate settings."""
        modes = [RateLimitMode.CONSERVATIVE, RateLimitMode.BALANCED, RateLimitMode.AGGRESSIVE]
        
        for mode in modes:
            limiter = APIRateLimiter(mode=mode)
            config = limiter._get_model_config("gpt-4o")
            
            assert config.requests_per_minute > 0
            assert config.tokens_per_minute > 0
            assert config.max_concurrent > 0
            assert config.burst_allowance >= 1.0
    
    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """Test that rate limits are actually enforced."""
        # Create limiter with very restrictive limits for testing
        custom_config = {
            "test-model": RateLimitConfig(
                requests_per_minute=1,  # Very low for testing
                tokens_per_minute=100,
                max_concurrent=1
            )
        }
        
        limiter = APIRateLimiter(mode=RateLimitMode.CUSTOM, custom_config=custom_config)
        
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        
        # First request should succeed
        can_proceed, wait_time = await limiter.acquire(mock_llm, estimated_tokens=50)
        assert can_proceed == True
        
        # Second request should be rate limited
        can_proceed, wait_time = await limiter.acquire(mock_llm, estimated_tokens=50)
        assert can_proceed == False
        assert wait_time is not None
        assert wait_time > 0