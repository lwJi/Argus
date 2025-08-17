# rate_limiting.py
import asyncio
import time
import logging
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class RateLimitMode(Enum):
    """Rate limiting operational modes."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class RateLimitConfig:
    """Configuration for API rate limiting."""
    requests_per_minute: int
    tokens_per_minute: int
    max_concurrent: int
    burst_allowance: float = 1.2  # Allow 20% burst above sustained rate
    circuit_breaker_threshold: int = 5  # Failures before circuit breaker trips
    circuit_breaker_timeout: float = 60.0  # Seconds before trying again


@dataclass
class RateLimitStats:
    """Statistics for rate limiting monitoring."""
    total_requests: int = 0
    successful_requests: int = 0
    rate_limited_requests: int = 0
    circuit_breaker_trips: int = 0
    average_response_time: float = 0.0
    current_queue_depth: int = 0
    tokens_used_this_minute: int = 0
    requests_used_this_minute: int = 0


class TokenBucket:
    """Token bucket algorithm implementation for rate limiting."""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Attempt to consume tokens from bucket. Returns True if successful."""
        with self._lock:
            now = time.time()
            # Refill tokens based on time elapsed
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def available_tokens(self) -> int:
        """Get current number of available tokens."""
        with self._lock:
            now = time.time()
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            return min(self.capacity, self.tokens + tokens_to_add)
    
    def time_until_tokens(self, needed_tokens: int) -> float:
        """Calculate seconds until enough tokens are available."""
        available = self.available_tokens()
        if available >= needed_tokens:
            return 0.0
        
        shortage = needed_tokens - available
        return shortage / self.refill_rate


class CircuitBreaker:
    """Circuit breaker pattern for API health monitoring."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call_succeeded(self):
        """Record successful call."""
        with self._lock:
            self.failure_count = 0
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
    
    def call_failed(self):
        """Record failed call."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
    
    def can_execute(self) -> bool:
        """Check if calls are allowed through the circuit breaker."""
        with self._lock:
            if self.state == "CLOSED":
                return True
            elif self.state == "OPEN":
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            else:  # HALF_OPEN
                return True


class APIRateLimiter:
    """Comprehensive rate limiter for OpenAI API calls."""
    
    # Default rate limits based on common OpenAI tiers
    DEFAULT_LIMITS = {
        RateLimitMode.CONSERVATIVE: {
            "gpt-4o": RateLimitConfig(requests_per_minute=100, tokens_per_minute=10000, max_concurrent=3),
            "gpt-4o-mini": RateLimitConfig(requests_per_minute=500, tokens_per_minute=50000, max_concurrent=5),
            "default": RateLimitConfig(requests_per_minute=60, tokens_per_minute=5000, max_concurrent=2)
        },
        RateLimitMode.BALANCED: {
            "gpt-4o": RateLimitConfig(requests_per_minute=500, tokens_per_minute=30000, max_concurrent=10),
            "gpt-4o-mini": RateLimitConfig(requests_per_minute=2000, tokens_per_minute=150000, max_concurrent=15),
            "default": RateLimitConfig(requests_per_minute=300, tokens_per_minute=20000, max_concurrent=5)
        },
        RateLimitMode.AGGRESSIVE: {
            "gpt-4o": RateLimitConfig(requests_per_minute=1000, tokens_per_minute=60000, max_concurrent=20),
            "gpt-4o-mini": RateLimitConfig(requests_per_minute=5000, tokens_per_minute=200000, max_concurrent=30),
            "default": RateLimitConfig(requests_per_minute=600, tokens_per_minute=40000, max_concurrent=10)
        }
    }
    
    def __init__(self, mode: RateLimitMode = RateLimitMode.CONSERVATIVE, custom_config: Optional[Dict[str, RateLimitConfig]] = None):
        self.mode = mode
        self.custom_config = custom_config or {}
        
        # Initialize rate limiting components
        self._request_buckets: Dict[str, TokenBucket] = {}
        self._token_buckets: Dict[str, TokenBucket] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Statistics and monitoring
        self._stats: Dict[str, RateLimitStats] = defaultdict(RateLimitStats)
        self._request_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Request queue for priority handling
        self._request_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._queue_processor_task: Optional[asyncio.Task] = None
        
        # Initialize components for known models
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize rate limiting components for known models."""
        config_dict = self.custom_config if self.mode == RateLimitMode.CUSTOM else self.DEFAULT_LIMITS[self.mode]
        
        for model_name, config in config_dict.items():
            self._setup_model_limits(model_name, config)
    
    def _setup_model_limits(self, model_name: str, config: RateLimitConfig):
        """Setup rate limiting components for a specific model."""
        # Token buckets for requests and tokens per minute
        self._request_buckets[model_name] = TokenBucket(
            capacity=int(config.requests_per_minute * config.burst_allowance),
            refill_rate=config.requests_per_minute / 60.0  # per second
        )
        
        self._token_buckets[model_name] = TokenBucket(
            capacity=int(config.tokens_per_minute * config.burst_allowance),
            refill_rate=config.tokens_per_minute / 60.0  # per second
        )
        
        # Semaphore for concurrent request limiting
        self._semaphores[model_name] = asyncio.Semaphore(config.max_concurrent)
        
        # Circuit breaker for API health monitoring
        self._circuit_breakers[model_name] = CircuitBreaker(
            failure_threshold=config.circuit_breaker_threshold,
            timeout=config.circuit_breaker_timeout
        )
    
    def _get_model_name(self, llm: Any) -> str:
        """Extract model name from LLM instance."""
        for attr in ("model", "model_name", "model_id"):
            val = getattr(llm, attr, None)
            if isinstance(val, str) and val:
                return val
        return "default"
    
    def _get_model_config(self, model_name: str) -> RateLimitConfig:
        """Get rate limit configuration for a model."""
        if self.mode == RateLimitMode.CUSTOM:
            return self.custom_config.get(model_name, self.custom_config.get("default", 
                   self.DEFAULT_LIMITS[RateLimitMode.CONSERVATIVE]["default"]))
        else:
            return self.DEFAULT_LIMITS[self.mode].get(model_name, 
                   self.DEFAULT_LIMITS[self.mode]["default"])
    
    async def acquire(self, llm: Any, estimated_tokens: int = 100) -> Tuple[bool, Optional[float]]:
        """
        Acquire permission to make an API call.
        Returns (can_proceed, wait_time_seconds).
        """
        model_name = self._get_model_name(llm)
        
        # Ensure model is set up
        if model_name not in self._request_buckets:
            config = self._get_model_config(model_name)
            self._setup_model_limits(model_name, config)
        
        # Check circuit breaker
        circuit_breaker = self._circuit_breakers[model_name]
        if not circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for model {model_name}")
            return False, circuit_breaker.timeout
        
        # Check token bucket for requests
        request_bucket = self._request_buckets[model_name]
        if not request_bucket.consume(1):
            wait_time = request_bucket.time_until_tokens(1)
            logger.debug(f"Request rate limit hit for {model_name}, wait: {wait_time:.2f}s")
            return False, wait_time
        
        # Check token bucket for estimated tokens
        token_bucket = self._token_buckets[model_name]
        if not token_bucket.consume(estimated_tokens):
            wait_time = token_bucket.time_until_tokens(estimated_tokens)
            logger.debug(f"Token rate limit hit for {model_name}, wait: {wait_time:.2f}s")
            # Return tokens to request bucket since we're not proceeding
            request_bucket.consume(-1)
            return False, wait_time
        
        # Update statistics
        stats = self._stats[model_name]
        stats.total_requests += 1
        
        return True, None
    
    async def acquire_with_semaphore(self, llm: Any, estimated_tokens: int = 100):
        """Acquire both rate limit permission and concurrency semaphore."""
        model_name = self._get_model_name(llm)
        
        # Ensure model is set up
        if model_name not in self._semaphores:
            config = self._get_model_config(model_name)
            self._setup_model_limits(model_name, config)
        
        # Wait for rate limit clearance
        while True:
            can_proceed, wait_time = await self.acquire(llm, estimated_tokens)
            if can_proceed:
                break
            if wait_time:
                await asyncio.sleep(min(wait_time, 1.0))  # Cap wait time at 1 second
        
        # Acquire concurrency semaphore
        semaphore = self._semaphores[model_name]
        await semaphore.acquire()
        
        return semaphore
    
    def record_success(self, llm: Any, response_time: float, actual_tokens: int = 0):
        """Record a successful API call."""
        model_name = self._get_model_name(llm)
        
        # Update circuit breaker
        if model_name in self._circuit_breakers:
            self._circuit_breakers[model_name].call_succeeded()
        
        # Update statistics
        stats = self._stats[model_name]
        stats.successful_requests += 1
        
        # Update response time tracking
        request_times = self._request_times[model_name]
        request_times.append(response_time)
        if request_times:
            stats.average_response_time = sum(request_times) / len(request_times)
    
    def record_failure(self, llm: Any, error: Exception):
        """Record a failed API call."""
        model_name = self._get_model_name(llm)
        
        # Update circuit breaker
        if model_name in self._circuit_breakers:
            self._circuit_breakers[model_name].call_failed()
        
        # Update statistics based on error type
        stats = self._stats[model_name]
        if "rate limit" in str(error).lower() or "429" in str(error):
            stats.rate_limited_requests += 1
        
        logger.warning(f"API call failed for {model_name}: {error}")
    
    def get_stats(self, model_name: Optional[str] = None) -> Dict[str, RateLimitStats]:
        """Get rate limiting statistics."""
        if model_name:
            return {model_name: self._stats[model_name]}
        return dict(self._stats)
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get detailed status of all rate limiters."""
        status = {}
        
        for model_name in self._request_buckets.keys():
            request_bucket = self._request_buckets[model_name]
            token_bucket = self._token_buckets[model_name]
            circuit_breaker = self._circuit_breakers[model_name]
            stats = self._stats[model_name]
            
            status[model_name] = {
                "requests_available": request_bucket.available_tokens(),
                "tokens_available": token_bucket.available_tokens(),
                "circuit_breaker_state": circuit_breaker.state,
                "concurrent_requests": self._get_model_config(model_name).max_concurrent - self._semaphores[model_name]._value,
                "stats": stats
            }
        
        return status
    
    async def wait_for_capacity(self, llm: Any, estimated_tokens: int = 100, max_wait: float = 300.0) -> bool:
        """
        Wait for sufficient capacity to make a request.
        Returns True if capacity is available, False if max_wait exceeded.
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            can_proceed, wait_time = await self.acquire(llm, estimated_tokens)
            if can_proceed:
                return True
            
            if wait_time:
                sleep_time = min(wait_time, max_wait - (time.time() - start_time))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
        
        logger.warning(f"Max wait time {max_wait}s exceeded waiting for API capacity")
        return False