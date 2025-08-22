"""
FlashMM Azure OpenAI Client

High-performance async client for Azure OpenAI o4-mini integration
with rate limiting, error handling, and performance optimization.
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import aiohttp
from azure.identity import DefaultAzureCredential
from openai import AsyncAzureOpenAI

from flashmm.config.settings import get_config
from flashmm.data.storage.data_models import MarketStats, OrderBookSnapshot, Trade
from flashmm.ml.prompts.market_prompts import MarketPredictionPrompt, PredictionResponseParser
from flashmm.utils.exceptions import InferenceError
from flashmm.utils.logging import get_logger

logger = get_logger(__name__)


class AzureOpenAIConfig:
    """Configuration for Azure OpenAI integration."""

    def __init__(self, config_manager=None):
        """Initialize Azure OpenAI configuration."""
        self.config = config_manager or get_config()

        # Azure OpenAI settings
        self.endpoint = self.config.get("azure_openai.endpoint", "https://your-resource.openai.azure.com/")
        self.api_key = self.config.get("azure_openai.api_key")
        self.api_version = self.config.get("azure_openai.api_version", "2024-02-15-preview")
        self.model_deployment = self.config.get("azure_openai.model_deployment", "o4-mini-deployment")

        # Performance settings
        self.max_tokens = self.config.get("azure_openai.max_tokens", 150)
        self.temperature = self.config.get("azure_openai.temperature", 0.1)
        self.timeout_seconds = self.config.get("azure_openai.timeout_seconds", 10)

        # Rate limiting
        self.requests_per_minute = self.config.get("azure_openai.requests_per_minute", 1000)
        self.concurrent_requests = self.config.get("azure_openai.concurrent_requests", 10)

        # Retry settings
        self.max_retries = self.config.get("azure_openai.max_retries", 3)
        self.retry_delay = self.config.get("azure_openai.retry_delay", 0.1)
        self.backoff_multiplier = self.config.get("azure_openai.backoff_multiplier", 2.0)


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(self, requests_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute
        """
        self.max_tokens = requests_per_minute
        self.tokens = requests_per_minute
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Acquire a token for rate limiting.

        Returns:
            True if token acquired, False if rate limited
        """
        async with self.lock:
            now = time.time()
            # Add tokens based on time elapsed
            time_passed = now - self.last_update
            tokens_to_add = time_passed * (self.max_tokens / 60.0)  # tokens per second
            self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def wait_for_token(self) -> None:
        """Wait until a token becomes available."""
        while not await self.acquire():
            await asyncio.sleep(0.1)


class AzureOpenAIClient:
    """High-performance Azure OpenAI client for trading predictions."""

    def __init__(self, config: AzureOpenAIConfig | None = None):
        """Initialize Azure OpenAI client.

        Args:
            config: Azure OpenAI configuration
        """
        self.config = config or AzureOpenAIConfig()
        self.client: AsyncAzureOpenAI | None = None
        self.session: aiohttp.ClientSession | None = None

        # Rate limiting and concurrency
        self.rate_limiter = RateLimiter(self.config.requests_per_minute)
        self.semaphore = asyncio.Semaphore(self.config.concurrent_requests)

        # Prompt components
        self.prompt_builder = MarketPredictionPrompt()
        self.response_parser = PredictionResponseParser()

        # Performance tracking
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.success_count = 0
        self.last_request_time = None

        # Cost tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize Azure OpenAI client and HTTP session."""
        if self._initialized:
            return

        try:
            # Setup HTTP session with optimized connection pooling
            connector = aiohttp.TCPConnector(
                limit=50,  # Total connection pool size
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )

            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout_seconds,
                connect=5,
                sock_read=self.config.timeout_seconds
            )

            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={
                    "User-Agent": "FlashMM/1.0",
                    "Accept": "application/json",
                    "Content-Type": "application/json"
                }
            )

            # Initialize Azure OpenAI client
            if self.config.api_key:
                # Use API key authentication
                self.client = AsyncAzureOpenAI(
                    azure_endpoint=self.config.endpoint,
                    api_key=self.config.api_key,
                    api_version=self.config.api_version
                )
            else:
                # Use Azure Identity (managed identity or default credential)
                credential = DefaultAzureCredential()
                # Create a token provider function for Azure AD authentication
                async def get_azure_ad_token():
                    """Get Azure AD token for authentication."""
                    token = credential.get_token("https://cognitiveservices.azure.com/.default")
                    return token.token

                self.client = AsyncAzureOpenAI(
                    azure_endpoint=self.config.endpoint,
                    azure_ad_token_provider=get_azure_ad_token,
                    api_version=self.config.api_version
                )

            # Test connection
            await self._test_connection()

            self._initialized = True
            logger.info("Azure OpenAI client initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            await self.cleanup()
            raise InferenceError(f"Azure OpenAI client initialization failed: {e}") from e

    async def _test_connection(self) -> None:
        """Test Azure OpenAI connection with a simple request."""
        if not self.client:
            raise Exception("Azure OpenAI client not initialized")

        try:
            # Simple test request
            response = await self.client.chat.completions.create(
                model=self.config.model_deployment,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello"}
                ],
                max_tokens=10,
                temperature=0.1
            )

            if not response.choices:
                raise Exception("No response from Azure OpenAI")

            logger.info("Azure OpenAI connection test successful")

        except Exception as e:
            logger.error(f"Azure OpenAI connection test failed: {e}")
            raise

    async def predict_market_direction(
        self,
        order_book: OrderBookSnapshot,
        recent_trades: list[Trade],
        market_stats: MarketStats | None = None,
        prediction_horizon_ms: int = 200
    ) -> dict[str, Any]:
        """Get market prediction from Azure OpenAI.

        Args:
            order_book: Current order book snapshot
            recent_trades: Recent trades list
            market_stats: Market statistics (optional)
            prediction_horizon_ms: Prediction horizon in milliseconds

        Returns:
            Structured prediction dictionary
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Rate limiting
            await self.rate_limiter.wait_for_token()

            async with self.semaphore:
                # Build prompt
                user_prompt = self.prompt_builder.build_market_prompt(
                    order_book=order_book,
                    recent_trades=recent_trades,
                    market_stats=market_stats,
                    prediction_horizon_ms=prediction_horizon_ms
                )

                # Make API call with retry logic
                response_content = await self._call_with_retry(
                    system_prompt=self.prompt_builder.SYSTEM_PROMPT,
                    user_prompt=user_prompt
                )

                # Parse response
                prediction = await self.response_parser.parse_prediction(response_content)

                # Add performance metadata
                response_time = (time.time() - start_time) * 1000
                prediction['response_time_ms'] = response_time
                prediction['api_success'] = True
                prediction['symbol'] = order_book.symbol

                # Update metrics
                self._update_metrics(response_time, success=True)

                return prediction

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self._update_metrics(response_time, success=False)

            logger.error(f"Azure OpenAI prediction failed: {e}")

            # Return fallback prediction
            return {
                'direction': 'neutral',
                'confidence': 0.5,
                'price_change_bps': 0.0,
                'magnitude': 'low',
                'reasoning': f'API error: {str(e)}',
                'key_factors': ['api_error'],
                'timestamp': datetime.utcnow().isoformat(),
                'model_version': 'azure-openai-fallback',
                'response_time_ms': response_time,
                'api_success': False,
                'symbol': order_book.symbol
            }

    async def _call_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Make API call with exponential backoff retry.

        Args:
            system_prompt: System prompt for the model
            user_prompt: User prompt with market data

        Returns:
            Response content from Azure OpenAI
        """
        last_exception = None

        if not self.client:
            raise Exception("Azure OpenAI client not initialized")

        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.model_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    timeout=self.config.timeout_seconds
                )

                if not response.choices:
                    raise Exception("Empty response from Azure OpenAI")

                content = response.choices[0].message.content
                if not content:
                    raise Exception("No content in Azure OpenAI response")

                # Update token usage
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += response.usage.prompt_tokens or 0
                    self.total_output_tokens += response.usage.completion_tokens or 0
                    self._update_cost_estimate()

                return content

            except Exception as e:
                last_exception = e

                if attempt == self.config.max_retries - 1:
                    break  # Last attempt failed

                # Exponential backoff with jitter
                delay = self.config.retry_delay * (self.config.backoff_multiplier ** attempt)
                jitter = delay * 0.1 * (0.5 - asyncio.get_event_loop().time() % 1)
                await asyncio.sleep(delay + jitter)

                logger.warning(f"Azure OpenAI retry {attempt + 1}/{self.config.max_retries}: {e}")

        # All retries failed
        raise last_exception or Exception("Azure OpenAI call failed after retries")

    def _update_metrics(self, response_time: float, success: bool) -> None:
        """Update performance metrics.

        Args:
            response_time: Response time in milliseconds
            success: Whether the request was successful
        """
        self.request_count += 1
        self.total_latency += response_time
        self.last_request_time = datetime.utcnow()

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def _update_cost_estimate(self) -> None:
        """Update estimated API cost based on token usage."""
        # Azure OpenAI o4-mini pricing (approximate)
        input_cost_per_1k = 0.000150  # USD per 1K input tokens
        output_cost_per_1k = 0.000600  # USD per 1K output tokens

        input_cost = (self.total_input_tokens / 1000) * input_cost_per_1k
        output_cost = (self.total_output_tokens / 1000) * output_cost_per_1k

        self.estimated_cost = input_cost + output_cost

    @property
    def average_latency_ms(self) -> float:
        """Get average response latency in milliseconds."""
        if self.request_count == 0:
            return 0.0
        return self.total_latency / self.request_count

    @property
    def error_rate(self) -> float:
        """Get error rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100

    @property
    def success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.success_count / self.request_count) * 100

    @property
    def requests_per_second(self) -> float:
        """Get current requests per second rate."""
        if not self.last_request_time or self.request_count < 2:
            return 0.0

        # Use last minute of requests for rate calculation
        time_window = 60  # seconds
        now = datetime.utcnow()
        if (now - self.last_request_time).total_seconds() < time_window:
            return self.request_count / ((now - self.last_request_time).total_seconds())

        return 0.0

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'request_count': self.request_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'average_latency_ms': self.average_latency_ms,
            'requests_per_second': self.requests_per_second,
            'total_input_tokens': self.total_input_tokens,
            'total_output_tokens': self.total_output_tokens,
            'estimated_cost_usd': self.estimated_cost,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'initialized': self._initialized
        }

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on Azure OpenAI service."""
        if not self._initialized:
            return {
                'status': 'unhealthy',
                'reason': 'client_not_initialized',
                'timestamp': datetime.utcnow().isoformat()
            }

        if not self.client:
            return {
                'status': 'unhealthy',
                'reason': 'client_not_initialized',
                'timestamp': datetime.utcnow().isoformat()
            }

        try:
            start_time = time.time()

            # Simple health check request
            response = await self.client.chat.completions.create(
                model=self.config.model_deployment,
                messages=[
                    {"role": "system", "content": "Health check"},
                    {"role": "user", "content": "ping"}
                ],
                max_tokens=5,
                temperature=0.0
            )

            latency = (time.time() - start_time) * 1000

            if response.choices and response.choices[0].message.content:
                return {
                    'status': 'healthy',
                    'latency_ms': latency,
                    'timestamp': datetime.utcnow().isoformat()
                }
            else:
                return {
                    'status': 'unhealthy',
                    'reason': 'empty_response',
                    'latency_ms': latency,
                    'timestamp': datetime.utcnow().isoformat()
                }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'reason': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }

    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()

            self.client = None
            self.session = None
            self._initialized = False

            logger.info("Azure OpenAI client cleanup completed")

        except Exception as e:
            logger.error(f"Error during Azure OpenAI client cleanup: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
