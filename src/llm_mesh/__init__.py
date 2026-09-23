"""Async LLM clients and shared request/response types."""

from llm_mesh.protocol import (
    AsyncClosable,
    BatchLLMClient,
    EventStreamGenerator,
    LLMClient,
    StreamGenerator,
    StructuredGenerator,
    TextGenerator,
)
from llm_mesh.types import (
    LLMAuthError,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
)

__all__ = [
    "AsyncClosable",
    "BaseLLMClient",
    "BatchLLMClient",
    "Capability",
    "EventStreamGenerator",
    "LLMAuthError",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMValidationError",
    "StreamGenerator",
    "StructuredGenerator",
    "TextGenerator",
]

from llm_mesh.base import BaseLLMClient, Capability

from .openai import OpenAIClient
from .anthropic import AnthropicClient, AnthropicError
from .gemini import GeminiClient, GeminiError
from .gigachat import GigaChatAsyncClient
from .types import LLMUsage, LLMStreamChunk

from importlib.metadata import version

__version__ = version("llm-mesh")
__all__ += [
    "AnthropicClient",
    "AnthropicError",
    "GeminiClient",
    "GeminiError",
    "OpenAIClient",
    "GigaChatAsyncClient",
    "LLMUsage",
    "LLMStreamChunk",
]

from .openai.discovery import list_models
from llm_mesh.probe import (
    ConnectionCheck,
    ProbeKind,
    check_client,
    check_route,
    check_routes,
)

__all__ += [
    "ConnectionCheck",
    "ProbeKind",
    "check_client",
    "check_route",
    "check_routes",
    "list_models",
]
