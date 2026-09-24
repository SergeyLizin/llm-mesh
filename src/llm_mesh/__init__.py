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
    Budget,
    BudgetState,
    CallRecord,
    ImageAttachment,
    LLMAuthError,
    LLMBudgetExceeded,
    LLMError,
    LLMRequest,
    LLMRequestBlocked,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
)

__all__ = [
    "AsyncClosable",
    "BaseLLMClient",
    "BatchLLMClient",
    "Budget",
    "BudgetState",
    "CallRecord",
    "Capability",
    "Embedding",
    "EventStreamGenerator",
    "ImageAttachment",
    "LLMAuthError",
    "LLMBudgetExceeded",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMRequestBlocked",
    "LLMResponse",
    "LLMTimeoutError",
    "LLMValidationError",
    "LocalCrossEncoder",
    "QueryInstruction",
    "RerankHit",
    "StreamGenerator",
    "StructuredGenerator",
    "TextGenerator",
]

from llm_mesh.base import BaseLLMClient, Capability
from llm_mesh.embeddings import Embedding, QueryInstruction
from llm_mesh.rerank import LocalCrossEncoder, RerankHit

from .openai import OpenAIClient
from .anthropic import AnthropicClient, AnthropicError
from .gemini import GeminiClient, GeminiError
from .gigachat import GigaChatAsyncClient, GigaChatClient
from .types import LLMUsage, LLMStreamChunk

from importlib.metadata import version

__version__ = version("llm-mesh")
__all__ += [
    "AnthropicClient",
    "AnthropicError",
    "GeminiClient",
    "GeminiError",
    "OpenAIClient",
    "GigaChatClient",
    "GigaChatAsyncClient",  # Deprecated alias; removed in 3.0.0.
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
