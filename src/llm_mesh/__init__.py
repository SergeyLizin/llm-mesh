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
    "BatchLLMClient",
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

from .openai import OpenAIClient
from .gigachat import GigaChatAsyncClient
from .types import LLMUsage, LLMStreamChunk

from importlib.metadata import version

__version__ = version("llm-mesh")
__all__ += ["OpenAIClient", "GigaChatAsyncClient", "LLMUsage", "LLMStreamChunk"]

from .openai.discovery import list_models

__all__ += ["list_models"]
