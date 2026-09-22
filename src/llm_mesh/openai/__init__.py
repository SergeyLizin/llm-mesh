"""Clients and model discovery for OpenAI-compatible endpoints."""

from .client import OpenAIClient, OpenAIError, ToolArgsCorruptedError
from .discovery import list_models

__all__ = ["OpenAIClient", "OpenAIError", "ToolArgsCorruptedError", "list_models"]
