"""Model discovery for OpenAI-compatible endpoints."""

import httpx

from llm_mesh.types import LLMAuthError, LLMError, LLMTimeoutError, LLMValidationError


def list_models(
    base_url: str, api_key: str, *, timeout_s: float = 15.0,
    verify: bool = True,
) -> list[str]:
    """Return model IDs in the endpoint's order using its synchronous /models API.

    Accept either a base URL or a full chat/completions URL. Discovery is separate
    from client construction so callers may choose a model before starting an
    asynchronous session.
    """
    url = base_url.rstrip("/").removesuffix("/chat/completions") + '/models'
    try:
        with httpx.Client(timeout=timeout_s, verify=verify) as client:
            response = client.get(url, headers={"Authorization": f"Bearer {api_key}"})
            if response.status_code in (401, 403):
                raise LLMAuthError(f"Model discovery HTTP {response.status_code}")
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise LLMTimeoutError("Model discovery timed out") from exc
    except httpx.HTTPError as exc:
        raise LLMError("Model discovery request failed") from exc
    try:
        data = response.json()
        entries = data.get("data") or data.get("models") or []
        if not isinstance(entries, list):
            raise ValueError("model list is not an array")
        ids = [entry.get("id") or entry.get("name") for entry in entries]
        if any(not isinstance(model, str) or not model for model in ids):
            raise ValueError("model ID is missing")
        return ids
    except (AttributeError, TypeError, ValueError) as exc:
        raise LLMValidationError("Invalid model discovery response") from exc
