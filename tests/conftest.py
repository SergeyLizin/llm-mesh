import os
import pytest

from llm_mesh.canary import (
    build_canary_prompt,
    check_and_warn,
    get_canary_context_token,
)
from llm_mesh.hooks import configure_canary_hooks

# Restore the canary implementations, not the hooks dispatchers: wiring
# hooks.get_canary_context_token back into itself would recurse.


@pytest.fixture(autouse=True)
def isolated_llm_options(monkeypatch):
    """Factory calls assign LLM_OPTIONS directly; prevent cross-test route leakage."""
    monkeypatch.delenv("LLM_OPTIONS", raising=False)
    yield
    os.environ.pop("LLM_OPTIONS", None)


@pytest.fixture(autouse=True)
def restore_canary_hooks():
    """Keep the built-in canary as the process default after tests that replace hooks."""
    configure_canary_hooks(
        get_token=get_canary_context_token,
        build_prompt=build_canary_prompt,
        check=check_and_warn,
    )
    yield
    configure_canary_hooks(
        get_token=get_canary_context_token,
        build_prompt=build_canary_prompt,
        check=check_and_warn,
    )
