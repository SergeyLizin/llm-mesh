from contextvars import ContextVar

from llm_mesh import LLMRequest, _common, hooks


def test_application_hooks_do_not_require_application_imports(monkeypatch):
    token = ContextVar("test_canary", default=None)
    seen = []
    for name in ("_get_token", "_build_prompt", "_check"):
        monkeypatch.setattr(hooks, name, getattr(hooks, name))
    hooks.configure_canary_hooks(
        get_token=token.get,
        build_prompt=lambda value: f" marker={value}",
        check=lambda text, value, **kw: seen.append((text, value, kw["context"])),
    )
    request = LLMRequest(system="system", user="user")
    assert _common.build_text_messages(request)[0]["content"] == "system"
    handle = token.set("test-marker")
    try:
        assert _common.build_text_messages(request)[0]["content"] == "system marker=test-marker"
        _common.check_response_canary("reply", context="text")
        assert seen == [("reply", "test-marker", "text")]
    finally:
        token.reset(handle)
    assert _common.build_text_messages(request)[0]["content"] == "system"
