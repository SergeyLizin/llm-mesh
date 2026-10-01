"""Load provider routes, resolve credentials and construct LLM clients.

Catalog paths are explicit, environment-selected, or bundled with the package.
Application-specific catalogs never depend on the current working directory.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shlex
import sys
from pathlib import Path
from typing import Any, Callable

from llm_mesh.config import CONNECTION_ENV, route_options

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent

# Fields that belong to individual provider routes, not model-level defaults.
ROUTE_ONLY_FIELDS = frozenset({
    "base_url", "base_url_env", "api_key", "api_key_env",
    "scope", "scope_env", "auth_url", "auth_url_env",
    "model", "model_env", "extra_body", "extra_headers",
    "reasoning_off", "reasoning_on", "reasoning_field",
    # Concurrency limits describe an endpoint, not a model. Hoisting one to model scope can make
    # another provider inherit an unsupported concurrency level and trigger rate limits.
    "eval_concurrency",
})

# Always assign every managed environment variable, using an empty string for unset values, so
# configuration from the previous route cannot leak into the next.
_MANAGED_ENV = (*CONNECTION_ENV, "LLM_OPTIONS")


class ModelCatalogError(RuntimeError):
    """The catalog cannot resolve the requested route or its credentials."""


# Catalog loading.


def catalog_path(path: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the catalog path: explicit argument, LLM_MODELS_CONFIG, then bundled models.json."""
    if path:
        return Path(path)
    env_path = os.environ.get("LLM_MODELS_CONFIG", "").strip()
    if env_path:
        return Path(env_path)
    return _REPO_ROOT / "models.json"


def route_label(model_id: str, provider: str) -> str:
    """Derive the route label from its model id and provider."""
    return f"{model_id} ({provider})"


def expand_catalog(data: list[dict]) -> list[dict]:
    """Expand model entries into provider routes. Provider fields override model defaults, provider
    name becomes provider, and a default label is derived unless explicitly supplied.
    """
    out: list[dict] = []
    for model in data:
        provs = model.get("providers")
        if not provs:
            out.append(model)
            continue
        base = {k: v for k, v in model.items() if k not in ("providers", "cluster")}
        names: list[str] = []
        seen_names: set[str] = set()
        drafted: list[tuple[str, dict]] = []
        for prov in provs:
            name = prov.get("name") if isinstance(prov, dict) else None
            if not isinstance(name, str) or not name:
                raise ModelCatalogError(
                    f"catalog model {base.get('id')!r}: a provider entry has no name"
                )
            route = dict(base)
            route.update({k: v for k, v in prov.items() if k != "name"})
            if not route.get("id"):
                raise ModelCatalogError(
                    f"catalog provider {name!r}: model entry has no id"
                )
            route["provider"] = name
            route.setdefault("label", route_label(route["id"], name))
            if name in seen_names:
                raise ModelCatalogError(
                    f"catalog model {base.get('id')!r}: duplicate provider {name!r}"
                )
            seen_names.add(name)
            names.append(name)
            drafted.append((name, route))
        cluster = model.get("cluster")
        if cluster is not None:
            _validate_provider_cluster(base.get("id"), cluster, names)
            if "cluster" in names:
                raise ModelCatalogError(
                    f"catalog model {base.get('id')!r}: a provider cannot be named "
                    "'cluster'; that name selects the pool"
                )
        for _name, route in drafted:
            if cluster is not None:
                route["cluster"] = list(cluster)
            out.append(route)
    return out


def _patch_fields(original: dict, patch: dict, *, skip: set[str]) -> dict:
    """Apply shallow field changes; null removes a field instead of overriding it."""
    result = dict(original)
    for key, value in patch.items():
        if key in skip:
            continue
        if value is None:
            result.pop(key, None)
        else:
            result[key] = value
    return result


def _patch_model(original: dict, patch: dict) -> dict:
    result = _patch_fields(original, patch, skip={"id", "providers", "exclude_providers"})
    if "providers" not in patch and "exclude_providers" not in patch:
        return result
    providers = {p["name"]: dict(p) for p in original.get("providers", [])}
    changes = patch.get("providers", [])
    if not isinstance(changes, list):
        raise ModelCatalogError("patch providers must be an array")
    seen = set()
    for change in changes:
        if not isinstance(change, dict) or not isinstance(change.get("name"), str):
            raise ModelCatalogError("provider patches require a string name")
        name = change["name"]
        if name in seen:
            raise ModelCatalogError(f"duplicate provider patch: {name}")
        seen.add(name)
        providers[name] = _patch_fields(providers.get(name, {"name": name}), change, skip={"name"})
    excluded = patch.get("exclude_providers", [])
    if not isinstance(excluded, list) or any(not isinstance(name, str) for name in excluded):
        raise ModelCatalogError("exclude_providers must be an array of names")
    for name in excluded:
        providers.pop(name, None)
    result["providers"] = list(providers.values())
    if not result["providers"]:
        raise ModelCatalogError("a provider patch must leave at least one provider")
    _reconcile_patched_cluster(result, cluster_in_patch="cluster" in patch)
    return result


def _reconcile_patched_cluster(result: dict, *, cluster_in_patch: bool) -> None:
    """Drop providers this patch removed from an inherited ``cluster``.

    ``exclude_providers`` used to leave the name list untouched, and
    ``expand_catalog`` then refused to load the whole catalog. A name the
    patch itself still lists is an error. An inherited list shrinks to the
    providers that remain. Fewer than two is not a pool: the patch has to
    name the survivors or remove ``cluster``.
    """
    cluster = result.get("cluster")
    if not isinstance(cluster, list):
        return
    names = [provider.get("name") for provider in result.get("providers", [])]
    missing = [name for name in cluster if name not in names]
    if not missing:
        return
    if cluster_in_patch:
        raise ModelCatalogError(
            f"{result.get('id')!r}: cluster names provider {missing[0]!r}, "
            "which this patch removes"
        )
    kept = [name for name in cluster if name in names]
    if len(kept) < 2:
        raise ModelCatalogError(
            f"{result.get('id')!r}: exclude_providers leaves cluster with "
            f"{len(kept)} provider(s); set cluster to the providers that remain "
            "or remove it"
        )
    result["cluster"] = kept


def load_catalog_data(path: str | os.PathLike[str] | None = None) -> list[dict]:
    """Read a nested catalog, or project overrides of the bundled catalog.

    An override document has ``extends: llm-mesh``, optional ``models``
    (complete replacements/additions keyed by id), field-level ``patches``,
    ``exclude`` ids and optional priority ``order`` ids. A normal JSON array remains a standalone catalog.
    """
    p = catalog_path(path)
    if not p.is_file():
        raise ModelCatalogError(f"model catalog not found: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ModelCatalogError(f"cannot read model catalog {p}: {exc}") from exc
    if isinstance(data, dict) and data.get("extends") == "llm-mesh":
        base = json.loads((_REPO_ROOT / "models.json").read_text(encoding="utf-8"))
        models = {m["id"]: m for m in base}
        overrides = data.get("models", [])
        if not isinstance(overrides, list):
            raise ModelCatalogError("catalog models must be an array")
        for model in overrides:
            if not isinstance(model, dict) or not isinstance(model.get("id"), str):
                raise ModelCatalogError("catalog overrides require a string id")
            models[model["id"]] = model
        patches = data.get("patches", [])
        if not isinstance(patches, list):
            raise ModelCatalogError("catalog patches must be an array")
        seen = set()
        for patch in patches:
            if not isinstance(patch, dict) or not isinstance(patch.get("id"), str):
                raise ModelCatalogError("catalog patches require a string id")
            model_id = patch["id"]
            if model_id not in models or model_id in seen:
                raise ModelCatalogError(f"unknown or duplicate model patch: {model_id}")
            seen.add(model_id)
            models[model_id] = _patch_model(models[model_id], patch)
        excluded = data.get("exclude", [])
        if not isinstance(excluded, list) or any(not isinstance(i, str) for i in excluded):
            raise ModelCatalogError("catalog exclude must be an array of ids")
        for model_id in excluded:
            models.pop(model_id, None)
        order = data.get("order", list(models))
        if (not isinstance(order, list) or any(not isinstance(i, str) for i in order)
                or len(set(order)) != len(order) or not set(order) <= set(models)):
            raise ModelCatalogError("catalog order must contain unique known model ids")
        data = [models[model_id] for model_id in order]
        data.extend(model for model_id, model in models.items() if model_id not in set(order))
    if not isinstance(data, list):
        raise ModelCatalogError(
            f"{p}: expected JSON model array, got {type(data).__name__}"
        )
    if any(not isinstance(m, dict) for m in data):
        raise ModelCatalogError("catalog entries must be objects")
    return data


def load_catalog(path: str | os.PathLike[str] | None = None) -> list[dict]:
    """Read a catalog and expand it into provider routes."""
    data = load_catalog_data(path)
    return expand_catalog(data)


def catalog_digest(path: str | os.PathLike[str] | None = None) -> str:
    """Fingerprint effective data, including the base of an override catalog.

    Standalone arrays retain the historical file-byte digest.
    """
    p = catalog_path(path)
    raw = p.read_bytes()
    if isinstance(json.loads(raw), dict):
        raw = json.dumps(load_catalog_data(p), ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# Route resolution.


def _env(name: str | None) -> str:
    return os.environ.get(name or "", "").strip() if name else ""


def route_task(route: dict) -> str:
    """``chat`` when the route omits ``task``. An unknown task is rejected."""
    from llm_mesh.embeddings import normalize_task

    try:
        return normalize_task(route.get("task"))
    except ValueError as exc:
        raise ModelCatalogError(f"{route.get('id')!r}: {exc}") from exc


def route_model(route: dict) -> str:
    """Resolve the route model id from model, then the environment variable named by model_env."""
    return str(route.get("model") or "") or _env(route.get("model_env"))


def _missing_endpoint_route(route: dict) -> list[str]:
    """Missing pieces of a multi-server route. An empty list means it can run.

    The API key may sit on the route or on every endpoint. Each endpoint has
    its own base URL and ``max_concurrent``. A non-GigaChat model may sit on
    the route or on the endpoint that uses it.
    """
    raw = route.get("endpoints")
    if not isinstance(raw, list) or len(raw) < 2 or any(not isinstance(item, dict) for item in raw):
        return ["endpoints"]
    missing: list[str] = []
    kind = route.get("kind")
    key_name = route.get("api_key_env") or "LLM_API_KEY"
    route_key = bool(route.get("api_key") or _env(route.get("api_key_env")))
    route_model_ready = bool(route_model(route))
    if kind != "gigachat" and not route_model_ready:
        for index, endpoint in enumerate(raw):
            if endpoint.get("model") or _env(endpoint.get("model_env")):
                continue
            name = (
                endpoint.get("model_env")
                or route.get("model_env")
                or f"endpoints[{index}].model"
            )
            if name not in missing:
                missing.append(name)
    key_missing = False
    for index, endpoint in enumerate(raw):
        if not (endpoint.get("base_url") or _env(endpoint.get("base_url_env"))):
            missing.append(endpoint.get("base_url_env") or f"endpoints[{index}].base_url")
        if endpoint.get("max_concurrent", route.get("max_concurrent")) is None:
            missing.append(f"endpoints[{index}].max_concurrent")
        if not route_key and not (endpoint.get("api_key") or _env(endpoint.get("api_key_env"))):
            key_missing = True
    if key_missing:
        missing.append(key_name)
    return missing


def missing_credentials(route: dict) -> list[str]:
    """Return missing environment credentials; an empty list means the route is ready."""
    if route.get("endpoints") is not None:
        return _missing_endpoint_route(route)
    missing: list[str] = []
    if route.get("kind") == "gigachat":
        if not (route.get("api_key") or _env(route.get("api_key_env"))):
            missing.append(route.get("api_key_env") or "LLM_API_KEY")
        return missing
    if route.get("kind") in ("anthropic", "gemini"):
        # Both APIs have a fixed default endpoint, so a base URL is optional.
        # The key is the same neutral variable as every other kind.
        key_name = route.get("api_key_env") or "LLM_API_KEY"
        if not (route.get("api_key") or _env(key_name)):
            missing.append(key_name)
        if not route_model(route) and route.get("model_env"):
            missing.append(route["model_env"])
        return missing
    if not (route.get("base_url") or _env(route.get("base_url_env"))):
        missing.append(route.get("base_url_env") or "LLM_BASE_URL")
    if not (route.get("api_key") or _env(route.get("api_key_env"))):
        missing.append(route.get("api_key_env") or "LLM_API_KEY")
    # Account-specific model URIs are required configuration just like API keys.
    if not route_model(route) and route.get("model_env"):
        missing.append(route["model_env"])
    return missing


def _selector_parts(selector: str) -> tuple[str, str | None]:
    """`id` / `id@provider` / `id (provider)` → (id, provider|None)."""
    s = selector.strip()
    if s.endswith(")") and " (" in s:
        head, _, tail = s.rpartition(" (")
        return head.strip(), tail[:-1].strip()
    if "@" in s:
        head, _, tail = s.partition("@")
        return head.strip(), tail.strip()
    return s, None


def routes_for(selector: str, routes: list[dict] | None = None) -> list[dict]:
    """Return all routes for the selected model in catalog priority order."""
    routes = load_catalog() if routes is None else routes
    model_id, provider = _selector_parts(selector)
    found = [r for r in routes if r.get("id") == model_id]
    if not found:
        known = ", ".join(sorted({str(r.get("id")) for r in routes}))
        raise ModelCatalogError(
            f"model {model_id!r} not found in catalog. Known id: {known}"
        )
    if provider == "cluster":
        pooled = _provider_cluster_route(selector, found)
        if pooled is not None:
            return [pooled]
    if provider:
        found = [r for r in found if r.get("provider") == provider]
        if not found:
            raise ModelCatalogError(
                f"model {model_id!r} has no provider {provider!r}"
            )
    return found


def _validate_provider_cluster(model_id: object, cluster: object, names: list[str]) -> None:
    """``cluster`` names providers of one model that share a queue."""
    if (
        not isinstance(cluster, list)
        or len(cluster) < 2
        or any(not isinstance(name, str) or not name for name in cluster)
    ):
        raise ModelCatalogError(
            f"{model_id!r}: cluster must name at least two providers"
        )
    if len(set(cluster)) != len(cluster):
        raise ModelCatalogError(f"{model_id!r}: cluster repeats a provider")
    unknown = [name for name in cluster if name not in names]
    if unknown:
        raise ModelCatalogError(
            f"{model_id!r}: cluster names unknown provider {unknown[0]!r}"
        )


def _is_member_cluster(route: dict) -> bool:
    """True when ``cluster`` holds full provider routes, not just their names."""
    members = route.get("cluster")
    return (
        isinstance(members, list)
        and len(members) >= 2
        and all(isinstance(member, dict) for member in members)
    )


def _provider_cluster_route(selector: str, found: list[dict]) -> dict | None:
    """The bare model id's pool, or None when this model is not a cluster.

    A named provider (``id@name``) does not come here. Members that are not
    ready fail the cluster; they are not dropped and not replaced by a
    provider outside the list.
    """
    names = found[0].get("cluster") if found else None
    if not isinstance(names, list) or not names or not isinstance(names[0], str):
        return None
    by_name = {route.get("provider"): route for route in found}
    missing: list[str] = []
    members: list[dict] = []
    for name in names:
        route = by_name.get(name)
        if route is None:
            raise ModelCatalogError(
                f"{found[0].get('id')!r}: cluster names unknown provider {name!r}"
            )
        gap = missing_credentials(route)
        if gap:
            missing.append(f"{route.get('label')}: missing {', '.join(gap)}")
            continue
        members.append({key: value for key, value in route.items() if key != "cluster"})
    if missing:
        raise ModelCatalogError(
            f"no route for model {selector!r} is ready to run ({'; '.join(missing)})"
        )
    return {
        "id": found[0].get("id"),
        "label": f"{found[0].get('id')} (cluster)",
        "provider": "cluster",
        "cluster": members,
    }


def resolve_route(selector: str, routes: list[dict] | None = None) -> dict:
    """Choose the first route with available credentials, following provider order. If none is
    ready, raise ModelCatalogError listing missing variables; never silently substitute another
    model or gateway.

    A model that declares ``cluster`` is the exception for a bare id: the result is
    that pool of providers, not the first ready provider. ``id@cluster`` is the
    same pool. ``id@provider`` is still that one provider.
    """
    found = routes_for(selector, routes)
    _model_id, provider = _selector_parts(selector)
    if len(found) == 1 and _is_member_cluster(found[0]):
        return found[0]
    if provider is None:
        pooled = _provider_cluster_route(selector, found)
        if pooled is not None:
            return pooled
    for route in found:
        if not missing_credentials(route):
            return route
    detail = "; ".join(
        f"{r['label']}: missing {', '.join(missing_credentials(r))}" for r in found
    )
    raise ModelCatalogError(
        f"no route for model {selector!r} is ready to run ({detail})"
    )


# Route environment and client construction.


def route_env(route: dict) -> dict[str, str]:
    """Build neutral connection settings and one complete request-options object.

    A route with ``endpoints`` or a provider ``cluster`` is a pool of servers.
    One exported environment cannot name them; build that route with ``make_client``.
    """
    if route.get("endpoints") is not None or _is_member_cluster(route):
        raise ModelCatalogError(
            f"{route.get('id')!r}: a route with endpoints or a provider cluster "
            "is a pool of inference servers; build it with make_client, "
            "not with an exported environment"
        )
    kind = route.get("kind")
    if kind not in VALID_KINDS:
        raise ModelCatalogError(
            f"{route.get('id')!r}: kind={kind!r}, expected one of {VALID_KINDS}"
        )
    model = route_model(route)
    if not model:
        raise ModelCatalogError(f"{route.get('id')!r}: missing model/model_env")
    env = {key: "" for key in _MANAGED_ENV}
    env.update({
        "LLM_PROVIDER": kind,
        "LLM_MODEL": model,
        "LLM_PROVIDER_LABEL": str(route.get("provider") or kind),
        "LLM_API_KEY": str(route.get("api_key") or "") or _env(route.get("api_key_env")),
        "LLM_BASE_URL": str(route.get("base_url") or "") or _env(route.get("base_url_env")),
        "LLM_OPTIONS": json.dumps(route_options(route), ensure_ascii=False),
    })
    if kind == "gigachat":
        env["LLM_AUTH_SCOPE"] = str(route.get("scope") or "") or _env(route.get("scope_env"))
        env["LLM_AUTH_URL"] = str(route.get("auth_url") or "") or _env(route.get("auth_url_env"))
    if route.get("verify_ssl") is not None:
        env["LLM_VERIFY_SSL"] = "true" if route["verify_ssl"] else "false"
    return env


def apply_route_env(route: dict) -> dict[str, str]:
    """Apply and return the complete route environment. An explicit catalog selection overrides
    managed manual settings so model, endpoint, and behavior flags cannot come from different
    configurations.
    """
    env = route_env(route)
    os.environ.update(env)
    return env


def _apply_pool_env(route: dict) -> None:
    """Write a pool route's shared settings. Do not invent one server URL.

    ``route_env`` would set ``LLM_BASE_URL`` to an empty string and erase a
    URL already in the process. The same for ``LLM_MODEL`` when the route
    itself names none: each endpoint carries its own model.
    """
    shared = {key: value for key, value in route.items() if key != "endpoints"}
    named_model = bool(route_model(shared))
    if not named_model:
        shared = {**shared, "model": "-"}
    env = route_env(shared)
    if not named_model:
        env.pop("LLM_MODEL", None)
    if not env.get("LLM_BASE_URL"):
        env.pop("LLM_BASE_URL", None)
    os.environ.update(env)


def _build_gigachat_client(route: dict, env: dict) -> Any:
    """Build the GigaChat client. Batch mode swaps in the coalescing adapter."""
    from llm_mesh.gigachat import GigaChatClient

    model = env["LLM_MODEL"]
    # Supply resolved connection settings explicitly to the selected provider.
    kwargs: dict[str, Any] = {"model": model}
    if env.get("LLM_PROVIDER_LABEL"):
        kwargs["label"] = env["LLM_PROVIDER_LABEL"]
    if env.get("LLM_API_KEY"):
        kwargs["credentials"] = env["LLM_API_KEY"]
    if env.get("LLM_AUTH_SCOPE"):
        kwargs["scope"] = env["LLM_AUTH_SCOPE"]
    if env.get("LLM_BASE_URL"):
        kwargs["api_url"] = env["LLM_BASE_URL"]
    if env.get("LLM_AUTH_URL"):
        kwargs["auth_url"] = env["LLM_AUTH_URL"]
    if route.get("http_timeout"):
        kwargs["timeout_s"] = float(route["http_timeout"])
    from llm_mesh.gigachat.batch import batch_mode_enabled, get_batching_client

    if batch_mode_enabled() and route_task(route) == "chat":
        return get_batching_client(
            model,
            credentials=kwargs.get("credentials"),
            scope=kwargs.get("scope"),
        )
    kwargs.update(_concurrency_kwargs(route))
    return GigaChatClient(**kwargs)


def _build_gemini_client(route: dict, env: dict) -> Any:
    """Build the Gemini client. The import stays inside the builder."""
    from llm_mesh.gemini import GeminiClient

    kwargs: dict[str, Any] = {
        "model": env["LLM_MODEL"],
        "label": env.get("LLM_PROVIDER_LABEL") or None,
    }
    if env.get("LLM_API_KEY"):
        kwargs["api_key"] = env["LLM_API_KEY"]
    if env.get("LLM_BASE_URL"):
        kwargs["base_url"] = env["LLM_BASE_URL"]
    if route.get("http_timeout"):
        kwargs["http_timeout"] = float(route["http_timeout"])
    kwargs.update(_concurrency_kwargs(route))
    return GeminiClient(**kwargs)


def _build_anthropic_client(route: dict, env: dict) -> Any:
    """Build the Anthropic client. The import stays inside the builder."""
    from llm_mesh.anthropic import AnthropicClient

    model = env["LLM_MODEL"]
    anthropic_kwargs: dict[str, Any] = {
        "model": model,
        "label": env.get("LLM_PROVIDER_LABEL") or None,
    }
    if env.get("LLM_API_KEY"):
        anthropic_kwargs["api_key"] = env["LLM_API_KEY"]
    if env.get("LLM_BASE_URL"):
        anthropic_kwargs["base_url"] = env["LLM_BASE_URL"]
    if route.get("http_timeout"):
        anthropic_kwargs["http_timeout"] = float(route["http_timeout"])
    anthropic_kwargs.update(_concurrency_kwargs(route))
    return AnthropicClient(**anthropic_kwargs)


def _build_openai_client(route: dict, env: dict) -> Any:
    """Build the OpenAI-compatible client. Batch mode wraps it in the coalescer.

    The GigaChat singleton registry is not reused: that registry builds a GigaChat
    batch client. OpenAI gets its own batch client around the route's OpenAIClient.
    """
    from llm_mesh.gigachat.batch import BatchingLLMClient, batch_mode_enabled
    from llm_mesh.openai import OpenAIClient
    from llm_mesh.openai.batch import OpenAIBatchClient

    model = env["LLM_MODEL"]
    client = OpenAIClient(
        model=model,
        base_url=env["LLM_BASE_URL"],
        api_key=env["LLM_API_KEY"],
        label=env.get("LLM_PROVIDER_LABEL") or None,
        http_timeout=float(route["http_timeout"]) if route.get("http_timeout") else None,
        tiktoken_encoding=str(route["tiktoken_encoding"]) if route.get("tiktoken_encoding") else None,
        **_concurrency_kwargs(route),
    )
    if batch_mode_enabled() and route_task(route) == "chat":
        return BatchingLLMClient(OpenAIBatchClient(client=client), model=model)
    return client


# kind -> builder(route, env). Imports stay inside the builders so loading the
# catalog does not import every provider, and so this module is not imported
# while a provider package is still initializing.
_ROUTE_CLIENT_BUILDERS: dict[str, Callable[[dict, dict], Any]] = {
    "anthropic": _build_anthropic_client,
    "gemini": _build_gemini_client,
    "gigachat": _build_gigachat_client,
    "openai": _build_openai_client,
}

VALID_KINDS = tuple(sorted(_ROUTE_CLIENT_BUILDERS))


_ENDPOINT_FIELDS = frozenset({
    "base_url", "base_url_env", "api_key", "api_key_env",
    "model", "model_env", "max_concurrent", "max_concurrent_streams",
})


def _concurrency_kwargs(route: dict) -> dict[str, int]:
    """Constructor limits from a route. Empty when the route names neither."""
    from llm_mesh.concurrency import validate_max_concurrent

    kwargs: dict[str, int] = {}
    for name in ("max_concurrent", "max_concurrent_streams"):
        if route.get(name) is None:
            continue
        try:
            value = validate_max_concurrent(route[name], name=name)
        except ValueError as exc:
            raise ModelCatalogError(f"{route.get('id')!r}: {exc}") from exc
        if value is not None:
            kwargs[name] = value
    return kwargs


def _expand_endpoints(route: dict) -> list[dict]:
    """One ordinary route per inference server. The parent ``endpoints`` key is dropped."""
    from llm_mesh.concurrency import validate_max_concurrent

    raw = route.get("endpoints")
    label = route.get("id") or route.get("label") or "route"
    if route.get("base_url") or route.get("base_url_env"):
        raise ModelCatalogError(f"{label!r}: set either base_url or endpoints, not both")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ModelCatalogError(
            f"{label!r}: endpoints must be a list of at least two inference servers"
        )
    replicas: list[dict] = []
    seen: list[tuple[str, str]] = []
    for index, endpoint in enumerate(raw):
        if not isinstance(endpoint, dict):
            raise ModelCatalogError(f"{label!r}: endpoints[{index}] must be an object")
        unknown = sorted(set(endpoint) - _ENDPOINT_FIELDS)
        if unknown:
            raise ModelCatalogError(
                f"{label!r}: endpoints[{index}] has unknown fields {unknown}"
            )
        replica = {key: value for key, value in route.items() if key != "endpoints"}
        for key, value in endpoint.items():
            if value is not None:
                replica[key] = value
        base = str(replica.get("base_url") or "") or _env(replica.get("base_url_env"))
        if not base.strip():
            raise ModelCatalogError(f"{label!r}: endpoints[{index}] has no base_url")
        try:
            validate_max_concurrent(replica.get("max_concurrent"))
        except ValueError as exc:
            raise ModelCatalogError(f"{label!r}: endpoints[{index}]: {exc}") from exc
        if replica.get("max_concurrent") is None:
            raise ModelCatalogError(
                f"{label!r}: endpoints[{index}] needs max_concurrent"
            )
        if replica.get("max_concurrent_streams") is not None:
            try:
                validate_max_concurrent(
                    replica["max_concurrent_streams"], name="max_concurrent_streams",
                )
            except ValueError as exc:
                raise ModelCatalogError(f"{label!r}: endpoints[{index}]: {exc}") from exc
        key_material = str(replica.get("api_key") or "") or _env(replica.get("api_key_env"))
        identity = (base.strip().rstrip("/").lower(), key_material)
        if identity in seen:
            raise ModelCatalogError(
                f"{label!r}: endpoints[{index}] repeats inference server {base.strip()}"
            )
        seen.append(identity)
        replicas.append(replica)
    return replicas


def _construct_client(route: dict, env: dict) -> Any:
    """Build the provider client for one route. Does not touch the process environment."""
    kind = route.get("kind")
    builder = _ROUTE_CLIENT_BUILDERS.get(kind) if isinstance(kind, str) else None
    if builder is None:
        raise ModelCatalogError(
            f"{route.get('id')!r}: kind={kind!r}, expected one of {VALID_KINDS}"
        )
    task = route_task(route)
    if task == "rerank" and kind != "openai":
        raise ModelCatalogError(
            f"{route.get('id')!r}: task 'rerank' is implemented for kind 'openai'"
        )
    return builder(route, env)


def _bind_catalog_client(route: dict, client: Any) -> Any:
    bind = getattr(client, "bind_catalog_route", None)
    if bind is not None:
        try:
            bind(route)
        except ValueError as exc:
            raise ModelCatalogError(f"{route.get('id')!r}: {exc}") from exc
    return client


def _construct_isolated(route: dict) -> Any:
    """Build one client from ``route`` without keeping its env in the process.

    Call parameters are read from the process environment at construction.
    A pool member has to see its own route there, then the previous environment
    comes back, so the next member does not inherit them.
    """
    saved = {key: os.environ.get(key) for key in _MANAGED_ENV}
    try:
        env = route_env(route)
        os.environ.update(env)
        return _construct_client(route, env)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _clients_for_member(route: dict) -> list[Any]:
    """Clients for one provider route. An ``endpoints`` route contributes each server."""
    label = route.get("label") or route.get("provider") or route.get("id") or "provider"
    if route.get("endpoints") is not None:
        replicas = _expand_endpoints(route)
        return [
            _bind_catalog_client(replica, _construct_isolated(replica))
            for replica in replicas
        ]
    if route.get("max_concurrent") is None:
        raise ModelCatalogError(f"{label!r}: needs max_concurrent to join a cluster")
    return [_bind_catalog_client(route, _construct_isolated(route))]


def _abandon_clients(clients: list[Any]) -> None:
    """Close clients a failed pool build will not return.

    Construction does not open HTTP, so this is a no-op until a member has a
    transport. The limit rollback is separate and always runs.
    """
    open_clients = [client for client in clients if getattr(client, "_client", None) is not None]

    async def _close() -> None:
        for client in open_clients:
            try:
                await client.aclose()
            except Exception:
                logger.warning(
                    "llm-mesh: could not close %s after a failed pool build",
                    type(client).__name__,
                )

    if not open_clients:
        return
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(_close())
        return
    asyncio.get_running_loop().create_task(_close())


def _make_provider_cluster(route: dict) -> Any:
    """One queue across provider routes. Each member keeps its own call parameters."""
    from llm_mesh.concurrency import limits_snapshot, restore_limits
    from llm_mesh.gigachat.batch import batch_mode_enabled
    from llm_mesh.pool import InferencePool

    members: list[dict] = route["cluster"]
    if batch_mode_enabled() and any(route_task(member) == "chat" for member in members):
        raise ModelCatalogError(
            f"{route.get('id')!r}: a provider cluster dispatches live calls; "
            "turn LLM_BATCH_MODE off"
        )
    before = limits_snapshot()
    clients: list[Any] = []
    try:
        for member in members:
            clients.extend(_clients_for_member(member))
        pool = InferencePool(clients)
    except ValueError as exc:
        _abandon_clients(clients)
        restore_limits(before)
        raise ModelCatalogError(f"{route.get('id')!r}: {exc}") from exc
    except Exception:
        _abandon_clients(clients)
        restore_limits(before)
        raise
    task = getattr(clients[0], "_catalog_task", None)
    if task is not None:
        pool._catalog_task = task
    return pool


def make_client(route: dict) -> Any:
    """Construct the route's LLM client and apply its environment configuration.

    A route with ``endpoints`` returns an inference pool of one provider: one
    shared queue, one client per server, each capped by that server's
    ``max_concurrent``. A route whose ``cluster`` is a list of provider routes
    returns the same kind of pool across providers. Those members keep their
    own kind, credentials, model, and request options. The process environment
    is left as it was; there is no single server to publish.
    """
    if _is_member_cluster(route):
        return _make_provider_cluster(route)
    if route.get("endpoints") is None:
        return _bind_catalog_client(route, _construct_client(route, apply_route_env(route)))
    from llm_mesh.gigachat.batch import batch_mode_enabled
    from llm_mesh.pool import InferencePool

    if batch_mode_enabled() and route_task(route) == "chat":
        raise ModelCatalogError(
            f"{route.get('id')!r}: endpoints dispatch live calls; "
            "turn LLM_BATCH_MODE off or use a single base_url"
        )
    from llm_mesh.concurrency import limits_snapshot, restore_limits

    replicas = _expand_endpoints(route)
    _apply_pool_env(route)
    before = limits_snapshot()
    clients: list[Any] = []
    try:
        for replica in replicas:
            clients.append(_construct_isolated(replica))
        client = InferencePool(clients)
    except ValueError as exc:
        _abandon_clients(clients)
        restore_limits(before)
        raise ModelCatalogError(f"{route.get('id')!r}: {exc}") from exc
    except Exception:
        _abandon_clients(clients)
        restore_limits(before)
        raise
    return _bind_catalog_client(route, client)


def selected_model_id() -> str:
    """Read LLM_MODEL_ID; an empty value means no catalog selection."""
    return os.environ.get("LLM_MODEL_ID", "").strip()


def make_selected_client() -> Any | None:
    """Return the client selected by LLM_MODEL_ID, or None to retain the caller's manual
    LLM_PROVIDER/LLM_MODEL configuration.
    """
    selector = selected_model_id()
    if not selector:
        return None
    route = resolve_route(selector)
    if _is_member_cluster(route):
        names = ", ".join(
            str(member.get("provider") or member.get("kind")) for member in route["cluster"]
        )
        logger.info("LLM from model catalog: %s (cluster of %s)", route["label"], names)
    else:
        logger.info(
            "LLM from model catalog: %s (kind=%s, model=%s)",
            route["label"], route.get("kind"), route_model(route),
        )
    return make_client(route)


def _format_check_line(result: Any) -> str:
    """One stdout line for ``--check``. Failure text is capped at 120 characters."""
    if result.ok:
        latency = 0 if result.latency_ms is None else result.latency_ms
        return f"{result.label}  [ok]  {latency}ms"
    error = (result.error or "")[:120]
    return f"{result.label}  [FAIL] {result.error_type}: {error}"


def _cli_check(selector: str, routes: list[dict]) -> int:
    """Probe the selected route. Exit 0 when a probed route is ok.

    A bare model id checks every provider, unless the model declares
    ``cluster``: then the probe is that pool, and providers outside the list
    are not called. ``id@cluster`` is the same pool. ``id@provider`` checks
    that one route. The header is printed because, unlike ``--list``, this
    command sends real requests.
    """
    from llm_mesh.probe import check_routes

    print("--check makes real API calls (unlike --list).")
    try:
        results = check_routes(selector, routes)
    except ModelCatalogError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    for result in results:
        print(_format_check_line(result))
    return 0 if any(result.ok for result in results) else 1


# ───────────────────────────────── CLI ────────────────────────────────────


def _cli(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="llm-mesh",
        description="LLM model catalog (models.json)",
    )
    ap.add_argument("--config", default="", help="catalog path (defaults to bundled models.json)")
    ap.add_argument("--list", action="store_true", help="list all routes")
    ap.add_argument("--env", metavar="ID", default="",
                    help="print shell export statements")
    ap.add_argument("--json", metavar="ID", default="",
                    help="print the route as JSON")
    ap.add_argument(
        "--check", metavar="ID", default="",
        help=(
            "probe id or id@provider with a real API call "
            "(unlike --list). Exit 0 when a checked route succeeds"
        ),
    )
    args = ap.parse_args(argv)

    routes = load_catalog(args.config or None)
    if args.env or args.json:
        route = resolve_route(args.env or args.json, routes)
        if args.json:
            print(json.dumps(route, ensure_ascii=False, indent=2))
            return 0
        try:
            exported = route_env(route)
        except ModelCatalogError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        for key, value in sorted(exported.items()):
            print(f"export {key}={shlex.quote(value)}")
        return 0

    if args.check:
        return _cli_check(args.check, routes)

    if not args.list:
        ap.print_help()
        return 2
    for route in routes:
        missing = missing_credentials(route)
        status = "ready" if not missing else "missing " + ",".join(missing)
        title = route.get("title") or route["id"]
        extra = []
        if route.get("eval_concurrency"):
            extra.append(f"eval -c {route['eval_concurrency']}")
        if route.get("notes"):
            extra.append(str(route["notes"]))
        tail = ("  # " + "; ".join(extra)) if extra else ""
        print(f"{route['label']:<44} {title:<34} [{status}]{tail}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
