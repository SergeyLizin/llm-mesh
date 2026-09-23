import hashlib
import json
from pathlib import Path

import pytest

import llm_mesh
from llm_mesh import models_catalog as catalog


def test_bundled_catalog_is_available_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODELS_CONFIG", raising=False)
    path = catalog.catalog_path()
    assert path.parent == Path(llm_mesh.__file__).parent
    models = catalog.load_catalog_data()
    assert models and catalog.load_catalog()
    assert all("api_key" not in route for route in catalog.load_catalog())


def test_overrides_replace_add_exclude_and_preserve_order(tmp_path):
    base = catalog.load_catalog_data(Path(llm_mesh.__file__).parent / "models.json")
    first, second = base[0]["id"], base[1]["id"]
    custom = {"id": "custom", "kind": "openai", "model": "local"}
    replacement = {"id": first, "kind": "openai", "model": "replacement"}
    keep = [m["id"] for m in base if m["id"] != second]
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"extends": "llm-mesh", "models": [custom, replacement],
                                "exclude": [second], "order": ["custom", *keep]}))
    actual = catalog.load_catalog_data(path)
    assert actual[0] == custom
    assert actual[1] == replacement
    assert [m["id"] for m in actual] == ["custom", *keep]
    assert catalog.load_catalog_data(Path(llm_mesh.__file__).parent / "models.json") == base


def test_project_example_adds_models_to_public_catalog(tmp_path, monkeypatch):
    public_path = Path(llm_mesh.__file__).parent / "models.json"
    original = public_path.read_bytes()
    public_models = catalog.load_catalog_data(public_path)
    example = Path(__file__).resolve().parents[1] / "examples" / "models.json"
    project_path = tmp_path / "models.json"
    project_path.write_bytes(example.read_bytes())
    monkeypatch.setenv("LLM_MODELS_CONFIG", str(project_path))
    monkeypatch.setenv("CUSTOM_LLM_MODEL", "my-custom-model")
    monkeypatch.setenv("CUSTOM_LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CUSTOM_LLM_API_KEY", "test-key")

    combined = catalog.load_catalog_data()
    assert combined[:-1] == public_models
    assert combined[-1]["id"] == "custom-model"
    routes = catalog.load_catalog()
    custom = catalog.resolve_route("custom-model@custom", routes)
    assert catalog.route_model(custom) == "my-custom-model"
    assert catalog.expand_catalog(public_models) == routes[:-1]
    assert public_path.read_bytes() == original


def test_catalog_path_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODELS_CONFIG", raising=False)
    local = tmp_path / "models.json"
    local.write_text('[{"id":"local"}]')
    assert catalog.catalog_path().parent == Path(llm_mesh.__file__).parent
    assert catalog.load_catalog_data() != [{"id": "local"}]
    other = tmp_path / "other.json"
    other.write_text('[{"id":"other"}]')
    monkeypatch.setenv("LLM_MODELS_CONFIG", str(other))
    assert catalog.load_catalog_data() == [{"id": "other"}]
    assert catalog.load_catalog_data(local) == [{"id": "local"}]


def test_standalone_catalog_keeps_byte_digest(tmp_path):
    path = tmp_path / "models.json"
    raw = b'[ {"id":"custom"} ]\n'
    path.write_bytes(raw)
    assert catalog.catalog_digest(path) == hashlib.sha256(raw).hexdigest()


def test_override_digest_includes_bundled_data(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    base = bundle / "models.json"
    base.write_text('[{"id":"a","model":"one"}]')
    monkeypatch.setattr(catalog, "_REPO_ROOT", bundle)
    path = tmp_path / "override.json"
    path.write_text('{"extends":"llm-mesh"}')
    first = catalog.catalog_digest(path)
    base.write_text('[{"id":"a","model":"two"}]')
    assert first != catalog.catalog_digest(path)


@pytest.mark.parametrize("data", [
    {}, [42], {"extends": "llm-mesh", "models": {}},
    {"extends": "llm-mesh", "models": [{}]},
    {"extends": "llm-mesh", "order": ["missing"]},
    {"extends": "llm-mesh", "order": [{}]},
    {"extends": "llm-mesh", "exclude": "oops"},
])
def test_invalid_catalogs_raise_catalog_error(data, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data))
    with pytest.raises(catalog.ModelCatalogError):
        catalog.load_catalog_data(path)


def test_cli_exports_quote_shell_metacharacters(tmp_path, monkeypatch, capsys):
    import shlex
    path = tmp_path / "models.json"
    path.write_text('[{"id":"test","kind":"openai","provider":"test"}]')
    monkeypatch.setattr(catalog, "resolve_route", lambda selector, routes: routes[0])
    value = "literal $HOME `whoami` $(id) with spaces"
    monkeypatch.setattr(catalog, "route_env", lambda route: {"LLM_MODEL": value})
    assert catalog._cli(["--config", str(path), "--env", "test"]) == 0
    assert capsys.readouterr().out == f"export LLM_MODEL={shlex.quote(value)}\n"


def test_patches_inherit_future_fields_and_providers(tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    base = [{'id': 'a', 'kind': 'openai', 'providers': [
        {'name': 'one', 'model': 'm', 'extra_body': {'old': True}, 'notes': 'base'},
        {'name': 'removed', 'model': 'r'}]}]
    source = bundle / 'models.json'
    source.write_text(json.dumps(base))
    monkeypatch.setattr(catalog, '_REPO_ROOT', bundle)
    path = tmp_path / 'local.json'
    path.write_text(json.dumps({'extends': 'llm-mesh', 'patches': [{
        'id': 'a', 'notes': 'local', 'exclude_providers': ['removed'],
        'providers': [{'name': 'one', 'notes': None, 'extra_body': {'new': True}}],
    }], 'order': ['a']}))
    before = catalog.catalog_digest(path)
    base[0]['providers'][0]['response_format'] = 'json_object'
    base[0]['providers'].append({'name': 'new', 'model': 'n'})
    base.append({'id': 'b', 'model': 'second'})
    source.write_text(json.dumps(base))
    result = catalog.load_catalog_data(path)
    assert [m['id'] for m in result] == ['a', 'b']
    assert result[0]['notes'] == 'local'
    assert [p['name'] for p in result[0]['providers']] == ['one', 'new']
    assert result[0]['providers'][0] == {
        'name': 'one', 'model': 'm', 'extra_body': {'new': True}, 'response_format': 'json_object'}
    assert catalog.catalog_digest(path) != before
    assert json.loads(source.read_text()) == base


def test_expand_catalog_rejects_a_nameless_provider():
    with pytest.raises(catalog.ModelCatalogError, match="no name"):
        catalog.expand_catalog([{"id": "m", "providers": [{}]}])
    with pytest.raises(catalog.ModelCatalogError, match="no id"):
        catalog.expand_catalog([{"providers": [{"name": "p"}]}])


@pytest.mark.parametrize('patches', [None, {}, [{}], [{'id': 'missing'}],
    [{'id': 'a'}, {'id': 'a'}], [{'id': 'a', 'providers': {}}],
    [{'id': 'a', 'providers': [{}]}],
    [{'id': 'a', 'providers': [{'name': 'p'}, {'name': 'p'}]}],
    [{'id': 'a', 'exclude_providers': 'p'}],
    [{'id': 'a', 'exclude_providers': ['p']}],
])
def test_invalid_patches_fail_loudly(patches, tmp_path, monkeypatch):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    (bundle / 'models.json').write_text('[{"id":"a","providers":[{"name":"p"}]}]')
    monkeypatch.setattr(catalog, '_REPO_ROOT', bundle)
    path = tmp_path / 'local.json'
    path.write_text(json.dumps({'extends': 'llm-mesh', 'patches': patches}))
    with pytest.raises(catalog.ModelCatalogError):
        catalog.load_catalog_data(path)
