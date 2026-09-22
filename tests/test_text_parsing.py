"""Regression tests for tolerant text JSON extraction, including valid fenced objects followed by
prose that previously caused Extra data failures.
"""

from __future__ import annotations

import json

import pytest

from llm_mesh.text_parsing import (
    extract_json_fence,
    extract_json_fence_lenient,
    extract_json_fence_tolerant,
    extract_json_from_text,
    looks_degenerate_repetition,
    strip_code_fence,
)


def test_extract_json_fence_closed_or_raw():
    assert extract_json_fence('prefix ```json\n{"a": 1}\n``` suffix') == '{"a": 1}'
    assert extract_json_fence('  {"a": 1}  ') == '{"a": 1}'
    # Closed-only tier intentionally keeps an unclosed fence as raw text.
    assert extract_json_fence('```json\n{"a": 1}') == '```json\n{"a": 1}'


def test_extract_json_fence_tolerant_accepts_unclosed_fence():
    assert extract_json_fence_tolerant('```json\n{"a": 1}') == '{"a": 1}'
    assert extract_json_fence_tolerant('```\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_fence_tolerant('  {"a": 1}  ') == '{"a": 1}'


def test_extract_json_fence_lenient_repairs_invalid_escapes():
    assert extract_json_fence_lenient('```json\n{"pattern":"\\d+"}') == (
        r'{"pattern":"\\d+"}'
    )
    # Correct JSON escapes remain unchanged.
    assert extract_json_fence_lenient(r'{"line":"a\nb"}') == r'{"line":"a\nb"}'


def test_plain_valid_json():
    assert extract_json_from_text('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_fenced_valid_json():
    text = '```json\n{"artifact_kind": "forms", "ok": true}\n```'
    assert extract_json_from_text(text) == {"artifact_kind": "forms", "ok": True}


def test_extra_data_after_object_top_level():
    """Extract the first valid object before unfenced trailing prose."""
    text = '{"artifact_kind": "forms", "form_id": "F1"}\n\nDone, the form is ready.'
    assert extract_json_from_text(text) == {"artifact_kind": "forms", "form_id": "F1"}


def test_extra_data_inside_fence():
    """Recover an object followed by an explanation inside a fence."""
    payload = (
        '```json\n'
        '{\n'
        '  "artifact_kind": "forms",\n'
        '  "forms": {\n'
        '    "form_id": "Event_081ei5r",\n'
        '    "operation_mode": "create",\n'
        '    "full_schema": {"name": "Check account details", "xsdContent": []}\n'
        '  }\n'
        '}\n'
        'This form contains full name and tax ID fields.\n'
        '```'
    )
    result = extract_json_from_text(payload)
    assert result["artifact_kind"] == "forms"
    assert result["forms"]["form_id"] == "Event_081ei5r"
    assert result["forms"]["full_schema"]["name"] == "Check account details"


def test_extra_data_second_object():
    """Keep the first object when the model repeats it."""
    text = '{"a": 1}\n{"a": 2}'
    assert extract_json_from_text(text) == {"a": 1}


def test_invalid_escape_still_works():
    """Repair invalid single-backslash escapes in model prose."""
    text = r'{"pattern": "\d+ digits"}'
    result = extract_json_from_text(text)
    assert result["pattern"] in (r"\d+ digits", r"\\d+ digits".replace("\\\\", "\\"))


def test_no_json_raises():
    with pytest.raises(RuntimeError):
        extract_json_from_text("plain text without json")


def test_raw_decode_does_not_swallow_real_truncation():
    """A genuinely incomplete object cannot be rescued by raw_decode; leave it to later repair."""
    # Return a repaired dictionary or raise RuntimeError for unrecoverable truncation; never
    # fail silently.
    text = '{"a": 1, "b": [1, 2, 3'
    try:
        result = extract_json_from_text(text)
        assert isinstance(result, dict)
    except RuntimeError:
        pass  # Unrecoverable JSON may legitimately fail without repair support.


def test_function_call_control_marker_stripped_bare_args():
    """Strip the function-call control marker before parsing bare arguments."""
    text = '<|function_call|>{"form_id": "f1", "event_kind": "ON_OPEN"}'
    assert extract_json_from_text(text) == {"form_id": "f1", "event_kind": "ON_OPEN"}


def test_function_call_control_marker_stripped_before_fence():
    """A control marker before a fence must not prevent fence extraction."""
    text = '<|function_call|>```json\n{"a": 1}\n```'
    assert extract_json_from_text(text) == {"a": 1}


# --- Repetition detection protects length retries ---------------------------


def test_degenerate_repetition_detected():
    """Detect pathological repetition that cannot be repaired by increasing the token budget."""
    assert looks_degenerate_repetition("}\n" * 3000)
    assert looks_degenerate_repetition('{"a":1},' * 1000)
    assert looks_degenerate_repetition("approved " * 800)
    # Repeat one identical form-field line, a common generation-loop pattern.
    assert looks_degenerate_repetition('  "field": {"type": "string"},\n' * 200)


def test_legitimate_large_output_not_flagged():
    """Do not flag large valid output with varied values as repetitive; normal length retries must
    remain available.
    """
    fields = json.dumps(
        [{"id": f"field_{i}", "type": ["string", "number", "date", "bool"][i % 4],
          "label": f"Characteristic of object number {i} in section {i * 7 % 13}",
          "required": bool(i % 2)} for i in range(120)],
        ensure_ascii=False, indent=2,
    )
    assert not looks_degenerate_repetition(fields)
    bpmn = "".join(
        f'<bpmn:userTask id="Activity_{i}" name="Approval task {i}"/>\n'
        for i in range(100)
    )
    assert not looks_degenerate_repetition(bpmn)


def test_short_output_not_judged():
    """Do not classify short output as degenerate, since a larger budget may still help."""
    assert not looks_degenerate_repetition("}\n" * 10)
    assert not looks_degenerate_repetition("")
    assert not looks_degenerate_repetition(None)


# --- Strip outer code fences before previews or executable-code persistence --


def test_strip_code_fence_removes_single_wrap():
    raw = "```groovy\ndef foo() {\n    println \"hi\"\n}\n```"
    assert strip_code_fence(raw) == 'def foo() {\n    println "hi"\n}'


def test_strip_code_fence_removes_wrap_without_language_tag():
    raw = "```\nconst x = 1;\n```"
    assert strip_code_fence(raw) == "const x = 1;"


def test_strip_code_fence_idempotent_on_clean_code():
    clean = 'def foo() {\n    println "hi"\n}'
    assert strip_code_fence(clean) == clean


def test_strip_code_fence_preserves_backticks_in_middle_of_code():
    """Preserve backticks embedded in legitimate source code; remove a fence only when it wraps the
    entire string.
    """
    mid = 'def foo() {\n    def s = "some ```code``` inline"\n    return s\n}'
    assert strip_code_fence(mid) == mid


def test_strip_code_fence_handles_javascript_tag():
    raw = "```javascript\nfunction onOpen() { form.setField('x', 1); }\n```"
    assert strip_code_fence(raw) == "function onOpen() { form.setField('x', 1); }"
