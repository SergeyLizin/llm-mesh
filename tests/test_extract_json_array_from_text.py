"""Regression tests for top-level array extraction. An object-only parser can silently return the
first object inside an otherwise correct array and lose the remaining items.
"""
import pytest

from llm_mesh.text_parsing import extract_json_array_from_text


def test_plain_json_array():
    got = extract_json_array_from_text('[{"id": "N0"}, {"id": "N1"}]')
    assert got == [{"id": "N0"}, {"id": "N1"}]


def test_does_not_truncate_to_first_element():
    # The original object parser reduced this array to its first item.
    got = extract_json_array_from_text(
        '[{"id": "N0", "x": 1}, {"id": "N1", "x": 2}, {"id": "N2", "x": 3}]'
    )
    assert len(got) == 3
    assert [item["id"] for item in got] == ["N0", "N1", "N2"]


def test_markdown_fenced_array():
    text = '```json\n[{"id": "N0"}, {"id": "N1"}]\n```'
    got = extract_json_array_from_text(text)
    assert got == [{"id": "N0"}, {"id": "N1"}]


def test_array_with_surrounding_prose():
    text = 'Here is the result:\n[{"id": "N0"}, {"id": "N1"}]\nI hope, this helps.'
    got = extract_json_array_from_text(text)
    assert got == [{"id": "N0"}, {"id": "N1"}]


def test_nested_objects_with_braces_inside_strings():
    # Ignore braces and brackets inside quoted strings while balancing delimiters.
    text = '[{"phrase": "text with { braces } inside"}, {"phrase": "and ] too"}]'
    got = extract_json_array_from_text(text)
    assert got == [
        {"phrase": "text with { braces } inside"},
        {"phrase": "and ] too"},
    ]


def test_empty_text_raises():
    with pytest.raises(RuntimeError, match="empty"):
        extract_json_array_from_text("")


def test_no_array_raises():
    with pytest.raises(RuntimeError, match="missing JSON array"):
        extract_json_array_from_text('{"id": "N0"}')


def test_control_marker_stripped():
    text = '<|function_call|>[{"id": "N0"}]'
    got = extract_json_array_from_text(text)
    assert got == [{"id": "N0"}]
