"""Tests for threadcount.fit.process_settings and process_settings_dict (Phase 1.2)."""

import json
from types import SimpleNamespace

import pytest

from threadcount.fit import process_settings, process_settings_dict

# ---------------------------------------------------------------------------
# Shared minimal defaults used across all tests
# ---------------------------------------------------------------------------
DEFAULTS = {
    "output_filename": "default_output",
    "snr_lower_limit": 3,
    "parallel": False,
    "n_process": 4,
    "comment": "",
}


# ===========================================================================
# process_settings_dict
# ===========================================================================


class TestProcessSettingsDict:
    """Tests for the dict-based settings combiner."""

    def test_no_user_settings_returns_defaults(self):
        """When user_settings is None the defaults are returned unchanged."""
        result = process_settings_dict(DEFAULTS)
        assert isinstance(result, SimpleNamespace)
        assert result.output_filename == "default_output"
        assert result.snr_lower_limit == 3
        assert result.parallel is False
        assert result.n_process == 4
        assert result.comment == ""

    def test_empty_dict_returns_defaults(self):
        """An empty user dict is treated the same as None."""
        result = process_settings_dict(DEFAULTS, {})
        assert result.snr_lower_limit == 3
        assert result.output_filename == "default_output"

    def test_user_override_replaces_default(self):
        """A key present in user_settings takes the user's value."""
        user = {"output_filename": "my_output", "snr_lower_limit": 5}
        result = process_settings_dict(DEFAULTS, user)
        assert result.output_filename == "my_output"
        assert result.snr_lower_limit == 5

    def test_missing_user_key_uses_default(self):
        """A key absent from user_settings falls back to the default."""
        user = {"output_filename": "only_this"}
        result = process_settings_dict(DEFAULTS, user)
        # Not provided by user — should come from defaults
        assert result.snr_lower_limit == 3
        assert result.parallel is False

    def test_user_only_key_is_accessible(self):
        """Extra keys in user_settings that are not in defaults are kept."""
        user = {"output_filename": "x", "extra_key": 99}
        result = process_settings_dict(DEFAULTS, user)
        assert result.extra_key == 99

    def test_user_override_false_value(self):
        """Falsy individual values inside a non-empty dict are passed through unchanged."""
        user = {
            "output_filename": "",
            "snr_lower_limit": 0,
            "parallel": True,
            "n_process": 1,
            "comment": "hi",
        }
        result = process_settings_dict(DEFAULTS, user)
        # Each value — including falsy ones — must survive; none should be
        # silently replaced by the corresponding default.
        assert result.output_filename == ""
        assert result.snr_lower_limit == 0
        assert result.parallel is True
        assert result.n_process == 1
        assert result.comment == "hi"

    def test_missing_printed_warning(self, capsys):
        """When a default key is missing from user_settings, a warning naming that key is printed."""
        user = {"output_filename": "x"}
        process_settings_dict(DEFAULTS, user)
        captured = capsys.readouterr()
        # Every absent key must appear by name in the output.
        for missing_key in ("snr_lower_limit", "parallel", "n_process", "comment"):
            assert missing_key in captured.out, (
                f"Expected warning for '{missing_key}' not found"
            )

    def test_all_keys_present_no_warning(self, capsys):
        """When all default keys are supplied, no missing-setting warning is printed."""
        process_settings_dict(DEFAULTS, dict(DEFAULTS))
        captured = capsys.readouterr()
        assert "Missing setting" not in captured.out

    def test_returns_simple_namespace(self):
        """Return type is always SimpleNamespace."""
        result = process_settings_dict(DEFAULTS, {"output_filename": "x"})
        assert isinstance(result, SimpleNamespace)

    def test_does_not_mutate_defaults(self):
        """The original defaults dict must not be modified."""
        original = dict(DEFAULTS)
        process_settings_dict(DEFAULTS, {"output_filename": "x"})
        assert DEFAULTS == original

    def test_does_not_mutate_user_settings(self):
        """The original user_settings dict must not be modified."""
        user = {"output_filename": "x"}
        original_user = dict(user)
        process_settings_dict(DEFAULTS, user)
        assert user == original_user


# ===========================================================================
# Invalid-type handling — documents current (silent) behaviour
#
# Phase 5.2 will add real validation so that each of these cases raises a
# clear ValueError instead of silently storing a wrong type.  When that work
# is done, update these tests to assert pytest.raises(ValueError).
# ===========================================================================


class TestInvalidTypesSilentlyAccepted:
    """Document that process_settings_dict currently does NO type validation.

    These tests intentionally assert the *wrong* (current) behaviour so they
    pass today and act as a canary: once Phase 5.2 validation is added, each
    of these tests must be changed from "no error" to "raises ValueError".
    """

    def test_string_for_numeric_setting_is_stored_silently(self):
        """Passing a string for snr_lower_limit (expects a number) raises no error."""
        user = dict(DEFAULTS)
        user["snr_lower_limit"] = "not_a_number"
        # No error is raised — wrong type is silently accepted (known bug).
        result = process_settings_dict(DEFAULTS, user)
        assert result.snr_lower_limit == "not_a_number"

    def test_int_for_bool_setting_is_stored_silently(self):
        """Passing an int for parallel (expects bool) raises no error."""
        user = dict(DEFAULTS)
        user["parallel"] = 99
        result = process_settings_dict(DEFAULTS, user)
        assert result.parallel == 99

    def test_string_for_int_setting_is_stored_silently(self):
        """Passing a string for n_process (expects int) raises no error."""
        user = dict(DEFAULTS)
        user["n_process"] = "four"
        result = process_settings_dict(DEFAULTS, user)
        assert result.n_process == "four"

    def test_none_for_required_string_is_stored_silently(self):
        """Passing None for output_filename (expects str) raises no error."""
        user = dict(DEFAULTS)
        user["output_filename"] = None
        result = process_settings_dict(DEFAULTS, user)
        assert result.output_filename is None

    def test_string_for_numeric_via_json(self):
        """Same silent behaviour via the JSON-string path."""
        user = dict(DEFAULTS)
        user["snr_lower_limit"] = "bad"
        result = process_settings(DEFAULTS, json.dumps(user))
        assert result.snr_lower_limit == "bad"

    def test_none_for_required_string_via_json(self):
        """Passing JSON null for output_filename (expects str) raises no error."""
        user = dict(DEFAULTS)
        user["output_filename"] = None  # serialises to JSON null
        result = process_settings(DEFAULTS, json.dumps(user))
        assert result.output_filename is None

    def test_wrong_container_type_for_user_settings_raises(self):
        """Passing a non-dict (e.g. a list) as user_settings crashes via AttributeError.

        process_settings_dict calls user_settings.keys(), so any non-mapping type
        raises AttributeError.  This is an unguarded crash — document it as a known
        bug.  Phase 5.2 should replace this with a clear TypeError/ValueError.
        """
        with pytest.raises(AttributeError):
            process_settings_dict(DEFAULTS, ["snr_lower_limit", 3])


# ===========================================================================
# process_settings  (JSON-string variant)
# ===========================================================================


class TestProcessSettings:
    """Tests for the JSON-string-based settings combiner."""

    def test_empty_string_returns_defaults(self):
        """An empty string triggers the fast-path that returns defaults directly."""
        result = process_settings(DEFAULTS, "")
        assert isinstance(result, SimpleNamespace)
        assert result.snr_lower_limit == 3
        assert result.output_filename == "default_output"

    def test_no_user_string_returns_defaults(self):
        """Calling with only defaults (default arg) returns a SimpleNamespace of them."""
        result = process_settings(DEFAULTS)
        assert result.parallel is False
        assert result.n_process == 4

    def test_user_override_via_json_string(self):
        """Values in the JSON string override the defaults."""
        user = {"output_filename": "json_output", "snr_lower_limit": 7}
        result = process_settings(DEFAULTS, json.dumps(user))
        assert result.output_filename == "json_output"
        assert result.snr_lower_limit == 7

    def test_missing_key_in_json_falls_back_to_default(self):
        """Keys absent from the JSON string fall back to DEFAULTS."""
        user = {"output_filename": "partial"}
        result = process_settings(DEFAULTS, json.dumps(user))
        assert result.snr_lower_limit == 3
        assert result.parallel is False

    def test_missing_printed_warning_json(self, capsys):
        """A warning naming each absent default key is printed for the JSON path."""
        user = {"output_filename": "x"}
        process_settings(DEFAULTS, json.dumps(user))
        captured = capsys.readouterr()
        for missing_key in ("snr_lower_limit", "parallel", "n_process", "comment"):
            assert missing_key in captured.out, (
                f"Expected warning for '{missing_key}' not found"
            )

    def test_all_keys_present_no_warning_json(self, capsys):
        """No warning when the JSON string supplies every default key."""
        process_settings(DEFAULTS, json.dumps(DEFAULTS))
        captured = capsys.readouterr()
        assert "Missing setting" not in captured.out

    def test_invalid_json_raises(self):
        """Passing malformed JSON raises a json.JSONDecodeError."""
        with pytest.raises(json.JSONDecodeError):
            process_settings(DEFAULTS, "{not valid json}")

    def test_returns_simple_namespace(self):
        """Return type is always SimpleNamespace."""
        result = process_settings(DEFAULTS, json.dumps({"output_filename": "x"}))
        assert isinstance(result, SimpleNamespace)

    def test_does_not_mutate_defaults(self):
        """The original defaults dict must not be modified."""
        original = dict(DEFAULTS)
        process_settings(DEFAULTS, json.dumps({"output_filename": "x"}))
        assert DEFAULTS == original

    def test_none_as_string_arg_raises_type_error(self):
        """process_settings uses == '' to detect 'no settings', so None falls through
        to json.loads(None) and raises TypeError.  This asymmetry with
        process_settings_dict (which handles None via `if not`) is a known inconsistency.
        Phase 5.2 should guard both functions consistently.
        """
        with pytest.raises(TypeError):
            process_settings(DEFAULTS, None)
