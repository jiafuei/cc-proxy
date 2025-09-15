"""Tests for YAML utilities with environment variable support."""

import os
from unittest.mock import patch

import pytest
import yaml

from app.common.yaml_utils import safe_load_with_env


class TestEnvVarLoader:
    """Test the custom YAML loader with !env tag support."""

    @pytest.mark.parametrize("scenario,yaml_content,env_vars,expected_result,should_raise,error_match", [
        # Basic environment variable scenarios
        ("required env var present",
         "api_key: !env TEST_API_KEY",
         {'TEST_API_KEY': 'secret-key-123'},
         {'api_key': 'secret-key-123'},
         False, None),
        ("required env var missing",
         "api_key: !env MISSING_API_KEY",
         {},
         None,
         True, "Required environment variable 'MISSING_API_KEY' is not set"),
        ("optional env var present",
         "port: !env [TEST_PORT, 8000]",
         {'TEST_PORT': '9000'},
         {'port': '9000'},
         False, None),
        ("optional env var missing with defaults",
         "port: !env [MISSING_PORT, 8000]\ntimeout: !env [MISSING_TIMEOUT, null]",
         {},
         {'port': 8000, 'timeout': None},
         False, None),
    ])
    def test_env_var_scenarios(self, scenario, yaml_content, env_vars, expected_result, should_raise, error_match):
        """Test environment variable loading scenarios."""
        with patch.dict(os.environ, env_vars, clear=True):
            if should_raise:
                with pytest.raises(ValueError, match=error_match):
                    safe_load_with_env(yaml_content)
            else:
                result = safe_load_with_env(yaml_content)
                assert result == expected_result

    @pytest.mark.parametrize("test_type,yaml_content,env_vars,validation", [
        # Type preservation when using defaults
        ("default types preserved",
         "string_val: !env [TEST_STRING, 'default_string']\nint_val: !env [TEST_INT, 42]\nbool_val: !env [TEST_BOOL, true]\nfloat_val: !env [TEST_FLOAT, 3.14]\nnull_val: !env [TEST_NULL, null]",
         {},
         lambda r: r['string_val'] == 'default_string' and r['int_val'] == 42 and r['bool_val'] is True and r['float_val'] == 3.14 and r['null_val'] is None),
        # Environment variables override as strings
        ("env overrides as strings",
         "port: !env [TEST_PORT, 8000]\nenabled: !env [TEST_ENABLED, false]",
         {'TEST_PORT': '9000', 'TEST_ENABLED': 'true'},
         lambda r: r['port'] == '9000' and r['enabled'] == 'true'),
    ])
    def test_type_handling(self, test_type, yaml_content, env_vars, validation):
        """Test type handling for environment variables and defaults."""
        with patch.dict(os.environ, env_vars, clear=True):
            result = safe_load_with_env(yaml_content)
            assert validation(result)

    @pytest.mark.parametrize('invalid_sequence,description', [('[SINGLE_ITEM]', 'too few items'), ('[VAR_NAME, default, extra_item]', 'too many items')])
    def test_invalid_sequence_lengths(self, invalid_sequence, description):
        """Test error handling for invalid sequence lengths in optional env vars."""
        yaml_content = f"""
        api_key: !env {invalid_sequence}
        """

        with pytest.raises(yaml.constructor.ConstructorError, match='must have exactly 2 elements'):
            safe_load_with_env(yaml_content)

    def test_invalid_node_type(self):
        """Test error handling for invalid node types."""
        yaml_content = """
        api_key: !env {key: value}
        """

        with pytest.raises(yaml.constructor.ConstructorError, match='expects scalar.*or sequence'):
            safe_load_with_env(yaml_content)

    def test_complex_yaml_with_env_vars(self):
        """Test a complex YAML structure with multiple environment variables."""
        yaml_content = """
        providers:
          - name: anthropic
            api_key: !env ANTHROPIC_API_KEY
            url: !env [ANTHROPIC_URL, "https://api.anthropic.com/v1/messages"]
            timeout: !env [ANTHROPIC_TIMEOUT, 300]
            settings:
              debug: !env [DEBUG_MODE, false]
        
        routing:
          default: !env [DEFAULT_MODEL, "claude-3-5-sonnet"]
        """

        env_vars = {'ANTHROPIC_API_KEY': 'sk-ant-123', 'DEBUG_MODE': 'true'}

        with patch.dict(os.environ, env_vars, clear=True):
            result = safe_load_with_env(yaml_content)

            assert result['providers'][0]['api_key'] == 'sk-ant-123'
            assert result['providers'][0]['url'] == 'https://api.anthropic.com/v1/messages'  # Default
            assert result['providers'][0]['timeout'] == 300  # Default
            assert result['providers'][0]['settings']['debug'] == 'true'  # From env
            assert result['routing']['default'] == 'claude-3-5-sonnet'  # Default
