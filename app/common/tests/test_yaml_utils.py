"""Tests for YAML utilities with environment variable support."""

import os
from unittest.mock import patch

import pytest
import yaml

from app.common.yaml_utils import safe_load_with_env


class TestEnvVarLoader:
    """Test the custom YAML loader with !env tag support."""

    def test_required_env_var_present(self):
        """Test loading a required environment variable that exists."""
        yaml_content = """
        api_key: !env TEST_API_KEY
        """

        with patch.dict(os.environ, {'TEST_API_KEY': 'secret-key-123'}):
            result = safe_load_with_env(yaml_content)
            assert result['api_key'] == 'secret-key-123'

    def test_required_env_var_missing(self):
        """Test loading a required environment variable that doesn't exist."""
        yaml_content = """
        api_key: !env MISSING_API_KEY
        """

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Required environment variable 'MISSING_API_KEY' is not set"):
                safe_load_with_env(yaml_content)

    def test_optional_env_var_present(self):
        """Test loading an optional environment variable that exists."""
        yaml_content = """
        port: !env [TEST_PORT, 8000]
        """

        with patch.dict(os.environ, {'TEST_PORT': '9000'}):
            result = safe_load_with_env(yaml_content)
            assert result['port'] == '9000'

    def test_optional_env_var_missing_with_default(self):
        """Test loading an optional environment variable with default value."""
        yaml_content = """
        port: !env [MISSING_PORT, 8000]
        timeout: !env [MISSING_TIMEOUT, null]
        """

        with patch.dict(os.environ, {}, clear=True):
            result = safe_load_with_env(yaml_content)
            assert result['port'] == 8000
            assert result['timeout'] is None

    def test_env_var_types(self):
        """Test that environment variables maintain their YAML types when using defaults."""
        yaml_content = """
        string_val: !env [TEST_STRING, "default_string"]
        int_val: !env [TEST_INT, 42]
        bool_val: !env [TEST_BOOL, true]
        float_val: !env [TEST_FLOAT, 3.14]
        null_val: !env [TEST_NULL, null]
        """

        with patch.dict(os.environ, {}, clear=True):
            result = safe_load_with_env(yaml_content)
            assert result['string_val'] == 'default_string'
            assert result['int_val'] == 42
            assert result['bool_val'] is True
            assert result['float_val'] == 3.14
            assert result['null_val'] is None

    def test_env_var_overrides_default_types(self):
        """Test that environment variable values are always strings, regardless of default type."""
        yaml_content = """
        port: !env [TEST_PORT, 8000]
        enabled: !env [TEST_ENABLED, false]
        """

        with patch.dict(os.environ, {'TEST_PORT': '9000', 'TEST_ENABLED': 'true'}):
            result = safe_load_with_env(yaml_content)
            # Environment variables are always strings
            assert result['port'] == '9000'  # String, not int
            assert result['enabled'] == 'true'  # String, not bool

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
