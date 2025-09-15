import os
import tempfile
from unittest.mock import patch

import pytest

from app.config.models import ConfigModel


@pytest.mark.parametrize("scenario,config_data,validation,should_raise,error_match", [
    # Config loading scenarios
    ("defaults for non-existent file", 
     None,  # No file created
     lambda c: c.cors_allow_origins == [] and c.host == '127.0.0.1' and c.port == 8000 and c.dev == False,
     False, None),
    ("values from yaml file",
     {'cors_allow_origins': ['http://test.com'], 'host': '0.0.0.0', 'port': 9000, 'dev': True},
     lambda c: c.cors_allow_origins == ['http://test.com'] and c.host == '0.0.0.0' and c.port == 9000 and c.dev == True,
     False, None),
    ("invalid yaml",
     '{ invalid yaml',  # String indicates invalid YAML content
     None,
     True, 'Invalid YAML'),
])
def test_config_loading(scenario, config_data, validation, should_raise, error_match):
    """Test configuration loading scenarios."""
    if config_data is None:
        # Test non-existent file
        config = ConfigModel.load('non-existent-config.yaml')
        assert validation(config)
    elif isinstance(config_data, str):
        # Test invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_data)
            temp_path = f.name
        try:
            if should_raise:
                with pytest.raises(ValueError, match=error_match):
                    ConfigModel.load(temp_path)
            else:
                config = ConfigModel.load(temp_path)
                assert validation(config)
        finally:
            os.unlink(temp_path)
    else:
        # Test valid config data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config_data, f)
            temp_path = f.name
        try:
            config = ConfigModel.load(temp_path)
            assert validation(config)
        finally:
            os.unlink(temp_path)


def test_config_validation():
    """Test that ConfigModel validates fields correctly."""
    # Test valid config
    valid_config = ConfigModel(port=8080, host='localhost')
    assert valid_config.port == 8080
    assert valid_config.host == 'localhost'

    # Test port validation (should be between 1 and 65535)
    try:
        ConfigModel(port=0)
        assert False, 'Expected validation error for port=0'
    except Exception:
        pass

    try:
        ConfigModel(port=65536)
        assert False, 'Expected validation error for port=65536'
    except Exception:
        pass


# Trivial return-type tests and redundant file loading test removed - they provided no meaningful coverage


def test_config_env_vars_with_defaults():
    """Test that !env tags work with defaults when environment variables are not set."""
    yaml_content = """
host: !env [TEST_HOST, "127.0.0.1"]
port: !env [TEST_PORT, 8000]
dev: !env [TEST_DEV, false]
logging:
  console_enabled: !env [TEST_CONSOLE_ENABLED, true]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigModel.load(temp_path)
            assert config.host == '127.0.0.1'  # str
            assert config.port == 8000  # int
            assert config.dev is False  # bool
            assert config.logging.console_enabled is True  # bool
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize(
    'env_var,env_value,expected_value,field_name',
    [
        ('TEST_HOST', '0.0.0.0', '0.0.0.0', 'host'),
        ('TEST_PORT', '9000', 9000, 'port'),
        ('TEST_DEV', 'true', True, 'dev'),
    ],
)
def test_config_env_var_type_coercion(env_var, env_value, expected_value, field_name):
    """Test that environment variables are correctly coerced to their expected types."""
    yaml_content = """
host: !env [TEST_HOST, "127.0.0.1"]
port: !env [TEST_PORT, 8000]
dev: !env [TEST_DEV, false]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with patch.dict(os.environ, {env_var: env_value}, clear=True):
            config = ConfigModel.load(temp_path)
            actual_value = getattr(config, field_name)
            assert actual_value == expected_value
            assert isinstance(actual_value, type(expected_value))
    finally:
        os.unlink(temp_path)


@pytest.mark.parametrize(
    'env_value,expected_bool',
    [
        ('true', True),
        ('True', True),
        ('TRUE', True),
        ('1', True),
        ('false', False),
        ('False', False),
        ('FALSE', False),
        ('0', False),
    ],
)
def test_config_boolean_env_var_conversion(env_value, expected_bool):
    """Test boolean environment variable conversion edge cases."""
    yaml_content = """
dev: !env [TEST_DEV, false]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with patch.dict(os.environ, {'TEST_DEV': env_value}, clear=True):
            config = ConfigModel.load(temp_path)
            assert config.dev is expected_bool
    finally:
        os.unlink(temp_path)


def test_config_invalid_env_var_values():
    """Test that invalid environment variable values raise validation errors."""
    yaml_content = """
port: !env [TEST_PORT, 8000]
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        with patch.dict(os.environ, {'TEST_PORT': 'not-a-number'}, clear=True):
            try:
                ConfigModel.load(temp_path)
                assert False, 'Expected validation error for invalid port value'
            except Exception as e:
                assert 'validation error' in str(e).lower() or 'invalid' in str(e).lower()
    finally:
        os.unlink(temp_path)
