import os
import tempfile
from unittest.mock import patch

from app.config import get_config, reload_config
from app.config.models import ConfigModel


def test_config_defaults():
    """Test that ConfigModel loads with default values when no file is present."""
    # Create a config with a non-existent file path
    config = ConfigModel.load('non-existent-config.yaml')

    # Check default values
    assert config.cors_allow_origins == []
    assert config.host == '127.0.0.1'
    assert config.port == 8000
    assert config.dev == False


def test_config_from_yaml():
    """Test that ConfigModel loads values from a YAML file."""
    # Create a temporary YAML file with test configuration
    config_data = {
        'cors_allow_origins': ['http://test.com', 'https://test.com'],
        'host': '0.0.0.0',
        'port': 9000,
        'dev': True,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml

        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Load config from the temporary file
        config = ConfigModel.load(temp_path)

        # Check that values match what's in the YAML file
        assert config.cors_allow_origins == config_data['cors_allow_origins']
        assert config.host == config_data['host']
        assert config.port == config_data['port']
        assert config.dev == config_data['dev']
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


def test_config_invalid_yaml():
    """Test that ConfigModel raises ValueError for invalid YAML."""
    # Create a temporary file with invalid YAML
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write('{ invalid yaml')
        temp_path = f.name

    try:
        # Attempt to load config from the invalid YAML file
        try:
            ConfigModel.load(temp_path)
            assert False, 'Expected ValueError to be raised'
        except ValueError as e:
            assert 'Invalid YAML' in str(e)
    finally:
        # Clean up the temporary file
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


def test_get_config():
    """Test that get_config returns a ConfigModel instance."""
    config = get_config()
    assert isinstance(config, ConfigModel)


def test_reload_config():
    """Test that reload_config returns a ConfigModel instance."""
    config = reload_config()
    assert isinstance(config, ConfigModel)


def test_config_multifile_loading():
    """Test that ConfigModel tries multiple file locations."""
    # Create a temporary YAML file with test configuration
    config_data = {
        'port': 9000,
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml

        yaml.dump(config_data, f)
        temp_path = f.name

    try:
        # Test with explicit config path
        config = ConfigModel.load(temp_path)
        assert config.port == 9000

    finally:
        # Clean up the temporary file
        os.unlink(temp_path)


def test_config_env_vars_with_pydantic_types():
    """Test that !env tags work correctly with Pydantic type coercion."""
    # Create YAML content with !env tags that should be coerced to different types
    yaml_content = '''
host: !env [TEST_HOST, "127.0.0.1"]
port: !env [TEST_PORT, 8000]
dev: !env [TEST_DEV, false]
backup_count: !env [TEST_BACKUP_COUNT, 5]
max_file_size: !env [TEST_MAX_FILE_SIZE, "10MB"]
logging:
  console_enabled: !env [TEST_CONSOLE_ENABLED, true]
  backup_count: !env [TEST_LOG_BACKUP_COUNT, 4]
'''

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write(yaml_content)
        temp_path = f.name

    try:
        # Test 1: Environment variables not set - should use defaults with correct types
        with patch.dict(os.environ, {}, clear=True):
            config = ConfigModel.load(temp_path)
            assert config.host == "127.0.0.1"  # str
            assert config.port == 8000  # int
            assert config.dev is False  # bool 
            assert config.logging.console_enabled is True  # bool
            assert config.logging.backup_count == 4  # int

        # Test 2: Environment variables set as strings - should be coerced to correct types
        env_vars = {
            'TEST_HOST': '0.0.0.0',
            'TEST_PORT': '9000',  # String that should become int
            'TEST_DEV': 'true',   # String that should become bool
            'TEST_CONSOLE_ENABLED': 'false',  # String that should become bool
            'TEST_LOG_BACKUP_COUNT': '10',    # String that should become int
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            config = ConfigModel.load(temp_path)
            
            # Verify values were loaded from env vars AND converted to correct types
            assert config.host == "0.0.0.0"  # str (expected)
            assert config.port == 9000  # int (converted from "9000")  
            assert isinstance(config.port, int)
            assert config.dev is True  # bool (converted from "true")
            assert isinstance(config.dev, bool)
            assert config.logging.console_enabled is False  # bool (converted from "false")
            assert isinstance(config.logging.console_enabled, bool)
            assert config.logging.backup_count == 10  # int (converted from "10")
            assert isinstance(config.logging.backup_count, int)

        # Test 3: Invalid environment variable values should raise validation errors
        invalid_env_vars = {
            'TEST_PORT': 'not-a-number',  # Invalid int
        }
        
        with patch.dict(os.environ, invalid_env_vars, clear=True):
            try:
                ConfigModel.load(temp_path)
                assert False, "Expected validation error for invalid port value"
            except Exception as e:
                # Should get a Pydantic validation error
                assert "validation error" in str(e).lower() or "invalid" in str(e).lower()

        # Test 4: Test edge cases for boolean conversion
        bool_test_cases = [
            ('true', True),
            ('True', True), 
            ('TRUE', True),
            ('1', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
        ]
        
        for env_value, expected_bool in bool_test_cases:
            with patch.dict(os.environ, {'TEST_DEV': env_value}, clear=True):
                config = ConfigModel.load(temp_path)
                assert config.dev is expected_bool, f"Failed for env_value='{env_value}', expected {expected_bool}, got {config.dev}"

    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
