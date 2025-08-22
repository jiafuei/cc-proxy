import os
import tempfile

from app.config import Config


def test_config_defaults():
    """Test that Config loads with default values when no file is present."""
    # Create a config with a non-existent file path
    config = Config.load('non-existent-config.yaml')

    # Check default values
    assert config.cors_allow_origins == []
    assert config.host == '127.0.0.1'
    assert config.port == 8000
    assert config.dev == False


def test_config_from_yaml():
    """Test that Config loads values from a YAML file."""
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
        config = Config.load(temp_path)

        # Check that values match what's in the YAML file
        assert config.cors_allow_origins == config_data['cors_allow_origins']
        assert config.host == config_data['host']
        assert config.port == config_data['port']
        assert config.dev == config_data['dev']
    finally:
        # Clean up the temporary file
        os.unlink(temp_path)
