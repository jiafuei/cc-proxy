"""Configuration management API endpoints."""

from typing import Any, Dict

import yaml
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.common.yaml_utils import safe_load_with_env
from app.config.log import get_logger
from app.config.user_models import UserConfig
from app.dependencies import get_service_container_dependency
from app.services.config.simple_user_config_manager import get_user_config_manager

router = APIRouter(prefix='/api', tags=['Configuration'])
logger = get_logger(__name__)


class ConfigValidationRequest(BaseModel):
    """Request model for config validation endpoint."""

    yaml_content: str


@router.post('/reload')
async def reload_configuration() -> Dict[str, Any]:
    """Manually reload user configuration and rebuild services.

    This endpoint allows users to reload their configuration from ~/.cc-proxy/user.yaml
    without restarting the application.

    Returns:
        Dictionary with reload results and status
    """
    try:
        # Get config manager and trigger reload
        config_manager = get_user_config_manager()
        reload_result = await config_manager.trigger_reload()

        if not reload_result['success']:
            raise HTTPException(status_code=400, detail=reload_result)

        logger.info('Configuration reloaded successfully via API')
        return reload_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Error reloading configuration via API: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')


@router.get('/config/status')
async def get_configuration_status(service_container = Depends(get_service_container_dependency)) -> Dict[str, Any]:
    """Get current configuration status and information.

    Returns:
        Dictionary with configuration details and system status
    """
    try:
        config_manager = get_user_config_manager()
        status = config_manager.get_config_status()

        # Add simple system information
        try:
            # Create routing summary from the service container
            routing_summary = {}
            if service_container.provider_manager and service_container.router:
                routing_summary = {
                    'providers': len(service_container.provider_manager.list_providers()),
                    'models': len(service_container.provider_manager.list_models()),
                    'transformers': service_container.transformer_loader.get_cache_info()['cached_transformers'] if service_container.transformer_loader else 0,
                }
            status['routing'] = routing_summary

            # Basic validation - check if core components are initialized
            validation_errors = []
            if not service_container.provider_manager:
                validation_errors.append('Provider manager not initialized')
            if not service_container.router:
                validation_errors.append('Router not initialized')
            if not service_container.transformer_loader:
                validation_errors.append('Transformer loader not initialized')

            status['validation'] = {'valid': len(validation_errors) == 0, 'errors': validation_errors}

        except Exception as e:
            logger.warning(f'Could not get service container status: {e}')
            status['service_container'] = {'error': str(e)}

        return status

    except Exception as e:
        logger.error(f'Error getting configuration status via API: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')


@router.get('/config/validate')
async def validate_configuration(service_container = Depends(get_service_container_dependency)) -> Dict[str, Any]:
    """Validate current configuration without reloading.

    Returns:
        Dictionary with validation results
    """
    try:
        # Check if configuration file is valid
        config_manager = get_user_config_manager()
        current_config = config_manager.get_current_config()

        if current_config is None:
            return {'valid': False, 'errors': ['No configuration loaded']}

        errors = []

        # Validate configuration references
        try:
            current_config.validate_references()
        except ValueError as e:
            errors.append(f'Reference validation failed: {str(e)}')

        # Get simple system validation if available
        try:
            # Basic validation of service container state
            if not service_container.provider_manager:
                errors.append('System validation failed: Provider manager not initialized')
            if not service_container.router:
                errors.append('System validation failed: Router not initialized')
            if not service_container.transformer_loader:
                errors.append('System validation failed: Transformer loader not initialized')

        except Exception as e:
            errors.append(f'System validation failed: {str(e)}')

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'config_summary': {
                'providers': len(current_config.providers),
                'models': len(current_config.models),
                'routing_configured': current_config.routing is not None,
                'transformer_paths': len(current_config.transformer_paths) if hasattr(current_config, 'transformer_paths') else 0,
            },
        }

    except Exception as e:
        logger.error(f'Error validating configuration via API: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Internal error: {str(e)}')


@router.post('/config/validate-yaml')
async def validate_yaml_content(request: ConfigValidationRequest) -> Dict[str, Any]:
    """Validate YAML configuration content without loading it into the system.

    This endpoint allows users to test their configuration YAML before applying it.

    Args:
        request: Request containing yaml_content to validate

    Returns:
        Dictionary with validation results and detailed error information
    """
    try:
        errors = []
        warnings = []

        # 1. Parse YAML syntax
        try:
            yaml_data = safe_load_with_env(request.yaml_content)
            if yaml_data is None:
                yaml_data = {}
        except yaml.YAMLError as e:
            return {'valid': False, 'errors': [f'Invalid YAML syntax: {str(e)}'], 'warnings': [], 'stage': 'yaml_parsing'}

        # 2. Validate against UserConfig model
        try:
            config = UserConfig(**yaml_data)
            logger.debug('YAML content successfully parsed into UserConfig model')
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Schema validation failed: {str(e)}'],
                'warnings': warnings,
                'stage': 'schema_validation',
                'yaml_structure': _get_yaml_structure_info(yaml_data),
            }

        # 3. Validate references between components
        try:
            config.validate_references()
            logger.debug('Reference validation passed')
        except ValueError as e:
            errors.append(f'Reference validation failed: {str(e)}')

        # 4. Additional validation checks
        if not config.providers:
            warnings.append('No providers configured - system will not be able to process requests')

        if not config.models:
            warnings.append('No models configured - routing will not work')

        if config.routing is None:
            warnings.append('No routing configuration - will use default routing')

        # Check for potential transformer loading issues
        total_request_transformers = 0
        total_response_transformers = 0
        total_stream_transformers = 0

        for provider in config.providers:
            for channel, stage_config in provider.transformers.items():
                for transformer in stage_config.request:
                    total_request_transformers += 1
                    if not transformer.class_path:
                        errors.append(f"Provider '{provider.name}' channel '{channel}' request transformer missing class path")
                    if not isinstance(transformer.params, dict):
                        errors.append(f"Provider '{provider.name}' channel '{channel}' request transformer params must be a dictionary")

                for transformer in stage_config.response:
                    total_response_transformers += 1
                    if not transformer.class_path:
                        errors.append(f"Provider '{provider.name}' channel '{channel}' response transformer missing class path")
                    if not isinstance(transformer.params, dict):
                        errors.append(f"Provider '{provider.name}' channel '{channel}' response transformer params must be a dictionary")

                for transformer in stage_config.stream:
                    total_stream_transformers += 1
                    if not transformer.class_path:
                        errors.append(f"Provider '{provider.name}' channel '{channel}' stream transformer missing class path")
                    if not isinstance(transformer.params, dict):
                        errors.append(f"Provider '{provider.name}' channel '{channel}' stream transformer params must be a dictionary")

        # Generate summary
        config_summary = {
            'providers': len(config.providers),
            'models': len(config.models),
            'routing_configured': config.routing is not None,
            'transformer_paths': len(config.transformer_paths),
            'total_request_transformers': total_request_transformers,
            'total_response_transformers': total_response_transformers,
            'total_stream_transformers': total_stream_transformers,
        }

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config_summary': config_summary,
            'stage': 'complete',
            'message': 'Configuration is valid and ready to use' if len(errors) == 0 else f'Configuration has {len(errors)} error(s)',
        }

    except Exception as e:
        logger.error(f'Unexpected error validating YAML content: {e}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'Internal validation error: {str(e)}')


def _get_yaml_structure_info(yaml_data: Dict) -> Dict[str, Any]:
    """Get information about the YAML structure for debugging."""
    return {
        'root_keys': list(yaml_data.keys()) if isinstance(yaml_data, dict) else ['not_a_dict'],
        'has_providers': 'providers' in yaml_data if isinstance(yaml_data, dict) else False,
        'has_models': 'models' in yaml_data if isinstance(yaml_data, dict) else False,
        'has_routing': 'routing' in yaml_data if isinstance(yaml_data, dict) else False,
        'has_transformer_paths': 'transformer_paths' in yaml_data if isinstance(yaml_data, dict) else False,
        'provider_count': len(yaml_data.get('providers', [])) if isinstance(yaml_data, dict) else 0,
    }
