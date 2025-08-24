"""Configuration management API endpoints."""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.config.log import get_logger
from app.dependencies.service_container import get_service_container
from app.services.config.simple_user_config_manager import get_user_config_manager

router = APIRouter(prefix='/api', tags=['Configuration'])
logger = get_logger(__name__)


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
        reload_result = config_manager.trigger_reload()

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
async def get_configuration_status() -> Dict[str, Any]:
    """Get current configuration status and information.

    Returns:
        Dictionary with configuration details and system status
    """
    try:
        config_manager = get_user_config_manager()
        status = config_manager.get_config_status()

        # Add simple system information
        try:
            service_container = get_service_container()

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
async def validate_configuration() -> Dict[str, Any]:
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
            service_container = get_service_container()

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
