"""Dynamic transformer loader for user-defined transformers."""

import importlib
import sys
from typing import Any, Dict, List, Optional

from app.config.log import get_logger

logger = get_logger(__name__)


class TransformerLoader:
    """Loads transformers from external modules and built-in classes."""

    def __init__(self, transformer_paths: Optional[List[str]] = None):
        """Initialize transformer loader with optional search paths.

        Args:
            transformer_paths: List of paths to add to Python path for transformer modules
        """
        self._cache: Dict[str, Any] = {}
        self._setup_paths(transformer_paths or [])

    def _setup_paths(self, transformer_paths: List[str]):
        """Add transformer paths to Python path."""
        for path in transformer_paths:
            if path not in sys.path:
                sys.path.insert(0, path)
                logger.debug(f'Added transformer path: {path}')

    def load_transformer(self, transformer_config: Dict[str, Any]) -> Any:
        """Load transformer from configuration.

        Args:
            transformer_config: Dictionary with 'class' key and optional 'params'

        Returns:
            Instantiated transformer object

        Example config:
            {
                'class': 'my_transformers.CustomAuthTransformer',
                'params': {'api_key': 'sk-...'}
            }
        """
        class_path = transformer_config['class']
        params = transformer_config.get('params', {})

        # Use cached instance if available
        cache_key = f'{class_path}:{hash(frozenset(params.items()))}'
        if cache_key in self._cache:
            logger.debug(f'Using cached transformer: {class_path}')
            return self._cache[cache_key]

        try:
            # Dynamic import
            module_name, class_name = class_path.rsplit('.', 1)
            logger.debug(f'Loading transformer: {class_path}')

            module = importlib.import_module(module_name)
            transformer_class = getattr(module, class_name)

            # Instantiate with params
            instance = transformer_class(**params)

            # Cache the instance
            self._cache[cache_key] = instance
            logger.info(f'Loaded transformer: {class_path}')

            return instance

        except Exception as e:
            logger.error(f"Failed to load transformer '{class_path}': {e}", exc_info=True)
            raise RuntimeError(f"Cannot load transformer '{class_path}': {e}") from e

    def load_transformers(self, transformer_configs: List[Dict[str, Any]]) -> List[Any]:
        """Load multiple transformers from configuration.

        Args:
            transformer_configs: List of transformer configuration dictionaries

        Returns:
            List of instantiated transformer objects
        """
        transformers = []
        for config in transformer_configs:
            try:
                transformer = self.load_transformer(config)
                transformers.append(transformer)
            except Exception as e:
                logger.error(f'Skipping failed transformer: {e}')
                # Continue loading other transformers even if one fails

        return transformers

    def clear_cache(self):
        """Clear the transformer cache."""
        self._cache.clear()
        logger.debug('Transformer cache cleared')

    def get_cache_info(self) -> Dict[str, int]:
        """Get information about the transformer cache."""
        return {'cached_transformers': len(self._cache), 'cache_keys': list(self._cache.keys())}
