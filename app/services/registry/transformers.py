"""Dynamic transformer registry with importlib-based loading."""

import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from app.config.log import get_logger
from app.config.user_models import TransformerArgument, TransformerConfig
from app.services.config.interfaces import ComponentRegistry
from app.services.pipeline.interfaces import RequestTransformer, ResponseTransformer, StreamTransformer

logger = get_logger(__name__)


class TransformerFactory:
    """Factory for creating transformer instances with configured arguments."""

    def __init__(self, transformer_classes: Dict[str, Type], default_args: List[TransformerArgument]):
        """Initialize factory with transformer classes and default arguments.

        Args:
            transformer_classes: Dictionary mapping transformer type to class
            default_args: Default arguments to pass to transformers
        """
        self.request_transformer_class = transformer_classes.get('request')
        self.response_transformer_class = transformer_classes.get('response')
        self.stream_transformer_class = transformer_classes.get('stream')
        self.default_args = {arg.key: arg.value for arg in default_args}

    def create_request_transformer(self, override_args: Optional[List[TransformerArgument]] = None) -> Optional[RequestTransformer]:
        """Create a request transformer instance."""
        if not self.request_transformer_class:
            return None

        args = self._merge_args(override_args)
        return self._instantiate_transformer(self.request_transformer_class, args)

    def create_response_transformer(self, override_args: Optional[List[TransformerArgument]] = None) -> Optional[ResponseTransformer]:
        """Create a response transformer instance."""
        if not self.response_transformer_class:
            return None

        args = self._merge_args(override_args)
        return self._instantiate_transformer(self.response_transformer_class, args)

    def create_stream_transformer(self, override_args: Optional[List[TransformerArgument]] = None) -> Optional[StreamTransformer]:
        """Create a stream transformer instance."""
        if not self.stream_transformer_class:
            return None

        args = self._merge_args(override_args)
        return self._instantiate_transformer(self.stream_transformer_class, args)

    def _merge_args(self, override_args: Optional[List[TransformerArgument]]) -> Dict[str, str]:
        """Merge default args with override args."""
        merged = self.default_args.copy()
        if override_args:
            for arg in override_args:
                merged[arg.key] = arg.value
        return merged

    def _instantiate_transformer(self, transformer_class: Type, args: Dict[str, str]) -> Any:
        """Instantiate a transformer with arguments."""
        try:
            # Try to pass args as keyword arguments
            sig = inspect.signature(transformer_class.__init__)
            filtered_args = {}

            for key, value in args.items():
                if key in sig.parameters:
                    # Try to convert to appropriate type based on parameter annotation
                    param = sig.parameters[key]
                    if param.annotation != inspect.Parameter.empty:
                        try:
                            if param.annotation is bool:
                                value = value.lower() in ('true', '1', 'yes', 'on')
                            elif param.annotation is int:
                                value = int(value)
                            elif param.annotation is float:
                                value = float(value)
                            # Keep as string for other types
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Failed to convert arg '{key}' to {param.annotation}: {e}")

                    filtered_args[key] = value

            return transformer_class(**filtered_args)

        except Exception as e:
            logger.error(f'Failed to instantiate transformer {transformer_class.__name__}: {e}')
            # Try without arguments as fallback
            try:
                return transformer_class()
            except Exception as e2:
                logger.error(f'Failed to instantiate transformer without args: {e2}')
                raise


class TransformerRegistry(ComponentRegistry[TransformerFactory]):
    """Registry for managing custom transformers with dynamic loading."""

    def __init__(self):
        super().__init__()
        self._loaded_modules: Dict[str, Any] = {}

    def load_transformer(self, config: TransformerConfig) -> bool:
        """Load a transformer from a Python file.

        Args:
            config: Transformer configuration

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            path = Path(config.path)
            if not path.exists():
                logger.error(f'Transformer file does not exist: {path}')
                return False

            # Load module from file
            spec = importlib.util.spec_from_file_location(config.name, path)
            if spec is None or spec.loader is None:
                logger.error(f'Could not create module spec for {path}')
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find transformer classes in the module
            transformer_classes = self._find_transformer_classes(module)

            if not transformer_classes:
                logger.error(f'No transformer classes found in {path}')
                return False

            # Create factory
            factory = TransformerFactory(transformer_classes, config.args)

            # Register the factory
            self.register(config.name, factory)
            self._loaded_modules[config.name] = module

            logger.info(f"Successfully loaded transformer '{config.name}' from {path}")
            logger.debug(f'Found transformer classes: {list(transformer_classes.keys())}')

            return True

        except Exception as e:
            logger.error(f"Failed to load transformer '{config.name}' from {config.path}: {e}", exc_info=True)
            return False

    def unload_transformer(self, name: str) -> None:
        """Unload a transformer and clean up its module."""
        self.unregister(name)
        self._loaded_modules.pop(name, None)
        logger.info(f"Unloaded transformer '{name}'")

    def reload_transformer(self, config: TransformerConfig) -> bool:
        """Reload a transformer (unload then load)."""
        logger.info(f"Reloading transformer '{config.name}'")
        self.unload_transformer(config.name)
        return self.load_transformer(config)

    def load_transformers_from_config(self, transformers: List[TransformerConfig]) -> int:
        """Load multiple transformers from configuration.

        Args:
            transformers: List of transformer configurations

        Returns:
            Number of transformers loaded successfully
        """
        success_count = 0

        for transformer_config in transformers:
            if self.load_transformer(transformer_config):
                success_count += 1

        logger.info(f'Loaded {success_count}/{len(transformers)} transformers successfully')
        return success_count

    def clear_all(self) -> None:
        """Clear all transformers and modules."""
        self._loaded_modules.clear()
        self.clear()
        logger.info('Cleared all transformers')

    def _find_transformer_classes(self, module: Any) -> Dict[str, Type]:
        """Find transformer classes in a loaded module.

        Args:
            module: The loaded Python module

        Returns:
            Dictionary mapping transformer type ('request', 'response', 'stream') to class
        """
        transformer_classes = {}

        # Look for classes that inherit from transformer interfaces
        for name in dir(module):
            obj = getattr(module, name)

            if not inspect.isclass(obj):
                continue

            # Skip imported classes (they should be defined in this module)
            if obj.__module__ != module.__name__:
                continue

            # Check if it implements transformer interfaces
            if issubclass(obj, RequestTransformer) and obj != RequestTransformer:
                transformer_classes['request'] = obj
                logger.debug(f'Found RequestTransformer: {name}')

            if issubclass(obj, ResponseTransformer) and obj != ResponseTransformer:
                transformer_classes['response'] = obj
                logger.debug(f'Found ResponseTransformer: {name}')

            if issubclass(obj, StreamTransformer) and obj != StreamTransformer:
                transformer_classes['stream'] = obj
                logger.debug(f'Found StreamTransformer: {name}')

        return transformer_classes
