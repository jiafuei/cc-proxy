"""Service provider for managing service generations with hot-swapping."""

import logging
import threading
import time
import uuid
from typing import Dict, Optional, Tuple

from app.config.models import ConfigModel
from app.config.user_models import UserConfig
from app.services.config.interfaces import ServiceBuilder, ServiceGeneration, ServiceProvider

logger = logging.getLogger(__name__)


class DynamicServiceProvider(ServiceProvider):
    """Service provider that manages service generations for hot-swapping."""

    def __init__(self, app_config: ConfigModel, service_builder: ServiceBuilder):
        """Initialize the service provider.

        Args:
            app_config: Static application configuration
            service_builder: Builder for creating services from configuration
        """
        self.app_config = app_config
        self.service_builder = service_builder
        self._lock = threading.RLock()

        # Service generation management
        self._current_generation: Optional[ServiceGeneration] = None
        self._old_generations: Dict[str, ServiceGeneration] = {}
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown_requested = False

        logger.info('Initialized dynamic service provider')

    def get_current_services(self) -> Tuple[str, any]:
        """Get current services with generation ID.

        Returns:
            Tuple of (generation_id, services)
        """
        with self._lock:
            if self._current_generation is None:
                raise RuntimeError('No services available - call rebuild_services first')

            return self._current_generation.generation_id, self._current_generation.services

    def acquire_services(self, generation_id: str) -> Optional[any]:
        """Acquire services for a specific generation.

        Args:
            generation_id: ID of the service generation to acquire

        Returns:
            Services instance or None if generation not found
        """
        with self._lock:
            # Check current generation
            if self._current_generation and self._current_generation.generation_id == generation_id:
                self._current_generation.acquire()
                logger.debug(f'Acquired current generation {generation_id} (ref_count: {self._current_generation.ref_count})')
                return self._current_generation.services

            # Check old generations
            if generation_id in self._old_generations:
                generation = self._old_generations[generation_id]
                if not generation.shutdown_requested:
                    generation.acquire()
                    logger.debug(f'Acquired old generation {generation_id} (ref_count: {generation.ref_count})')
                    return generation.services

            logger.warning(f'Services generation {generation_id} not found or shut down')
            return None

    def release_services(self, generation_id: str) -> None:
        """Release services for a specific generation.

        Args:
            generation_id: ID of the service generation to release
        """
        with self._lock:
            # Check current generation
            if self._current_generation and self._current_generation.generation_id == generation_id:
                self._current_generation.release()
                logger.debug(f'Released current generation {generation_id} (ref_count: {self._current_generation.ref_count})')
                return

            # Check old generations
            if generation_id in self._old_generations:
                generation = self._old_generations[generation_id]
                generation.release()
                logger.debug(f'Released old generation {generation_id} (ref_count: {generation.ref_count})')

                # Check if we can clean up this generation
                if generation.can_shutdown():
                    self._cleanup_generation(generation_id)

                return

            logger.debug(f'Services generation {generation_id} not found for release')

    def rebuild_services(self, config: UserConfig) -> str:
        """Rebuild services from new configuration.

        Args:
            config: New user configuration

        Returns:
            New generation ID
        """
        logger.info('Rebuilding services from new configuration')

        try:
            # Build new services
            new_services = self.service_builder.build_services(config)
            new_generation_id = self._generate_id()

            with self._lock:
                # Create new generation
                new_generation = ServiceGeneration(new_generation_id, new_services)

                # Move current generation to old generations if it exists
                if self._current_generation is not None:
                    old_generation = self._current_generation
                    old_generation.shutdown_requested = True
                    self._old_generations[old_generation.generation_id] = old_generation
                    logger.info(f'Moved generation {old_generation.generation_id} to old generations (ref_count: {old_generation.ref_count})')

                # Set new current generation
                self._current_generation = new_generation

                # Start cleanup thread if not already running
                if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
                    self._start_cleanup_thread()

                logger.info(f'Services rebuilt successfully with generation ID: {new_generation_id}')
                return new_generation_id

        except Exception as e:
            logger.error(f'Failed to rebuild services: {e}', exc_info=True)
            raise

    def shutdown(self) -> None:
        """Shutdown the service provider and clean up all generations."""
        logger.info('Shutting down service provider')

        with self._lock:
            self._shutdown_requested = True

            # Mark all generations for shutdown
            if self._current_generation:
                self._current_generation.shutdown_requested = True

            for generation in self._old_generations.values():
                generation.shutdown_requested = True

        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=10.0)

        logger.info('Service provider shutdown complete')

    def get_generation_stats(self) -> Dict[str, any]:
        """Get statistics about current generations.

        Returns:
            Dictionary with generation statistics
        """
        with self._lock:
            stats = {'current_generation': None, 'old_generations': [], 'total_generations': len(self._old_generations)}

            if self._current_generation:
                stats['current_generation'] = {
                    'id': self._current_generation.generation_id,
                    'ref_count': self._current_generation.ref_count,
                    'shutdown_requested': self._current_generation.shutdown_requested,
                }
                stats['total_generations'] += 1

            for gen_id, generation in self._old_generations.items():
                stats['old_generations'].append({'id': gen_id, 'ref_count': generation.ref_count, 'shutdown_requested': generation.shutdown_requested})

            return stats

    def _generate_id(self) -> str:
        """Generate a unique generation ID."""
        return f'gen_{uuid.uuid4().hex[:8]}_{int(time.time())}'

    def _start_cleanup_thread(self) -> None:
        """Start the cleanup thread for old generations."""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.debug('Started cleanup thread')

    def _cleanup_loop(self) -> None:
        """Main loop for cleaning up old generations."""
        logger.debug('Cleanup thread started')

        while not self._shutdown_requested:
            try:
                self._cleanup_old_generations()
                time.sleep(5.0)  # Check every 5 seconds

            except Exception as e:
                logger.error(f'Error in cleanup loop: {e}', exc_info=True)

        # Final cleanup on shutdown
        self._cleanup_old_generations()
        logger.debug('Cleanup thread finished')

    def _cleanup_old_generations(self) -> None:
        """Clean up old generations that can be shut down."""
        with self._lock:
            generations_to_cleanup = []

            for gen_id, generation in self._old_generations.items():
                if generation.can_shutdown():
                    generations_to_cleanup.append(gen_id)

            for gen_id in generations_to_cleanup:
                self._cleanup_generation(gen_id)

    def _cleanup_generation(self, generation_id: str) -> None:
        """Clean up a specific generation.

        Args:
            generation_id: ID of generation to clean up
        """
        generation = self._old_generations.pop(generation_id, None)
        if generation:
            try:
                # Perform any necessary cleanup of services
                if hasattr(generation.services, 'cleanup'):
                    generation.services.cleanup()

                logger.info(f'Cleaned up generation {generation_id}')

            except Exception as e:
                logger.error(f'Error cleaning up generation {generation_id}: {e}', exc_info=True)
