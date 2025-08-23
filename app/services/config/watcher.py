"""File watcher implementation using watchdog."""

import logging
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .interfaces import ConfigWatcher

logger = logging.getLogger(__name__)


class _ConfigFileHandler(FileSystemEventHandler):
    """Internal handler for file system events."""

    def __init__(self, watched_file: Path, callback: Callable[[Path], None]):
        self._watched_file = watched_file.resolve()
        self._callback = callback
        super().__init__()

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        event_path = Path(event.src_path).resolve()
        if event_path == self._watched_file:
            logger.info(f'Configuration file changed: {event_path}')
            try:
                self._callback(self._watched_file)
            except Exception as e:
                logger.error(f'Error in config change callback: {e}', exc_info=True)

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move events (covers atomic writes)."""
        if event.is_directory:
            return

        # Check if file was moved to our watched location (atomic write pattern)
        if hasattr(event, 'dest_path'):
            dest_path = Path(event.dest_path).resolve()
            if dest_path == self._watched_file:
                logger.info(f'Configuration file updated via move: {dest_path}')
                try:
                    self._callback(self._watched_file)
                except Exception as e:
                    logger.error(f'Error in config change callback: {e}', exc_info=True)


class WatchdogConfigWatcher(ConfigWatcher):
    """Configuration file watcher using watchdog library."""

    def __init__(self):
        self._observer: Optional[Observer] = None
        self._watched_path: Optional[Path] = None
        self._callback: Optional[Callable[[Path], None]] = None

    def watch(self, path: Path, callback: Callable[[Path], None]) -> None:
        """Start watching a file for changes.

        Args:
            path: Path to file to watch
            callback: Function to call when file changes
        """
        # Stop any existing watching
        self.stop_watching()

        self._watched_path = path.resolve()
        self._callback = callback

        # Create the directory if it doesn't exist
        directory = self._watched_path.parent
        directory.mkdir(parents=True, exist_ok=True)

        # Create the observer and handler
        self._observer = Observer()
        handler = _ConfigFileHandler(self._watched_path, callback)

        # Watch the directory containing the file
        self._observer.schedule(handler, str(directory), recursive=False)
        self._observer.start()

        logger.info(f'Started watching configuration file: {self._watched_path}')

    def stop_watching(self) -> None:
        """Stop watching for changes."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)  # Wait up to 5 seconds for clean shutdown

            if self._observer.is_alive():
                logger.warning('Config watcher did not stop cleanly')
            else:
                logger.info('Stopped watching configuration file')

            self._observer = None
            self._watched_path = None
            self._callback = None

    def is_watching(self) -> bool:
        """Check if currently watching for changes."""
        return self._observer is not None and self._observer.is_alive()

    def __del__(self):
        """Ensure watcher is stopped on cleanup."""
        if self._observer is not None:
            try:
                self.stop_watching()
            except Exception as e:
                logger.error(f'Error stopping config watcher during cleanup: {e}')
