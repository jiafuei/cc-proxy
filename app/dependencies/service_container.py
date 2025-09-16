"""Compatibility shim for relocated service container."""

from app.di.container import ServiceContainer, build_service_container

__all__ = ['ServiceContainer', 'build_service_container']
