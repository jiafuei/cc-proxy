# CLAUDE.md
This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Code organization
- main.py (Entry point)
- app/ (application package)
  - app/routers/ - Routers and API layer handlers
  - app/middlewares/ - Custom middlewares
  - app/dependencies/ - App dependency injection
  - app/services/{service}/ - Business logics
  - app/services/{service}/models.py - package-level pydantic models, abstract classes
  - app/common/models.py - Common models used across multiple packages
  - app/{component}/tests/ - Unit tests

## Commands
- `uvx ruff check --fix`: Run linter and fix
- `uvx ruff format path/to/code/`: Run formatter for styling
- `python -m pytest path/to/test -v`: Run tests for directory or file

## Configuration Files
- `config.example.yaml`: The example config

## Testing-first implementation
- When writing code, always start by adding empty unit test, and incrementally build up tests for code implementation
- Do not make large changes immediately, always create small tests, then the implementation to make the test pass, and keep repeating this until the feature is finished

## Developer notes
- Design clear interfaces for component interactions
- Dependency injection must be used extensively when creating components and services.
- Components should be composible to allow for injecting mocks for tests
- Do not lint or format tests
