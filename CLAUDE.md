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
- `uvx ruff check --fix && uvx ruff format path/to/code`: Run linter, fix styles and formatter
- `python -m pytest path/to/test -v`: Run tests for directory or file

## Configuration Files
- `config.example.yaml`: The example static server config
- `user.example.yaml` The example dynamic user config


## Developer notes
- Design clear interfaces for component interactions and dependency injection
- Dependency injection must be used extensively when creating components and services.
- Components should be composible to allow for injecting mocks for tests
- Lint guidelines: Do not lint or format tests. Run linter and formatter together using &&
- Each package should be focused and only handle one or two concerns
- Each method and component should be easily mockable and unit testable
- Design and write tests first before implementation, each test must be focused
- `app.config.log.get_logger` is direct, compatible replacement of `logging.getLogger`. use it to get logger instead
- add '[ai]' to message when doing git commit. eg: 'feat: [ai] some feature'
- Prioritize simple, readable code with minimal abstraction. Strive for elegant, minimal solutions that reduce complexity. Avoid premature optimization and over-engineering.
- Ignore backward compatibility unless specified.