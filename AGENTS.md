# AGENTS.md

Python 3.12+ application to recognize text structures in images of documents.
It uses the htrflow package(https://ai-riksarkivet.github.io/htrflow/v0.2.6/index.html)
for Yolo segmentation and Kraken for baseline and linemask detection.
The package also includes utilities for processing XML files, including
functions for validating input and output data using Pydantic models.
It uses uv as package manager and includes a Makefile for easy installation and development setup.

## Commands

- Makefile @Makefile

## Conventions

- Use htrflow v0.2.6 for Yolo segmentation
- Use Kraken for baseline and linemask detection
- Use Pydantic v2 for input and output validation
- Use uv for package management
- Use lxml and pagexml-tools for XML processing
- Use SnakeCase for function and variable names
- Use PascalCase for class names
- Use pathlib for file path handling, not os.path

## Rules

- Follow standard Python packaging conventions for structure and distribution
- Use loguru for logging and create an environment variable to define the 
  log level and persistance of logs
- Ensure all functions and classes are well-documented with docstrings
- Include type hints for all functions and methods
- Do not add new dependencies without asking first
- Do not change dependency versions without asking first
- Do not change the structure of the package without asking first
- Ask when changing pyproject.toml
- Create new git branches for new features or bug fixes and follow a consistent commit message format

## Testing

- Use pytest for testing and include tests for all major functionalities
- Ensure tests cover edge cases and potential failure points
- Use mocking to isolate tests and avoid dependencies on external services or resources
- Do linting and formating checks as part of the testing process to maintain code quality

## Documentation

- Use Sphinx for documentation and include comprehensive documentation for all functions, classes, and modules
- Include examples and usage guides in the documentation
- Keep documentation up to date with code changes and ensure it is clear and easy to understand
- Do not yet push the documentation, but create it locally and ask for review before pushing to the repository