# Agent Instructions

This document provides instructions for AI agents working on this codebase.

## Project Overview

This project, "SPN-Benchmarks", appears to be a collection of benchmarks for Stochastic Petri Nets (SPNs), likely involving Graph Neural Networks (GNNs).

## Development Guidelines

### Dependencies

This project uses `uv` as the primary tool for managing Python dependencies, with `pyproject.toml` as the single source of truth. When adding new dependencies, you should add them to the `[project.dependencies]` section of `pyproject.toml`.

`environment.yml` is also present for conda users and should be kept in sync with `pyproject.toml` when possible. `conda` is a secondary option for dependency management. `requirements.txt` is provided for compatibility with tools that do not support `pyproject.toml`.

### Code Style

This project uses `black` for code formatting. Please format your code before submitting. The command to run `black` is:
`black . -l 120 --py39`

### Testing

This project uses `pytest` for testing. Before submitting changes, please ensure that all existing tests pass and that any new functionality is covered by new tests written with `pytest`.
