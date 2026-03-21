# Contributing

Thanks for contributing to this project.

## Before You Start

- Read [README.md](README.md) for the project overview.
- Use [QUICKSTART.md](QUICKSTART.md) for the local setup path.
- Check existing issues and pull requests before opening new work.

## Development Setup

```bash
git clone https://github.com/KazKozDev/dspy-optimization-patterns.git
cd dspy-optimization-patterns
poetry install --with dev
cp .env.example .env
```

Set `OPENAI_API_KEY` before running optimization or API flows that require model access.

## Recommended Workflow

1. Create a branch for your change.
2. Keep changes scoped to one problem or feature.
3. Update docs, examples, or tests when behavior changes.
4. Run local checks before opening a pull request.

## Local Checks

```bash
make lint
make test
```

Useful commands:

```bash
make prepare-sample-data
make optimize-rag
make run-api
```

## Pull Requests

Please make sure your pull request:

- explains the problem and the chosen solution
- stays focused and avoids unrelated cleanup
- includes test coverage or explains why tests were not added
- updates README or other docs if the public workflow changed

Small, reviewable pull requests are preferred over large multi-purpose changes.

## Issues

When opening an issue, include:

- what you expected
- what happened instead
- steps to reproduce
- relevant logs, stack traces, or screenshots
- environment details if the bug is setup-dependent

## Style Notes

- Follow the existing Python and project structure.
- Keep documentation concise and operational.
- Avoid introducing new tooling unless it solves a clear project need.
