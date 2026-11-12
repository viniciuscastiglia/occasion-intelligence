.PHONY: ambiente teste teste-dados qualidade

ambiente:
	uv sync

teste:
	uv run pytest -m "not dados"

teste-dados:
	uv run pytest

qualidade:
	uv run ruff check . && uv run pyright
