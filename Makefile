.PHONY: ambiente teste teste-dados qualidade dados-brutos auditoria

ambiente:
	uv sync

teste:
	uv run pytest -m "not dados"

teste-dados:
	uv run pytest

qualidade:
	uv run ruff check . && uv run pyright

dados-brutos:
	uv run python -m occasion.ingest.download

auditoria:
	uv run python -m occasion.pipeline auditoria
