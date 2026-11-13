.PHONY: ambiente teste teste-dados qualidade dados-brutos bases auditoria baseline tudo

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

bases:
	uv run python -m occasion.pipeline interim --edicao 2008-2009
	uv run python -m occasion.pipeline interim --edicao 2017-2018
	uv run python -m occasion.pipeline ocasioes
	uv run python -m occasion.pipeline modelagem

auditoria:
	uv run python -m occasion.pipeline auditoria

baseline:
	uv run python -m occasion.pipeline baseline

tudo: dados-brutos bases auditoria baseline
