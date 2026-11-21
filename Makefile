.PHONY: ambiente teste teste-dados qualidade dados-brutos bases auditoria baseline tudo app investigacao-datas eda dobras peso escada h4 embaralhado calibracao produto cortes receita experimentos avaliado

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

avaliado:
	uv run python -m occasion.pipeline avaliado

tudo: dados-brutos bases auditoria baseline

investigacao-datas:
	uv run python scripts/investigacao_datas.py

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda_2008.ipynb

dobras:
	uv run python -m occasion.experimentos dobras

peso:
	uv run python -m occasion.experimentos peso

escada:
	uv run python -m occasion.experimentos escada

h4:
	uv run python -m occasion.experimentos h4

embaralhado:
	uv run python -m occasion.experimentos embaralhado

calibracao:
	uv run python -m occasion.experimentos calibracao

produto:
	uv run python -m occasion.experimentos produto

cortes:
	uv run python -m occasion.experimentos cortes

receita:
	uv run python -m occasion.experimentos receita

# a cadeia inteira dos experimentos de 2008-2009, na ordem (make sem -j roda em sequência)
experimentos: dobras peso escada h4 embaralhado calibracao produto cortes receita

app:
	uv run streamlit run app/streamlit_app.py
