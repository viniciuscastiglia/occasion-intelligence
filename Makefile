.PHONY: ambiente teste teste-dados qualidade dados-brutos bases auditoria baseline tudo app investigacao-datas eda dobras peso escada h4 embaralhado calibracao produto cortes receita avaliacao-2008 experimentos avaliado teste-final

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

# exige a árvore limpa nos arquivos da receita (ARQUIVOS_DA_RECEITA): o artefato guarda a assinatura
avaliado:
	uv run python -m occasion.pipeline avaliado

# H8 em 2017-2018: só depois da tag receita-congelada e de `make avaliado` (seção 7 do pré-registro)
teste-final:
	uv run python -m occasion.experimentos teste-final

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

avaliacao-2008:
	uv run python -m occasion.experimentos avaliacao-2008

# a cadeia inteira dos experimentos de 2008-2009, na ordem (make sem -j roda em sequência)
experimentos: dobras peso escada h4 embaralhado calibracao produto cortes receita avaliacao-2008

app:
	uv run streamlit run app/streamlit_app.py
