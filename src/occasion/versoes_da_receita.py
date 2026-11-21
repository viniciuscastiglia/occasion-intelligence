"""Versões das bibliotecas que mudam os números da receita, fixadas no congelamento.

Este arquivo é assinado com a receita (`ARQUIVOS_DA_RECEITA`) no lugar de `pyproject.toml` e
`uv.lock`, que o Plano 3 precisa mudar (FastAPI e o cliente do briefing). O teste final confere as
versões instaladas (`importlib.metadata`) contra este dicionário e recusa rodar se alguma
divergir. Os valores são literais: as versões instaladas quando a receita foi congelada.
"""

VERSOES_DA_RECEITA: dict[str, str] = {
    "numpy": "2.5.4",
    "pandas": "3.1.1",
    "scipy": "1.18.1",
    "scikit-learn": "1.9.1",
    "lightgbm": "4.7.0",
    "pyarrow": "26.0.1",
}
