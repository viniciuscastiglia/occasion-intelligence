"""Funções de variável usadas no treino e na inferência, e os domínios fechados das categóricas."""

from typing import Any, cast

import pandas as pd

from occasion.config import FAIXAS_HORARIO, LOCAIS, SALARIO_MINIMO
from occasion.inference.contract import CODIGO_UF

COLUNAS_CONTEXTO: list[str] = [
    "dia_semana",
    "faixa_horario",
    "local",
    "uf",
    "regiao",
    "situacao",
    "idade",
    "sexo",
    "renda_pc_sm",
    "n_moradores_uc",
]

_REGIAO_POR_DIGITO: dict[int, str] = {
    1: "norte",
    2: "nordeste",
    3: "sudeste",
    4: "sul",
    5: "centro_oeste",
}
_UF_MIN, _UF_MAX = 11, 53

CATEGORICAS: list[str] = [
    "dia_semana",
    "faixa_horario",
    "local",
    "uf",
    "regiao",
    "situacao",
    "sexo",
]
NUMERICAS: list[str] = ["idade", "renda_pc_sm", "n_moradores_uc"]

# Sexta, sábado e domingo (1 é segunda): o agrupamento "sexta a domingo" do TAPI, usado na EDA.
DIAS_FIM_DE_SEMANA: frozenset[int] = frozenset({5, 6, 7})

# Valores possíveis de cada categórica, montados a partir de constantes e nunca dos dados: assim
# nenhuma base (muito menos a de 2017-2018) decide quais níveis existem. `tem_pizza` (parte 2) e
# `fim_de_semana` (derivada de `dia_semana`) entram porque também viram colunas dos modelos.
DOMINIOS: dict[str, list[Any]] = {
    "dia_semana": list(range(1, 8)),
    "faixa_horario": list(FAIXAS_HORARIO),
    "local": list(LOCAIS),
    "uf": list(CODIGO_UF.values()),
    "regiao": list(_REGIAO_POR_DIGITO.values()),
    "situacao": ["urbano", "rural"],
    "sexo": ["masculino", "feminino"],
    "tem_pizza": [0, 1],
    "fim_de_semana": [0, 1],
}


def regiao_da_uf(uf: int) -> str:
    """Grande região pelo primeiro dígito do código IBGE da UF (11 a 53)."""
    if not _UF_MIN <= uf <= _UF_MAX or uf // 10 not in _REGIAO_POR_DIGITO:
        raise ValueError(f"código de UF fora do domínio de {_UF_MIN} a {_UF_MAX}: {uf}")
    return _REGIAO_POR_DIGITO[uf // 10]


def renda_pc_sm(renda_total_uc: pd.Series, n_moradores_uc: pd.Series, edicao: str) -> pd.Series:
    """Renda mensal por pessoa da unidade de consumo, em salários mínimos da edição."""
    if edicao not in SALARIO_MINIMO:
        raise ValueError(f"edição sem salário mínimo em config.SALARIO_MINIMO: {edicao}")
    return renda_total_uc / n_moradores_uc / SALARIO_MINIMO[edicao]


def adicionar_derivadas(X: pd.DataFrame) -> pd.DataFrame:
    """Cópia de `X` com as variáveis derivadas: `fim_de_semana` (1 de sexta a domingo, senão 0).

    Os modelos chamam esta função no `fit` e no `predict_proba`, para as bases e o contrato não
    mudarem. Sem `dia_semana` em `X`, devolve só a cópia.
    """
    com = X.copy()
    if "dia_semana" in com.columns:
        dia = cast(pd.Series, com["dia_semana"])
        com["fim_de_semana"] = dia.isin(sorted(DIAS_FIM_DE_SEMANA)).astype("int8")
    return com


def checar_dominios(X: pd.DataFrame) -> None:
    """Levanta `ValueError` se uma coluna de `DOMINIOS` presente em `X` tem valor fora do domínio.

    Colunas ausentes são ignoradas; valor ausente (`NaN`, `None`) conta como fora do domínio.
    """
    for coluna, dominio in DOMINIOS.items():
        if coluna not in X.columns:
            continue
        valores = cast(pd.Series, X[coluna])
        fora = ~valores.isin(dominio)
        if bool(fora.any()):
            valor = cast(pd.Series, valores[fora]).iloc[0]
            valor = valor.item() if hasattr(valor, "item") else valor  # escalar do NumPy
            raise ValueError(
                f"valor fora do domínio na coluna {coluna}: {valor!r} (domínio: {dominio})"
            )
