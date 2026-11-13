"""Funções de variável usadas no treino e na inferência."""

import pandas as pd

from occasion.config import SALARIO_MINIMO

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
