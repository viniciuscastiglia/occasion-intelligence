"""Bases de modelagem: células da parte 1 e ocasiões com refrigerante da parte 2.

Parte 1: uma linha por dia-pessoa x faixa de horário x local (5 x 2 = 10 células por dia). O alvo
é 1 se alguma ocasião do dia naquela faixa e local tem o evento (máximo, nunca soma). Ocasiões com
local `indeterminado` não pertencem a nenhuma célula e são contadas.
Parte 2: uma linha por ocasião com refrigerante e local determinado.
"""

from typing import cast

import pandas as pd

from occasion.config import FAIXAS_HORARIO, LOCAIS, faixa_da_hora
from occasion.features.variaveis import COLUNAS_CONTEXTO, regiao_da_uf, renda_pc_sm

_IDENTIFICACAO: list[str] = ["id_dia", "id_pessoa", "edicao", "cod_upa", "estrato", "peso_final"]
_DO_DIA: list[str] = [
    "dia_semana",
    "uf",
    "regiao",
    "situacao",
    "idade",
    "sexo",
    "renda_pc_sm",
    "n_moradores_uc",
]


def _col(df: pd.DataFrame, nome: str) -> pd.Series:
    return cast(pd.Series, df[nome])


def _faixa_das_horas(horas: pd.Series) -> pd.Series:
    """Faixa de horário de cada hora; hora fora de 0 a 23 levanta `ValueError`."""
    tabela = {hora: faixa_da_hora(hora) for hora in range(24)}
    faixas = horas.map(tabela)
    if faixas.isna().any():
        invalida = cast(pd.Series, horas[faixas.isna()]).iloc[0]
        raise ValueError(f"hora fora do intervalo de 0 a 23: {invalida}")
    return faixas


def _contexto_do_dia(pessoa_dia: pd.DataFrame) -> pd.DataFrame:
    """Identificação e variáveis de contexto que não dependem da faixa nem do local."""
    dias = pessoa_dia.copy()
    dias["regiao"] = _col(dias, "uf").map(regiao_da_uf)
    edicoes = _col(dias, "edicao").unique()
    if len(edicoes) != 1:
        raise ValueError(f"a base deve ter uma única edição, veio: {list(edicoes)}")
    dias["renda_pc_sm"] = renda_pc_sm(
        _col(dias, "renda_total_uc"), _col(dias, "n_moradores_uc"), str(edicoes[0])
    )
    return cast(pd.DataFrame, dias[[*_IDENTIFICACAO, *_DO_DIA]])


def montar_celulas(pessoa_dia: pd.DataFrame, ocasioes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por `id_dia`, faixa e local, com os alvos `y_pizza_refri` e `y_pizza_guarana`.

    `attrs["dias_positivos_excluidos"]`: dias com pizza e refrigerante na mesma ocasião que não
    ficaram com nenhuma célula positiva porque o local das ocasiões positivas é indeterminado.
    """
    dias = _contexto_do_dia(pessoa_dia)
    grade = pd.DataFrame(
        [(f, loc) for f in FAIXAS_HORARIO for loc in LOCAIS], columns=["faixa_horario", "local"]
    )
    celulas = dias.merge(grade, how="cross")
    celulas["id_celula"] = (
        _col(celulas, "id_dia")
        + "|"
        + _col(celulas, "faixa_horario")
        + "|"
        + _col(celulas, "local")
    )

    oc = ocasioes.assign(
        faixa_horario=_faixa_das_horas(_col(ocasioes, "hora")),
        y_pizza_refri=_col(ocasioes, "tem_pizza") & _col(ocasioes, "tem_refri"),
        y_pizza_guarana=_col(ocasioes, "tem_pizza") & _col(ocasioes, "tem_guarana"),
    )
    determinadas = cast(pd.DataFrame, oc[_col(oc, "local").isin(LOCAIS)])
    alvos = (
        determinadas.groupby(["id_dia", "faixa_horario", "local"])[
            ["y_pizza_refri", "y_pizza_guarana"]
        ]
        .any()
        .astype("int8")
        .reset_index()
    )
    celulas = celulas.merge(
        alvos, on=["id_dia", "faixa_horario", "local"], how="left", validate="one_to_one"
    )
    for alvo in ("y_pizza_refri", "y_pizza_guarana"):
        celulas[alvo] = celulas[alvo].fillna(0).astype("int8")

    celulas_positivas = cast(pd.DataFrame, celulas[celulas["y_pizza_refri"] == 1])
    dias_com_celula = set(_col(celulas_positivas, "id_dia"))
    positivos = cast(pd.Series, _col(pessoa_dia, "id_dia")[_col(pessoa_dia, "tem_pizza_refri")])
    excluidos = int((~positivos.isin(sorted(dias_com_celula))).sum())

    colunas = ["id_celula", *_IDENTIFICACAO, *COLUNAS_CONTEXTO, "y_pizza_refri", "y_pizza_guarana"]
    resultado = cast(pd.DataFrame, celulas[colunas]).reset_index(drop=True)
    resultado.attrs["dias_positivos_excluidos"] = excluidos
    return resultado


def montar_ocasioes_refri(pessoa_dia: pd.DataFrame, ocasioes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ocasião com refrigerante e local determinado, com `tem_pizza` e `y_guarana`.

    `attrs["ocasioes_excluidas"]`: ocasiões com refrigerante fora da base por local indeterminado.
    """
    com_refri = cast(pd.DataFrame, ocasioes[_col(ocasioes, "tem_refri")])
    local_ok = _col(com_refri, "local").isin(LOCAIS)
    base = cast(pd.DataFrame, com_refri[local_ok])
    excluidas = int((~local_ok).sum())

    base = cast(
        pd.DataFrame,
        base.assign(
            faixa_horario=_faixa_das_horas(_col(base, "hora")),
            tem_pizza=_col(base, "tem_pizza").astype("int8"),
            y_guarana=_col(base, "tem_guarana").astype("int8"),
        ),
    )
    base = base.drop(columns=["id_pessoa"]).merge(
        _contexto_do_dia(pessoa_dia), on="id_dia", how="left", validate="many_to_one"
    )
    if _col(base, "id_pessoa").isna().any():
        raise ValueError("ocasião com refrigerante sem linha em pessoa_dia")
    colunas = ["id_ocasiao", *_IDENTIFICACAO, *COLUNAS_CONTEXTO, "tem_pizza", "y_guarana"]
    resultado = cast(pd.DataFrame, base[colunas]).reset_index(drop=True)
    resultado.attrs["ocasioes_excluidas"] = excluidas
    return resultado
