"""Ocasiões (pessoa x dia x hora) e base pessoa x dia a partir dos itens consumidos.

Ocasião: itens da mesma pessoa, no mesmo dia, na mesma hora cheia. "Pizza com refrigerante"
significa os dois na mesma ocasião. As marcações são sim ou não por ocasião, então um item
repetido na mesma hora não conta em dobro.
"""

from typing import cast

import pandas as pd

from occasion.config import CODIGOS

COLUNAS_OCASIAO: list[str] = [
    "id_ocasiao",
    "id_dia",
    "id_pessoa",
    "hora",
    "local",
    "tem_pizza",
    "tem_refri",
    "tem_guarana",
    "tem_cola",
]
COLUNAS_CONSTANTES_NO_DIA: list[str] = ["dia_semana", "peso_final", "renda_total_uc"]


def _col(df: pd.DataFrame, nome: str) -> pd.Series:
    """Devolve a coluna como Series (as colunas do esquema são únicas)."""
    return cast(pd.Series, df[nome])


def _em(df: pd.DataFrame, codigos: frozenset[int]) -> pd.Series:
    """Marca os itens cujo código de alimento está na lista."""
    return _col(df, "cod_alimento").isin(sorted(codigos))


def montar_ocasioes(consumo: pd.DataFrame, edicao: str) -> pd.DataFrame:
    """Uma linha por `id_dia` e `hora` não nula; itens sem hora não formam ocasião.

    O local é o do item de pizza, se houver; senão o do primeiro item de refrigerante;
    senão o do primeiro item. O número de itens sem hora fica em `attrs["itens_sem_hora"]`.
    """
    codigos = CODIGOS[edicao]
    itens = pd.DataFrame(
        {
            "id_dia": consumo["id_dia"],
            "id_pessoa": consumo["id_pessoa"],
            "hora": consumo["hora"],
            "local": consumo["local"],
            "pizza": _em(consumo, codigos["pizza"]),
            "guarana": _em(consumo, codigos["guarana"]),
            "cola": _em(consumo, codigos["cola"]),
        }
    )
    itens["refri"] = itens["guarana"] | itens["cola"] | _em(consumo, codigos["outros_refri"])
    sem_hora = int(_col(itens, "hora").isna().sum())
    itens = itens.dropna(subset=["hora"]).copy()
    itens["hora"] = itens["hora"].astype("int64")

    # Prioridade do local: 0 pizza, 1 refrigerante, 2 qualquer item. Ordenação estável.
    itens["prioridade"] = 2
    itens.loc[itens["refri"], "prioridade"] = 1
    itens.loc[itens["pizza"], "prioridade"] = 0
    chave = ["id_dia", "hora"]
    primeiros = itens.sort_values("prioridade", kind="stable").drop_duplicates(chave)
    local = primeiros.set_index(chave)[["local"]]
    marcas = itens.groupby(chave)[["pizza", "refri", "guarana", "cola"]].any()
    marcas.columns = pd.Index(["tem_pizza", "tem_refri", "tem_guarana", "tem_cola"])
    ocasioes = marcas.join(local).reset_index()
    pessoa = dict(zip(itens["id_dia"], itens["id_pessoa"]))
    ocasioes["id_pessoa"] = _col(ocasioes, "id_dia").map(pessoa)
    ocasioes["id_ocasiao"] = _col(ocasioes, "id_dia") + "|" + _col(ocasioes, "hora").astype(str)
    ocasioes = cast(pd.DataFrame, ocasioes[COLUNAS_OCASIAO]).reset_index(drop=True)
    ocasioes.attrs["itens_sem_hora"] = sem_hora
    return ocasioes


def montar_pessoa_dia(
    consumo: pd.DataFrame, morador: pd.DataFrame, ocasioes: pd.DataFrame
) -> pd.DataFrame:
    """Uma linha por `id_dia`, com as colunas do morador e as marcas de pizza do dia.

    `tem_pizza_dia`: algum item de pizza no dia, com ou sem hora. `tem_pizza_refri` e
    `tem_pizza_guarana`: alguma ocasião do dia (mesma hora) com pizza e refrigerante, ou
    pizza e guaraná. Levanta `ValueError` se alguma pessoa do consumo não estiver no morador
    ou se `dia_semana`, `peso_final` ou `renda_total_uc` variarem dentro do dia.
    """
    ausentes = set(consumo["id_pessoa"]) - set(morador["id_pessoa"])
    if ausentes:
        raise ValueError(
            f"{len(ausentes)} pessoa(s) do consumo sem linha no registro de moradores "
            f"(exemplo: {min(ausentes)})"
        )
    for coluna in COLUNAS_CONSTANTES_NO_DIA:
        variam = int((cast(pd.Series, consumo.groupby("id_dia")[coluna].nunique()) > 1).sum())
        if variam:
            raise ValueError(f"{coluna} varia dentro do dia em {variam} dia(s)")

    dias = consumo.drop_duplicates("id_dia")[
        ["id_dia", "id_pessoa", "quadro", *COLUNAS_CONSTANTES_NO_DIA]
    ].reset_index(drop=True)
    dias = dias.merge(morador, on="id_pessoa", how="left", validate="many_to_one")
    pizza_nos_itens = _marca_pizza(consumo).groupby(_col(consumo, "id_dia")).any().to_dict()
    conjunta = (
        ocasioes.assign(
            pizza_refri=ocasioes["tem_pizza"] & ocasioes["tem_refri"],
            pizza_guarana=ocasioes["tem_pizza"] & ocasioes["tem_guarana"],
        )
        .groupby("id_dia")[["pizza_refri", "pizza_guarana"]]
        .any()
    )
    id_dia = _col(dias, "id_dia")
    dias["tem_pizza_dia"] = id_dia.map(pizza_nos_itens).astype(bool)
    for coluna, origem in (
        ("tem_pizza_refri", "pizza_refri"),
        ("tem_pizza_guarana", "pizza_guarana"),
    ):
        dias[coluna] = (
            id_dia.map(cast(pd.Series, conjunta[origem]).to_dict()).fillna(False).astype(bool)
        )
    return dias


def _marca_pizza(consumo: pd.DataFrame) -> pd.Series:
    """Marca os itens de pizza da edição do consumo."""
    edicao = str(_col(consumo, "edicao").iloc[0])
    return _em(consumo, CODIGOS[edicao]["pizza"])
