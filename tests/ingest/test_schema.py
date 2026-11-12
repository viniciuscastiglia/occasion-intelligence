"""Testes dos esquemas comuns de consumo e morador (contrato entre as duas edições)."""

from collections.abc import Callable

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from occasion.ingest.schema import (
    ESQUEMA_CONSUMO,
    ESQUEMA_MORADOR,
    validar_consumo,
    validar_morador,
)


def _exige_falha(validar: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame, motivo: str):
    """Exige SchemaErrors com ao menos uma falha da coluna ou da regra `motivo`.

    Alterar uma chave pode quebrar também regras derivadas (estrato, id_pessoa), então não se
    exige que `motivo` seja a única falha.
    """
    with pytest.raises(SchemaErrors) as excecao:
        validar(df)
    casos = excecao.value.failure_cases
    assert ((casos["column"] == motivo) | (casos["check"] == motivo)).any(), casos


def _chaves(edicao: str, quadro: int) -> dict[str, object]:
    uf, estrato, upa, dom, uc, inf = 35, "3501", "350000001", 1, 1, 2
    id_pessoa = f"{edicao}|{upa}|{dom}|{uc}|{inf}"
    return {
        "edicao": edicao,
        "uf": uf,
        "cod_upa": upa,
        "estrato": f"{uf}-{estrato}",
        "num_dom": dom,
        "num_uc": uc,
        "cod_informante": inf,
        "id_pessoa": id_pessoa,
        "quadro": quadro,
        "id_dia": f"{id_pessoa}|{quadro}",
    }


def consumo_valido() -> pd.DataFrame:
    """Duas linhas válidas, uma de cada edição, com hora nula na segunda."""
    linhas = [
        {
            **_chaves("2017-2018", 72),
            "cod_alimento": 8500903,
            "hora": 20,
            "local": "em_casa",
            "dia_semana": 7,
            "peso_final": 123.4,
            "renda_total_uc": 2500.0,
        },
        {
            **_chaves("2008-2009", 71),
            "cod_alimento": 8200301,
            "hora": None,
            "local": "indeterminado",
            "dia_semana": 1,
            "peso_final": 80.0,
            "renda_total_uc": 0.0,
        },
    ]
    df = pd.DataFrame(linhas)
    df["hora"] = df["hora"].astype("Int64")
    return df


def morador_valido() -> pd.DataFrame:
    """Duas pessoas válidas na mesma unidade de consumo."""
    linhas = []
    for informante in (1, 2):
        chaves = _chaves("2017-2018", 72)
        id_pessoa = f"2017-2018|350000001|1|1|{informante}"
        linhas.append(
            {
                **{k: chaves[k] for k in ("edicao", "uf", "cod_upa", "estrato")},
                "num_dom": 1,
                "num_uc": 1,
                "cod_informante": informante,
                "id_pessoa": id_pessoa,
                "situacao": "urbano",
                "idade": 30 + informante,
                "sexo": "feminino",
                "n_moradores_uc": 2,
            }
        )
    return pd.DataFrame(linhas)


def test_consumo_valido_passa() -> None:
    saida = validar_consumo(consumo_valido())
    assert len(saida) == 2
    assert saida["hora"].isna().tolist() == [False, True]


def test_morador_valido_passa() -> None:
    assert len(validar_morador(morador_valido())) == 2


def test_esquemas_expostos() -> None:
    assert "id_dia" in ESQUEMA_CONSUMO.columns
    assert "n_moradores_uc" in ESQUEMA_MORADOR.columns


@pytest.mark.parametrize(
    ("coluna", "valor", "trecho"),
    [
        ("dia_semana", 8, "dia_semana"),
        ("dia_semana", 0, "dia_semana"),
        ("local", "rua", "local"),
        ("peso_final", 0.0, "peso_final"),
        ("renda_total_uc", -1.0, "renda_total_uc"),
        ("hora", 24, "hora"),
        ("hora", -1, "hora"),
        ("uf", 10, "uf"),
        ("uf", 54, "uf"),
        ("cod_alimento", 850090, "cod_alimento"),
        ("cod_alimento", 85009030, "cod_alimento"),
        ("cod_upa", "", "cod_upa"),
        ("num_dom", 0, "num_dom"),
        ("edicao", "2012-2013", "edicao"),
    ],
)
def test_consumo_valor_invalido_falha(coluna: str, valor: object, trecho: str) -> None:
    df = consumo_valido()
    df[coluna] = pd.Series([valor, valor], dtype=df[coluna].dtype)
    _exige_falha(validar_consumo, df, trecho)


@pytest.mark.parametrize(
    ("edicao", "quadro"),
    [("2008-2009", 73), ("2017-2018", 71)],
)
def test_consumo_quadro_incompativel_com_a_edicao_falha(edicao: str, quadro: int) -> None:
    df = consumo_valido().iloc[[0]].copy()
    df["edicao"] = edicao
    df["quadro"] = quadro
    # mantém ids coerentes para que só a regra do quadro seja violada
    df["id_pessoa"] = f"{edicao}|350000001|1|1|2"
    df["id_dia"] = df["id_pessoa"] + f"|{quadro}"
    _exige_falha(validar_consumo, df, "quadro_conforme_edicao")


@pytest.mark.parametrize(
    ("coluna", "valor", "regra"),
    [
        ("estrato", "41-3501", "estrato_montado"),
        ("id_pessoa", "2017-2018|350000001|1|1|9", "id_pessoa_montado"),
        ("id_dia", "2017-2018|350000001|1|1|2|73", "id_dia_montado"),
    ],
)
def test_consumo_campo_montado_incoerente_falha(coluna: str, valor: str, regra: str) -> None:
    df = consumo_valido().iloc[[0]].copy()
    df[coluna] = valor
    _exige_falha(validar_consumo, df, regra)


def test_consumo_coluna_extra_falha() -> None:
    df = consumo_valido()
    df["extra"] = 1
    _exige_falha(validar_consumo, df, "column_in_schema")


def test_consumo_coluna_ausente_falha() -> None:
    df = consumo_valido().drop(columns=["peso_final"])
    _exige_falha(validar_consumo, df, "column_in_dataframe")


def test_consumo_nao_coage_tipos() -> None:
    df = consumo_valido()
    df["uf"] = df["uf"].astype(str)
    _exige_falha(validar_consumo, df, "uf")


def test_consumo_hora_float_falha() -> None:
    df = consumo_valido()
    df["hora"] = df["hora"].astype("float64")
    _exige_falha(validar_consumo, df, "hora")


@pytest.mark.parametrize(
    ("coluna", "valor"),
    [
        ("situacao", "periurbano"),
        ("idade", 131),
        ("idade", -1),
        ("sexo", "outro"),
        ("n_moradores_uc", 0),
    ],
)
def test_morador_valor_invalido_falha(coluna: str, valor: object) -> None:
    df = morador_valido()
    df[coluna] = valor
    _exige_falha(validar_morador, df, coluna)


def test_morador_id_pessoa_repetida_falha() -> None:
    df = morador_valido()
    df.loc[1, ["cod_informante", "id_pessoa"]] = [1, df.loc[0, "id_pessoa"]]
    _exige_falha(validar_morador, df, "id_pessoa")


def test_morador_id_pessoa_incoerente_falha() -> None:
    df = morador_valido()
    df.loc[0, "id_pessoa"] = "2017-2018|350000001|1|1|9"
    _exige_falha(validar_morador, df, "id_pessoa_montado")


def test_morador_n_moradores_incoerente_falha() -> None:
    df = morador_valido()
    df["n_moradores_uc"] = 5
    _exige_falha(validar_morador, df, "n_moradores_conforme_linhas")


def test_morador_coluna_extra_falha() -> None:
    df = morador_valido()
    df["extra"] = 1
    _exige_falha(validar_morador, df, "column_in_schema")
