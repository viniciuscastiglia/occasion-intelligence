"""Esquemas pandera do esquema comum: o contrato entre os parsers das duas edições da POF.

Validação estrita: colunas a mais ou a menos falham e os tipos são conferidos sem coerção,
para que um erro de parser não fique escondido.
"""

from typing import cast

import pandas as pd
import pandera.pandas as pa

from occasion.config import EDICOES

QUADROS_POR_EDICAO: dict[str, tuple[int, int]] = {
    "2008-2009": (71, 72),
    "2017-2018": (72, 73),
}
CHAVE_UC: list[str] = ["edicao", "cod_upa", "num_dom", "num_uc"]


def _col(df: pd.DataFrame, nome: str) -> pd.Series:
    """Devolve a coluna como Series (as colunas do esquema são únicas)."""
    return cast(pd.Series, df[nome])


def _quadro_conforme_edicao(df: pd.DataFrame) -> pd.Series:
    validos = [QUADROS_POR_EDICAO.get(str(e), ()) for e in df["edicao"]]
    return pd.Series([q in v for q, v in zip(df["quadro"], validos)], index=df.index)


def _estrato_montado(df: pd.DataFrame) -> pd.Series:
    return _col(df, "estrato").str.split("-", n=1).str[0] == _col(df, "uf").astype(str)


def _id_pessoa_esperado(df: pd.DataFrame) -> pd.Series:
    return (
        _col(df, "edicao")
        + "|"
        + _col(df, "cod_upa")
        + "|"
        + _col(df, "num_dom").astype(str)
        + "|"
        + _col(df, "num_uc").astype(str)
        + "|"
        + _col(df, "cod_informante").astype(str)
    )


def _id_pessoa_montado(df: pd.DataFrame) -> pd.Series:
    return _col(df, "id_pessoa") == _id_pessoa_esperado(df)


def _id_dia_montado(df: pd.DataFrame) -> pd.Series:
    return _col(df, "id_dia") == _col(df, "id_pessoa") + "|" + _col(df, "quadro").astype(str)


def _n_moradores_conforme_linhas(df: pd.DataFrame) -> pd.Series:
    linhas = df.groupby(CHAVE_UC)["cod_informante"].transform("size")
    return _col(df, "n_moradores_uc") == linhas


def _colunas_chave() -> dict[str, pa.Column]:
    """Colunas de identificação comuns a consumo e morador."""
    return {
        "edicao": pa.Column(str, pa.Check.isin(list(EDICOES))),
        "uf": pa.Column(int, pa.Check.in_range(11, 53)),
        "cod_upa": pa.Column(str, pa.Check.str_length(min_value=1)),
        "estrato": pa.Column(str),
        "num_dom": pa.Column(int, pa.Check.gt(0)),
        "num_uc": pa.Column(int, pa.Check.gt(0)),
        "cod_informante": pa.Column(int, pa.Check.gt(0)),
        "id_pessoa": pa.Column(str),
    }


ESQUEMA_CONSUMO = pa.DataFrameSchema(
    {
        **_colunas_chave(),
        "quadro": pa.Column(int),
        "id_dia": pa.Column(str),
        "cod_alimento": pa.Column(int, pa.Check.in_range(1_000_000, 9_999_999)),
        "hora": pa.Column("Int64", pa.Check.in_range(0, 23), nullable=True),
        "local": pa.Column(str, pa.Check.isin(["em_casa", "fora_de_casa", "indeterminado"])),
        "dia_semana": pa.Column(int, pa.Check.in_range(1, 7)),
        "peso_final": pa.Column(float, pa.Check.gt(0)),
        "renda_total_uc": pa.Column(float, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(_quadro_conforme_edicao, name="quadro_conforme_edicao"),
        pa.Check(_estrato_montado, name="estrato_montado"),
        pa.Check(_id_pessoa_montado, name="id_pessoa_montado"),
        pa.Check(_id_dia_montado, name="id_dia_montado"),
    ],
    strict=True,
    coerce=False,
    name="consumo",
)

ESQUEMA_MORADOR = pa.DataFrameSchema(
    {
        **_colunas_chave(),
        "situacao": pa.Column(str, pa.Check.isin(["urbano", "rural"])),
        "idade": pa.Column(int, pa.Check.in_range(0, 130)),
        "sexo": pa.Column(str, pa.Check.isin(["masculino", "feminino"])),
        "n_moradores_uc": pa.Column(int, pa.Check.gt(0)),
    },
    checks=[
        pa.Check(_estrato_montado, name="estrato_montado"),
        pa.Check(_id_pessoa_montado, name="id_pessoa_montado"),
        pa.Check(_n_moradores_conforme_linhas, name="n_moradores_conforme_linhas"),
    ],
    unique=["id_pessoa"],
    strict=True,
    coerce=False,
    name="morador",
)


def validar_consumo(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o quadro de consumo contra o esquema comum; levanta `SchemaError` se falhar."""
    return ESQUEMA_CONSUMO.validate(df)


def validar_morador(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o quadro de moradores contra o esquema comum; levanta `SchemaError` se falhar."""
    return ESQUEMA_MORADOR.validate(df)
