"""Testes das bases de modelagem: unitários com bases mínimas e conferência nos dados reais."""

import json
from typing import Any, cast

import pandas as pd
import pytest

from occasion.bases.modelagem import montar_celulas, montar_ocasioes_refri
from occasion.config import DADOS_PROCESSADOS, RELATORIOS
from occasion.features.variaveis import COLUNAS_CONTEXTO

EDICAO = "2008-2009"


def pessoa_dia_minimo(n_dias: int = 1, pizza_refri: bool = False) -> pd.DataFrame:
    ids = [f"{EDICAO}|111|1|1|1|{71 + i}" for i in range(n_dias)]
    return pd.DataFrame(
        {
            "id_dia": pd.array(ids, dtype="string"),
            "id_pessoa": pd.array([f"{EDICAO}|111|1|1|1"] * n_dias, dtype="string"),
            "quadro": [71 + i for i in range(n_dias)],
            "dia_semana": [3] * n_dias,
            "peso_final": [100.0] * n_dias,
            "renda_total_uc": [1660.0] * n_dias,
            "edicao": pd.array([EDICAO] * n_dias, dtype="string"),
            "uf": [35] * n_dias,
            "cod_upa": pd.array(["111"] * n_dias, dtype="string"),
            "estrato": pd.array(["35-01"] * n_dias, dtype="string"),
            "situacao": pd.array(["urbano"] * n_dias, dtype="string"),
            "idade": [30] * n_dias,
            "sexo": pd.array(["feminino"] * n_dias, dtype="string"),
            "n_moradores_uc": [4] * n_dias,
            "tem_pizza_dia": [pizza_refri] * n_dias,
            "tem_pizza_refri": [pizza_refri] * n_dias,
            "tem_pizza_guarana": [False] * n_dias,
        }
    )


def ocasioes_vazias() -> pd.DataFrame:
    return ocasiao_lista([])


def ocasiao_lista(linhas: list[dict]) -> pd.DataFrame:
    """Cada linha: hora, local, pizza, refri, guarana (padrão falso), id_dia (padrão o 1o dia)."""
    id_dia = f"{EDICAO}|111|1|1|1|71"
    registros = []
    for linha in linhas:
        hora = linha["hora"]
        dia = linha.get("id_dia", id_dia)
        registros.append(
            {
                "id_ocasiao": f"{dia}|{hora}",
                "id_dia": dia,
                "id_pessoa": f"{EDICAO}|111|1|1|1",
                "hora": hora,
                "local": linha["local"],
                "tem_pizza": linha.get("pizza", False),
                "tem_refri": linha.get("refri", False) or linha.get("guarana", False),
                "tem_guarana": linha.get("guarana", False),
                "tem_cola": False,
            }
        )
    colunas = [
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
    return pd.DataFrame(registros, columns=colunas).astype(
        {"tem_pizza": bool, "tem_refri": bool, "tem_guarana": bool, "tem_cola": bool}
    )


def test_cada_dia_tem_dez_celulas():
    celulas = montar_celulas(pessoa_dia_minimo(n_dias=3), ocasioes_vazias())
    assert len(celulas) == 3 * 10
    assert bool(celulas.groupby("id_dia").size().eq(10).all())
    assert celulas["id_celula"].is_unique


def test_celulas_trazem_contexto_e_identificacao():
    celulas = montar_celulas(pessoa_dia_minimo(), ocasioes_vazias())
    for coluna in [*COLUNAS_CONTEXTO, "id_celula", "id_dia", "id_pessoa", "edicao", "cod_upa"]:
        assert coluna in celulas.columns
    assert set(celulas["regiao"]) == {"sudeste"}
    assert celulas["renda_pc_sm"].iloc[0] == pytest.approx(1.0)
    assert celulas["id_celula"].iloc[0].startswith(celulas["id_dia"].iloc[0] + "|")


def test_ocasiao_positiva_marca_uma_unica_celula():
    ocasioes = ocasiao_lista([{"hora": 20, "local": "em_casa", "pizza": True, "refri": True}])
    celulas = montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)
    positivas = celulas[celulas.y_pizza_refri == 1]
    assert cast(Any, positivas[["faixa_horario", "local"]]).values.tolist() == [
        ["noite", "em_casa"]
    ]
    assert celulas.y_pizza_guarana.sum() == 0
    assert celulas.attrs["dias_positivos_excluidos"] == 0


def test_pizza_com_guarana_marca_os_dois_alvos():
    ocasioes = ocasiao_lista(
        [{"hora": 12, "local": "fora_de_casa", "pizza": True, "guarana": True}]
    )
    celulas = montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)
    marcadas = celulas[(celulas.y_pizza_refri == 1) & (celulas.y_pizza_guarana == 1)]
    assert cast(Any, marcadas[["faixa_horario", "local"]]).values.tolist() == [
        ["almoco", "fora_de_casa"]
    ]


def test_duas_ocasioes_positivas_na_mesma_celula_nao_somam():
    ocasioes = ocasiao_lista(
        [
            {"hora": 19, "local": "em_casa", "pizza": True, "refri": True},
            {"hora": 21, "local": "em_casa", "pizza": True, "refri": True},
        ]
    )
    celulas = montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)
    assert celulas.y_pizza_refri.max() == 1
    assert celulas.y_pizza_refri.sum() == 1


def test_pizza_sem_refrigerante_nao_e_positiva():
    ocasioes = ocasiao_lista([{"hora": 20, "local": "em_casa", "pizza": True}])
    celulas = montar_celulas(pessoa_dia_minimo(), ocasioes)
    assert celulas.y_pizza_refri.sum() == 0


def test_ocasiao_positiva_com_local_indeterminado_e_contada_e_nao_some():
    ocasioes = ocasiao_lista([{"hora": 20, "local": "indeterminado", "pizza": True, "refri": True}])
    celulas = montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)
    assert len(celulas) == 10
    assert celulas.y_pizza_refri.sum() == 0
    assert celulas.attrs["dias_positivos_excluidos"] == 1


def test_dia_com_positiva_determinada_e_indeterminada_nao_e_excluido():
    ocasioes = ocasiao_lista(
        [
            {"hora": 20, "local": "em_casa", "pizza": True, "refri": True},
            {"hora": 13, "local": "indeterminado", "pizza": True, "refri": True},
        ]
    )
    celulas = montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)
    assert celulas.y_pizza_refri.sum() == 1
    assert celulas.attrs["dias_positivos_excluidos"] == 0


def test_hora_fora_do_intervalo_levanta_erro():
    ocasioes = ocasiao_lista([{"hora": 24, "local": "em_casa", "pizza": True, "refri": True}])
    with pytest.raises(ValueError):
        montar_celulas(pessoa_dia_minimo(pizza_refri=True), ocasioes)


def test_ocasioes_refri_filtra_refrigerante_e_local_determinado():
    ocasioes = ocasiao_lista(
        [
            {"hora": 20, "local": "em_casa", "pizza": True, "guarana": True},
            {"hora": 12, "local": "fora_de_casa", "refri": True},
            {"hora": 8, "local": "em_casa"},  # sem refrigerante
            {"hora": 16, "local": "indeterminado", "refri": True},  # local indeterminado
        ]
    )
    base = montar_ocasioes_refri(pessoa_dia_minimo(), ocasioes)
    assert len(base) == 2
    assert base.attrs["ocasioes_excluidas"] == 1
    por_hora = base.assign(hora=base["id_ocasiao"].str.rsplit("|").str[-1]).set_index("hora")
    assert por_hora.loc["20", ["tem_pizza", "y_guarana"]].tolist() == [1, 1]
    assert por_hora.loc["12", ["tem_pizza", "y_guarana"]].tolist() == [0, 0]
    assert por_hora.loc["20", "faixa_horario"] == "noite"
    for coluna in [*COLUNAS_CONTEXTO, "id_ocasiao", "id_dia", "id_pessoa", "edicao", "peso_final"]:
        assert coluna in base.columns


def lido(edicao: str) -> dict:
    return json.loads((RELATORIOS / "bases.json").read_text(encoding="utf-8"))[edicao]


def ler_excluidos(edicao: str) -> int:
    return lido(edicao)["dias_positivos_excluidos"]


@pytest.mark.dados
@pytest.mark.parametrize(
    "edicao,dias,positivos", [("2008-2009", 66_903, 651), ("2017-2018", 85_018, 797)]
)
def test_celulas_fecham_com_a_referencia(edicao, dias, positivos):
    c = pd.read_parquet(DADOS_PROCESSADOS / f"celulas_{edicao}.parquet")
    assert len(c) == dias * 10
    dias_pos = c.loc[c.y_pizza_refri == 1, "id_dia"].nunique()
    assert dias_pos + ler_excluidos(edicao) == positivos


@pytest.mark.dados
@pytest.mark.parametrize("edicao", ["2008-2009", "2017-2018"])
def test_ocasioes_refri_fecham_com_a_base_de_ocasioes(edicao):
    base = pd.read_parquet(DADOS_PROCESSADOS / f"ocasioes_refri_{edicao}.parquet")
    oc = pd.read_parquet(DADOS_PROCESSADOS / f"ocasioes_{edicao}.parquet")
    if edicao == "2008-2009":  # nenhuma contagem do alvo de 2017-2018 além da referência
        assert len(base) + lido(edicao)["ocasioes_excluidas"] == int(
            cast(Any, oc["tem_refri"]).sum()
        )
    assert base["id_ocasiao"].is_unique
    assert not bool(cast(Any, base[COLUNAS_CONTEXTO].isna()).to_numpy().any())
