"""Testes das funções de variável: renda em salários mínimos e região."""

import pandas as pd
import pytest

from occasion.features.variaveis import COLUNAS_CONTEXTO, regiao_da_uf, renda_pc_sm


def test_renda_pc_sm():
    r = renda_pc_sm(pd.Series([1660.0]), pd.Series([4]), "2008-2009")
    assert r.iloc[0] == pytest.approx(1.0)


def test_renda_pc_sm_usa_o_salario_minimo_de_cada_edicao():
    r = renda_pc_sm(pd.Series([954.0]), pd.Series([1]), "2017-2018")
    assert r.iloc[0] == pytest.approx(1.0)


def test_renda_pc_sm_com_edicao_sem_salario_minimo_levanta_erro_claro():
    with pytest.raises(ValueError, match="2030-2031"):
        renda_pc_sm(pd.Series([1000.0]), pd.Series([1]), "2030-2031")


def test_regiao_da_uf():
    assert [regiao_da_uf(u) for u in (11, 29, 35, 43, 53)] == [
        "norte",
        "nordeste",
        "sudeste",
        "sul",
        "centro_oeste",
    ]


@pytest.mark.parametrize("uf", [0, 10, 54, 60, 99, -1])
def test_regiao_da_uf_fora_do_dominio_levanta_erro(uf):
    with pytest.raises(ValueError):
        regiao_da_uf(uf)


def test_colunas_de_contexto_sao_as_dez_do_plano():
    assert COLUNAS_CONTEXTO == [
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
