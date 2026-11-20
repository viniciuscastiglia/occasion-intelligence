"""Testes das funções de variável: renda, região, domínios fechados e variáveis derivadas."""

import pandas as pd
import pytest

from occasion.config import FAIXAS_HORARIO, LOCAIS
from occasion.features.variaveis import (
    CATEGORICAS,
    COLUNAS_CONTEXTO,
    DOMINIOS,
    NUMERICAS,
    adicionar_derivadas,
    checar_dominios,
    regiao_da_uf,
    renda_pc_sm,
)
from occasion.inference.contract import CODIGO_UF


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


def test_categoricas_e_numericas_cobrem_o_contexto():
    assert CATEGORICAS == [
        "dia_semana",
        "faixa_horario",
        "local",
        "uf",
        "regiao",
        "situacao",
        "sexo",
    ]
    assert NUMERICAS == ["idade", "renda_pc_sm", "n_moradores_uc"]
    assert sorted(CATEGORICAS + NUMERICAS) == sorted(COLUNAS_CONTEXTO)


def test_dominios_vem_das_constantes():
    assert DOMINIOS["dia_semana"] == [1, 2, 3, 4, 5, 6, 7]
    assert DOMINIOS["faixa_horario"] == list(FAIXAS_HORARIO)
    assert DOMINIOS["local"] == list(LOCAIS)
    assert DOMINIOS["uf"] == list(CODIGO_UF.values())
    assert DOMINIOS["regiao"] == ["norte", "nordeste", "sudeste", "sul", "centro_oeste"]
    assert DOMINIOS["situacao"] == ["urbano", "rural"]
    assert DOMINIOS["sexo"] == ["masculino", "feminino"]
    assert DOMINIOS["tem_pizza"] == [0, 1]
    assert DOMINIOS["fim_de_semana"] == [0, 1]
    assert set(CATEGORICAS) <= set(DOMINIOS)


def test_toda_uf_do_dominio_tem_regiao_do_dominio():
    assert {regiao_da_uf(uf) for uf in DOMINIOS["uf"]} == set(DOMINIOS["regiao"])


def test_adicionar_derivadas_marca_sexta_a_domingo_e_nao_altera_a_entrada():
    X = pd.DataFrame({"dia_semana": [1, 2, 3, 4, 5, 6, 7]})
    com = adicionar_derivadas(X)
    assert com["fim_de_semana"].tolist() == [0, 0, 0, 0, 1, 1, 1]
    assert "fim_de_semana" not in X.columns
    assert com is not X


def test_adicionar_derivadas_sem_dia_devolve_copia_sem_a_derivada():
    X = pd.DataFrame({"local": ["em_casa"]})
    com = adicionar_derivadas(X)
    assert list(com.columns) == ["local"] and com is not X


def test_checar_dominios_aceita_valores_validos_e_ignora_colunas_ausentes():
    X = pd.DataFrame(
        {
            "dia_semana": [1, 7],
            "faixa_horario": ["noite", "madrugada"],
            "local": pd.array(["em_casa", "fora_de_casa"], dtype="string"),
            "uf": [11, 53],
            "idade": [10, 200],  # numérica: fora da conferência
        }
    )
    checar_dominios(X)
    checar_dominios(adicionar_derivadas(X))


@pytest.mark.parametrize(
    ("coluna", "valor"),
    [
        ("dia_semana", 0),
        ("dia_semana", 8),
        ("faixa_horario", "Noite"),
        ("local", "indeterminado"),
        ("uf", 10),
        ("regiao", "Sudeste"),
        ("situacao", "periurbano"),
        ("sexo", "outro"),
        ("tem_pizza", 2),
        ("fim_de_semana", -1),
    ],
)
def test_checar_dominios_recusa_valor_fora_nomeando_coluna_e_valor(coluna, valor):
    X = pd.DataFrame({coluna: [DOMINIOS[coluna][0], valor]})
    with pytest.raises(ValueError, match=coluna) as erro:
        checar_dominios(X)
    assert repr(valor) in str(erro.value)


def test_checar_dominios_recusa_valor_ausente():
    with pytest.raises(ValueError, match="local"):
        checar_dominios(pd.DataFrame({"local": ["em_casa", None]}))
