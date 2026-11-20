"""Testes do pré-registro: espaços de busca, interações, graus de liberdade e domínios."""

import json
import math
from itertools import product

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import SplineTransformer

from occasion.audit.auditoria import tamanho_de_amostra
from occasion.bases.acesso import carregar_base
from occasion.bases.modelagem import montar_celulas, montar_ocasioes_refri
from occasion.config import EDICOES, RELATORIOS, SEMENTE
from occasion.experimentos.especificacao import (
    ESPACO_A,
    EVENTOS_DIRETO_2008,
    GRADE_C_LOGISTICA,
    INTERACOES,
    LIMITE_RILEY_DIRETO,
    N_CELULAS_2008,
    N_CONFIGS_LIGHTGBM,
    SPLINE,
    ParteModelo,
    espaco_lightgbm,
    graus_de_liberdade,
    parametros_de_pizza,
    variaveis_da_parte,
)
from occasion.features.variaveis import (
    COLUNAS_CONTEXTO,
    DOMINIOS,
    adicionar_derivadas,
    checar_dominios,
)
from occasion.inference.contract import CODIGO_UF

PARTES: tuple[ParteModelo, ...] = ("ocasiao", "preferencia", "direto")


# ---------------------------------------------------------------- variáveis e grade da logística


def test_variaveis_de_cada_parte():
    assert variaveis_da_parte("ocasiao") == COLUNAS_CONTEXTO
    assert variaveis_da_parte("direto") == COLUNAS_CONTEXTO
    assert variaveis_da_parte("preferencia") == [*COLUNAS_CONTEXTO, "tem_pizza"]
    assert variaveis_da_parte("ocasiao") is not COLUNAS_CONTEXTO


def test_grade_c_tem_8_valores_crescentes():
    assert len(GRADE_C_LOGISTICA) == 8
    assert GRADE_C_LOGISTICA == sorted(GRADE_C_LOGISTICA)
    np.testing.assert_allclose(GRADE_C_LOGISTICA, np.logspace(-3, 2, 8))
    assert all(type(c) is float for c in GRADE_C_LOGISTICA)


def test_curva_tem_5_colunas_iguais_aos_graus_de_liberdade():
    assert SPLINE == {"n_knots": 4, "degree": 3, "knots": "quantile", "include_bias": False}
    x = np.linspace(10, 90, 200).reshape(-1, 1)
    colunas = SplineTransformer(**SPLINE).fit_transform(x).shape[1]
    assert colunas == SPLINE["n_knots"] + SPLINE["degree"] - 2 == 5


# ---------------------------------------------------------------- interações


def test_interacoes_pre_declaradas():
    assert INTERACOES["ocasiao"] == {
        "pre_declarado": [
            ("dia_semana", "faixa_horario"),
            ("faixa_horario", "local"),
            ("local", "situacao"),
        ],
        "fim_de_semana": [
            ("fim_de_semana", "faixa_horario"),
            ("faixa_horario", "local"),
            ("local", "situacao"),
        ],
    }
    assert INTERACOES["direto"] == INTERACOES["ocasiao"]
    assert INTERACOES["preferencia"] == {
        "pre_declarado": [
            ("tem_pizza", "local"),
            ("tem_pizza", "regiao"),
            ("regiao", "local"),
            ("faixa_horario", "local"),
        ]
    }


def test_interacoes_so_usam_variaveis_da_parte_ou_derivadas_com_dominio():
    for parte in PARTES:
        permitidas = {*variaveis_da_parte(parte), "fim_de_semana"}
        for conjunto in INTERACOES[parte].values():
            for a, b in conjunto:
                assert a in permitidas and b in permitidas
                assert a in DOMINIOS and b in DOMINIOS


# ---------------------------------------------------------------- graus de liberdade


def test_graus_de_liberdade_da_ocasiao():
    assert graus_de_liberdade("ocasiao", "B") == 46
    assert graus_de_liberdade("ocasiao", "C") == 83
    assert graus_de_liberdade("ocasiao", "C", "pre_declarado") == 83
    assert graus_de_liberdade("ocasiao", "C", "fim_de_semana") == 63


def test_graus_de_liberdade_da_preferencia_e_do_direto():
    assert graus_de_liberdade("preferencia", "B") == 47
    assert graus_de_liberdade("preferencia", "C") == 47 - 2 + 10 + 13
    for conjunto in INTERACOES["direto"]:
        assert graus_de_liberdade("direto", "C", conjunto) == graus_de_liberdade(
            "ocasiao", "C", conjunto
        )


def test_graus_de_liberdade_recusa_conjunto_desconhecido():
    with pytest.raises(ValueError, match="fim_de_semana"):
        graus_de_liberdade("preferencia", "C", "fim_de_semana")


def test_parametros_de_pizza():
    assert parametros_de_pizza() == 1 + 1 + 4
    assert parametros_de_pizza("ocasiao") == 0


def test_degrau_c_cabe_no_tamanho_de_amostra():
    riley = json.loads((RELATORIOS / "auditoria.json").read_text())["tamanho_de_amostra"]
    for conjunto in INTERACOES["ocasiao"]:
        assert (
            graus_de_liberdade("ocasiao", "C", conjunto)
            <= riley["parte1"]["parametros_sustentados"]
        )
    assert graus_de_liberdade("preferencia", "C") <= riley["parte2"]["parametros_sustentados"]
    assert parametros_de_pizza() <= riley["parte2_recorte_pizza"]["parametros_sustentados"]


def test_limite_riley_do_direto_sai_do_calculo_da_auditoria():
    # Não exige que o direto caiba no limite: ele não cabe, e a decisão está no pré-registro.
    conta = tamanho_de_amostra(N_CELULAS_2008, EVENTOS_DIRETO_2008)
    assert conta["parametros_sustentados"] == LIMITE_RILEY_DIRETO


def test_celulas_do_direto_sao_as_da_parte_1_da_auditoria():
    riley = json.loads((RELATORIOS / "auditoria.json").read_text())["tamanho_de_amostra"]
    assert riley["parte1"]["n"] == N_CELULAS_2008


@pytest.mark.dados
def test_contagens_do_direto_batem_com_as_celulas_de_2008():
    celulas = carregar_base("celulas", "2008-2009")
    assert len(celulas) == N_CELULAS_2008
    assert int(celulas["y_pizza_guarana"].to_numpy().sum()) == EVENTOS_DIRETO_2008


# ---------------------------------------------------------------- LightGBM


def test_espaco_lightgbm_tem_30_configuracoes_reprodutiveis():
    assert N_CONFIGS_LIGHTGBM == 30
    assert len(espaco_lightgbm()) == 30 and espaco_lightgbm() == espaco_lightgbm()
    assert espaco_lightgbm(SEMENTE + 1) != espaco_lightgbm()


def test_espaco_lightgbm_respeita_os_limites_e_os_fixos():
    for c in espaco_lightgbm():
        assert 0.01 <= c["learning_rate"] <= 0.2
        assert 100 <= c["n_estimators"] <= 800
        assert c["max_depth"] in {2, 3, 4} and c["num_leaves"] == 2 ** c["max_depth"]
        assert c["min_child_samples"] in {200, 500, 1000, 2000}
        assert 0.01 <= c["reg_lambda"] <= 10
        assert c["subsample"] in {0.7, 1.0} and c["subsample_freq"] == 1
        assert c["colsample_bytree"] in {0.7, 1.0}
        assert {
            k: c[k]
            for k in ("objective", "deterministic", "force_row_wise", "n_jobs", "random_state")
        } == {
            "objective": "binary",
            "deterministic": True,
            "force_row_wise": True,
            "n_jobs": 4,
            "random_state": SEMENTE,
        }
        assert c["verbose"] == -1


def test_espaco_lightgbm_ordenado_do_mais_simples_e_serializavel():
    espaco = espaco_lightgbm()
    chaves = [(c["max_depth"], c["n_estimators"]) for c in espaco]
    assert chaves == sorted(chaves)
    json.dumps(espaco)  # tipos do Python, não do NumPy
    assert all(type(c["n_estimators"]) is int and type(c["learning_rate"]) is float for c in espaco)


# ---------------------------------------------------------------- degrau A


def _combinacoes(dimensoes: list[str]) -> int:
    return math.prod(len(DOMINIOS[d]) for d in dimensoes)


def test_espaco_a_tem_12_configuracoes_por_parte():
    dims_ocasiao = [
        ["dia_semana", "faixa_horario", "local"],
        ["fim_de_semana", "faixa_horario", "local"],
        ["faixa_horario", "local"],
        ["dia_semana", "faixa_horario"],
    ]
    dims_preferencia = [["regiao", "local"], ["regiao"], ["uf"], ["uf", "local"]]
    casos: list[tuple[ParteModelo, list[list[str]]]] = [
        ("ocasiao", dims_ocasiao),
        ("direto", dims_ocasiao),
        ("preferencia", dims_preferencia),
    ]
    for parte, dims in casos:
        espaco = ESPACO_A[parte]
        assert len(espaco) == 12
        esperado = {(tuple(d), m) for d, m in product(dims, (10, 100, 1000))}
        assert {(tuple(c["dimensoes"]), c["suavizacao"]) for c in espaco} == esperado


def test_espaco_a_ordenado_do_mais_simples_para_o_mais_complexo():
    for parte in PARTES:
        chaves = [(_combinacoes(c["dimensoes"]), -c["suavizacao"]) for c in ESPACO_A[parte]]
        assert chaves == sorted(chaves)
    assert ESPACO_A["ocasiao"][0] == {"dimensoes": ["faixa_horario", "local"], "suavizacao": 1000}
    assert ESPACO_A["preferencia"][0] == {"dimensoes": ["regiao"], "suavizacao": 1000}


def test_espaco_a_so_usa_dimensoes_com_dominio():
    for parte in PARTES:
        for c in ESPACO_A[parte]:
            assert all(d in DOMINIOS for d in c["dimensoes"])


# ---------------------------------------------------------------- domínios nas bases


def _pessoa_dia_sintetica(edicao: str) -> pd.DataFrame:
    ufs = list(CODIGO_UF.values())
    linhas = []
    for i, (uf, dia) in enumerate(product(ufs, range(1, 8))):
        id_pessoa = f"{edicao}|{i}|1|1|1"
        linhas.append(
            {
                "id_dia": f"{id_pessoa}|71",
                "id_pessoa": id_pessoa,
                "quadro": 71,
                "dia_semana": dia,
                "peso_final": 100.0 + i,
                "renda_total_uc": 500.0 * (1 + i % 9),
                "edicao": edicao,
                "uf": uf,
                "cod_upa": str(i),
                "estrato": f"{uf}-01",
                "situacao": ("urbano", "rural")[i % 2],
                "idade": 10 + i % 80,
                "sexo": ("masculino", "feminino")[i % 2],
                "n_moradores_uc": 1 + i % 6,
                "tem_pizza_dia": False,
                "tem_pizza_refri": False,
                "tem_pizza_guarana": False,
            }
        )
    df = pd.DataFrame(linhas)
    for c in ("id_dia", "id_pessoa", "edicao", "cod_upa", "estrato", "situacao", "sexo"):
        df[c] = df[c].astype("string")
    return df


def _ocasioes_sinteticas(pessoa_dia: pd.DataFrame) -> pd.DataFrame:
    locais = ("em_casa", "fora_de_casa", "indeterminado")
    registros = []
    for j, (id_dia, id_pessoa) in enumerate(zip(pessoa_dia["id_dia"], pessoa_dia["id_pessoa"])):
        hora = j % 24
        registros.append(
            {
                "id_ocasiao": f"{id_dia}|{hora}",
                "id_dia": id_dia,
                "id_pessoa": id_pessoa,
                "hora": hora,
                "local": locais[j % 3],
                "tem_pizza": j % 4 == 0,
                "tem_refri": True,
                "tem_guarana": j % 3 == 0,
                "tem_cola": False,
            }
        )
    return pd.DataFrame(registros)


@pytest.mark.parametrize("edicao", EDICOES)
def test_dominios_cobrem_as_bases_sinteticas(edicao):
    pessoa_dia = _pessoa_dia_sintetica(edicao)
    ocasioes = _ocasioes_sinteticas(pessoa_dia)
    for base in (
        montar_celulas(pessoa_dia, ocasioes),
        montar_ocasioes_refri(pessoa_dia, ocasioes),
    ):
        com_derivadas = adicionar_derivadas(base)
        checar_dominios(com_derivadas)
        for coluna, dominio in DOMINIOS.items():
            if coluna in com_derivadas.columns:
                assert set(com_derivadas[coluna]) <= set(dominio), coluna


def test_checar_dominios_recusa_valor_fora():
    pessoa_dia = _pessoa_dia_sintetica("2008-2009")
    base = montar_celulas(pessoa_dia, _ocasioes_sinteticas(pessoa_dia))
    checar_dominios(base)
    base.loc[3, "uf"] = 99
    with pytest.raises(ValueError, match=r"uf.*99"):
        checar_dominios(base)


@pytest.mark.dados
def test_dominios_cobrem_as_duas_edicoes():
    for edicao in EDICOES:
        for nome in ("celulas", "ocasioes_refri"):
            base = carregar_base(nome, edicao, so_contexto=True)
            checar_dominios(adicionar_derivadas(base))
