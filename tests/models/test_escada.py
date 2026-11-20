"""Testes da fábrica da escada de modelos e dos espaços de busca."""

from itertools import product

import numpy as np
import pytest
from tests.models.bases_sinteticas import base_sintetica

from occasion.experimentos.especificacao import (
    ESPACO_A,
    GRADE_C_LOGISTICA,
    INTERACOES,
    ParteModelo,
    espaco_lightgbm,
    variaveis_da_parte,
)
from occasion.models.arvores import ModeloLightGBM
from occasion.models.escada import construir, espaco
from occasion.models.logistica import ModeloLogistico
from occasion.models.taxas import TabelaDeTaxas

PARTES: list[ParteModelo] = ["ocasiao", "preferencia", "direto"]


def test_espacos_de_cada_degrau() -> None:
    for parte in PARTES:
        assert espaco("A", parte) == ESPACO_A[parte]
        assert espaco("B", parte) == [{"C": c} for c in GRADE_C_LOGISTICA]
        assert espaco("C", parte) == [
            {"C": c, "interacoes": nome}
            for nome, c in product(INTERACOES[parte], GRADE_C_LOGISTICA)
        ]
        assert espaco("D", parte) == espaco_lightgbm()
    assert len(espaco("C", "ocasiao")) == 2 * len(GRADE_C_LOGISTICA)
    assert len(espaco("C", "preferencia")) == len(GRADE_C_LOGISTICA)


def test_construir_devolve_o_tipo_certo() -> None:
    assert isinstance(construir("A", "ocasiao", ESPACO_A["ocasiao"][0]), TabelaDeTaxas)
    b = construir("B", "ocasiao", {"C": 0.5})
    assert isinstance(b, ModeloLogistico) and not b.com_curvas and b.interacoes == []
    c = construir("C", "ocasiao", {"C": 0.5, "interacoes": "fim_de_semana"})
    assert isinstance(c, ModeloLogistico) and c.com_curvas
    assert c.interacoes == INTERACOES["ocasiao"]["fim_de_semana"]
    d = construir("D", "ocasiao", espaco_lightgbm()[0])
    assert isinstance(d, ModeloLightGBM)


def test_construir_usa_a_suavizacao_como_forca() -> None:
    cfg = {"dimensoes": ["local"], "suavizacao": 7}
    t = construir("A", "ocasiao", cfg)
    assert isinstance(t, TabelaDeTaxas)
    assert t.dimensoes == ["local"] and t.forca_suavizacao == 7.0


def test_construir_degrau_desconhecido_da_erro() -> None:
    with pytest.raises(ValueError, match="degrau"):
        construir("E", "ocasiao", {})  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("parte", PARTES)
@pytest.mark.parametrize("degrau", ["A", "B", "C", "D"])
def test_construir_cobre_os_quatro_degraus_e_as_tres_partes(degrau, parte) -> None:
    X, y, w = base_sintetica(1500, semente=4, com_pizza=parte == "preferencia")
    params = espaco(degrau, parte)[0]
    if degrau == "D":
        params = {**params, "n_estimators": 10}
    m = construir(degrau, parte, params).fit(X, y, w)
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and np.isfinite(p).all()
    c = m.contribuicoes(X)
    assert list(c.columns) == m.variaveis
    if degrau != "A":
        assert m.variaveis == variaveis_da_parte(parte)
    assert np.isfinite(m.base_)
