"""Testes do calibrador de Platt sobre um modelo de parte."""

import numpy as np
import pandas as pd

from occasion.experimentos.especificacao import espaco_lightgbm, variaveis_da_parte
from occasion.models.arvores import ModeloLightGBM
from occasion.models.base import logit, sigmoide
from occasion.models.calibracao import ModeloCalibrado
from occasion.models.logistica import ModeloLogistico
from occasion.models.taxas import TabelaDeTaxas


def test_calibrado_identidade_quando_a1_b0(base_p1) -> None:
    X, y, w = base_p1
    base = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    m = ModeloCalibrado(base, 1.0, 0.0)
    np.testing.assert_allclose(m.predict_proba(X), base.predict_proba(X), atol=1e-12)
    np.testing.assert_allclose(m.base_, base.base_)
    pd.testing.assert_frame_equal(m.contribuicoes(X), base.contribuicoes(X))


def test_calibrado_mantem_a_soma_das_contribuicoes_com_base_b_e_d(base_p1) -> None:
    X, y, w = base_p1
    cfg = {**espaco_lightgbm()[0], "n_estimators": 30, "min_child_samples": 20}
    for base in (
        ModeloLogistico("ocasiao", C=1.0).fit(X, y, w),
        ModeloLightGBM("ocasiao", **cfg).fit(X, y, w),
    ):
        m = ModeloCalibrado(base, 0.7, -0.3)
        soma = m.contribuicoes(X).sum(axis=1).to_numpy() + m.base_
        np.testing.assert_allclose(soma, logit(m.predict_proba(X)), atol=1e-6)
        assert m.variaveis == base.variaveis == variaveis_da_parte("ocasiao")


def test_calibrado_aplica_a_formula(base_p1) -> None:
    X, y, w = base_p1
    base = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    m = ModeloCalibrado(base, 0.5, 0.2)
    esperado = sigmoide(0.5 * logit(base.predict_proba(X)) + 0.2)
    np.testing.assert_allclose(m.predict_proba(X), esperado, atol=1e-12)


def test_calibrado_sobre_tabela_tem_base(base_p1) -> None:
    X, y, w = base_p1
    tabela = TabelaDeTaxas(["faixa_horario", "local"], 20.0).fit(X, y, w)
    m = ModeloCalibrado(tabela, 0.9, 0.1)
    assert m.variaveis == ["faixa_horario", "local"]
    assert np.isfinite(m.base_)
    assert m.base_ == 0.9 * tabela.base_ + 0.1
    p = m.predict_proba(X)
    assert p.shape == (len(X),) and ((p > 0) & (p < 1)).all()
    assert list(m.contribuicoes(X).columns) == ["faixa_horario", "local"]


def test_calibrado_com_p_extremo_e_finito() -> None:
    X = pd.DataFrame({"local": ["em_casa"] * 3 + ["fora_de_casa"] * 3})
    for y in (np.zeros(6), np.ones(6)):
        tabela = TabelaDeTaxas(["local"], 0.0).fit(X, pd.Series(y), None)
        p_base = tabela.predict_proba(X)
        assert set(p_base) <= {0.0, 1.0}
        p = ModeloCalibrado(tabela, 1.2, 0.1).predict_proba(X)
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
