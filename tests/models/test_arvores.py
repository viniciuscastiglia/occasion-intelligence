"""Testes do degrau D (LightGBM), com bases sintéticas."""

import numpy as np
import pandas as pd
import pytest

from occasion.experimentos.especificacao import espaco_lightgbm, variaveis_da_parte
from occasion.models.arvores import ModeloLightGBM
from occasion.models.base import logit

CONFIG = {
    **espaco_lightgbm()[0],
    "n_estimators": 60,
    "min_child_samples": 20,
    "learning_rate": 0.1,
}


def test_contribuicoes_somam_o_logit(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w)
    c = m.contribuicoes(X)
    assert list(c.columns) == variaveis_da_parte("ocasiao")
    soma = c.sum(axis=1).to_numpy() + m.base_
    np.testing.assert_allclose(soma, logit(m.predict_proba(X)), atol=1e-6)


def test_contribuicoes_somam_o_logit_na_preferencia(base_p2) -> None:
    X, y, w = base_p2
    m = ModeloLightGBM("preferencia", **CONFIG).fit(X, y, w)
    c = m.contribuicoes(X)
    assert list(c.columns) == variaveis_da_parte("preferencia")
    np.testing.assert_allclose(
        c.sum(axis=1).to_numpy() + m.base_, logit(m.predict_proba(X)), atol=1e-6
    )
    assert float(np.abs(c["tem_pizza"].to_numpy()).max()) > 0


def test_predict_proba_e_1d_e_no_intervalo(base_p1) -> None:
    X, y, w = base_p1
    p = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w).predict_proba(X)
    assert p.shape == (len(X),)
    assert ((p > 0) & (p < 1)).all()


def test_lightgbm_reprodutivel(base_p1) -> None:
    X, y, w = base_p1
    a = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w).predict_proba(X)
    b = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w).predict_proba(X)
    np.testing.assert_array_equal(a, b)


def test_lightgbm_categoria_ausente_no_treino(base_p1) -> None:
    X, y, w = base_p1
    sem_uf_12 = X["uf"] != 12
    m = ModeloLightGBM("ocasiao", **CONFIG).fit(X[sem_uf_12], y[sem_uf_12], w[sem_uf_12])
    novo = X[X["uf"] == 12].head(10)
    assert len(novo) == 10
    p = m.predict_proba(novo)
    assert np.isfinite(p).all()
    assert np.isfinite(m.contribuicoes(novo).to_numpy()).all()


def test_valor_fora_do_dominio_e_recusado(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w)
    ruim = X.head(5).assign(local="no_carro")
    with pytest.raises(ValueError, match="local"):
        m.predict_proba(ruim)
    with pytest.raises(ValueError, match="local"):
        ModeloLightGBM("ocasiao", **CONFIG).fit(X.assign(local="no_carro"), y, w)


def test_nan_numerico_e_recusado(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w)
    ruim = X.head(5).copy()
    ruim["renda_pc_sm"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        m.predict_proba(ruim)


def test_coluna_ausente_e_recusada(base_p1) -> None:
    X, y, w = base_p1
    with pytest.raises(ValueError, match="uf"):
        ModeloLightGBM("ocasiao", **CONFIG).fit(X.drop(columns="uf"), y, w)


def test_peso_zero_e_aceito_e_pesos_invalidos_nao(base_p1) -> None:
    X, y, w = base_p1
    zero = pd.Series(np.where(np.arange(len(X)) % 3 == 0, 0.0, w))
    ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, zero)
    for ruim in (-w, w.where(w.index != 0, np.nan), w * 0.0, w.where(w.index != 0, np.inf)):
        with pytest.raises(ValueError, match="peso"):
            ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, ruim)


def test_sem_peso_e_igual_a_pesos_iguais(base_p1) -> None:
    X, y, _ = base_p1
    a = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, None).predict_proba(X)
    b = (
        ModeloLightGBM("ocasiao", **CONFIG)
        .fit(X, y, pd.Series(np.full(len(X), 3.0)))
        .predict_proba(X)
    )
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_usar_antes_de_ajustar_da_erro(base_p1) -> None:
    X, _, _ = base_p1
    with pytest.raises(RuntimeError):
        ModeloLightGBM("ocasiao", **CONFIG).predict_proba(X)


def test_ajuste_aprende_o_sinal(base_p1) -> None:
    X, y, w = base_p1
    c = ModeloLightGBM("ocasiao", **CONFIG).fit(X, y, w).contribuicoes(X)
    noite = X["faixa_horario"] == "noite"
    assert c.loc[noite, "faixa_horario"].mean() > c.loc[~noite, "faixa_horario"].mean()
