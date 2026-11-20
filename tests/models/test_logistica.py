"""Testes dos degraus B e C (regressão logística de uma parte), com bases sintéticas."""

import warnings

import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import base_sintetica

from occasion.experimentos.especificacao import INTERACOES, graus_de_liberdade, variaveis_da_parte
from occasion.models.base import logit
from occasion.models.logistica import ModeloLogistico

PRE = INTERACOES["ocasiao"]["pre_declarado"]
PRE_FDS = INTERACOES["ocasiao"]["fim_de_semana"]


def test_contribuicoes_somam_o_logit(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0, com_curvas=True, interacoes=PRE).fit(X, y, w)
    soma = m.contribuicoes(X).sum(axis=1).to_numpy() + m.base_
    np.testing.assert_allclose(soma, logit(m.predict_proba(X)), atol=1e-8)


def test_contribuicoes_somam_o_logit_na_preferencia_com_interacoes(base_p2) -> None:
    X, y, w = base_p2
    inter = INTERACOES["preferencia"]["pre_declarado"]
    m = ModeloLogistico("preferencia", C=0.5, com_curvas=True, interacoes=inter).fit(X, y, w)
    c = m.contribuicoes(X)
    assert list(c.columns) == variaveis_da_parte("preferencia")
    np.testing.assert_allclose(
        c.sum(axis=1).to_numpy() + m.base_, logit(m.predict_proba(X)), atol=1e-8
    )


def test_interacao_com_fim_de_semana_atribui_a_dia_semana(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0, interacoes=PRE_FDS).fit(X, y, w)
    c = m.contribuicoes(X)
    assert "fim_de_semana" not in c.columns
    assert list(c.columns) == variaveis_da_parte("ocasiao")
    np.testing.assert_allclose(
        c.sum(axis=1).to_numpy() + m.base_, logit(m.predict_proba(X)), atol=1e-8
    )
    # a metade da interação fim_de_semana x faixa vai para `dia_semana`: com o mesmo dia, a
    # contribuição do dia muda de uma faixa de horário para outra
    sabado = c.loc[X["dia_semana"] == 6, "dia_semana"]
    assert sabado.groupby(X.loc[sabado.index, "faixa_horario"]).first().nunique() > 1


def test_interacao_dividida_meio_a_meio() -> None:
    X, y, w = base_sintetica(3000, semente=3)
    m = ModeloLogistico("ocasiao", C=1.0, interacoes=[("faixa_horario", "local")]).fit(X, y, w)
    c = m.contribuicoes(X)
    np.testing.assert_allclose(
        c.sum(axis=1).to_numpy() + m.base_, logit(m.predict_proba(X)), atol=1e-8
    )
    # contribuição do bloco da interação, calculada à mão a partir do que foi ajustado
    fatia = m._indices["inter:faixa_horario|local"]
    coef = m._coef[fatia]
    Xt = m._transformar(X).tocsc()
    inter = np.asarray(Xt[:, fatia] @ coef).ravel() - float(m._media[fatia] @ coef)
    assert np.abs(inter).max() > 1e-3

    def principal(v: str) -> np.ndarray:
        f = m._indices[f"cat:{v}"]
        return np.asarray(Xt[:, f] @ m._coef[f]).ravel() - float(m._media[f] @ m._coef[f])

    np.testing.assert_allclose(
        c["faixa_horario"], principal("faixa_horario") + inter / 2, atol=1e-10
    )
    np.testing.assert_allclose(c["local"], principal("local") + inter / 2, atol=1e-10)


def test_chave_da_interacao_nao_depende_do_dtype(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0, interacoes=PRE).fit(X, y, w)
    esperado = m.predict_proba(X.head(20))
    com_float = X.head(20).assign(dia_semana=X.head(20)["dia_semana"].astype(float))
    np.testing.assert_allclose(m.predict_proba(com_float), esperado, atol=1e-12)


def test_categoria_ausente_no_treino_da_previsao_finita() -> None:
    X, y, w = base_sintetica(3000, semente=4, excluir_uf=13)
    m = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    X_am = base_sintetica(50, semente=5)[0].assign(uf=13, regiao="norte")
    p = m.predict_proba(X_am)
    assert p.shape == (50,)
    assert np.isfinite(p).all()


def test_valor_fora_do_dominio_e_recusado(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    with pytest.raises(ValueError, match="fora do domínio"):
        m.predict_proba(X.head(3).assign(local="na_lua"))
    with pytest.raises(ValueError, match="fora do domínio"):
        ModeloLogistico("ocasiao", C=1.0).fit(X.assign(dia_semana=9), y, w)


def test_faltando_coluna_e_recusado(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    with pytest.raises(ValueError, match="faltam"):
        m.predict_proba(X.drop(columns=["idade"]))


def test_numerica_ausente_e_recusada(base_p1) -> None:
    X, y, w = base_p1
    m = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    with pytest.raises(ValueError, match="NaN"):
        m.predict_proba(X.head(3).assign(idade=[30.0, np.nan, 40.0]))


def test_graus_de_liberdade_batem_com_a_especificacao(base_p2) -> None:
    X, y, w = base_p2
    inter = INTERACOES["preferencia"]["pre_declarado"]
    m = ModeloLogistico("preferencia", C=1.0, com_curvas=True, interacoes=inter).fit(X, y, w)
    assert m.n_parametros == graus_de_liberdade("preferencia", "C")


def test_graus_de_liberdade_do_degrau_b(base_p1, base_p2) -> None:
    X, y, w = base_p1
    assert ModeloLogistico("ocasiao", C=1.0).fit(X, y, w).n_parametros == graus_de_liberdade(
        "ocasiao", "B"
    )
    X2, y2, w2 = base_p2
    m = ModeloLogistico("preferencia", C=1.0).fit(X2, y2, w2)
    assert m.n_parametros == graus_de_liberdade("preferencia", "B")


def test_graus_de_liberdade_da_ocasiao_no_degrau_c(base_p1) -> None:
    X, y, w = base_p1
    for nome, inter in INTERACOES["ocasiao"].items():
        m = ModeloLogistico("ocasiao", C=1.0, com_curvas=True, interacoes=inter).fit(X, y, w)
        assert m.n_parametros == graus_de_liberdade("ocasiao", "C", nome)


def test_n_parametros_nao_depende_dos_niveis_do_treino() -> None:
    X, y, w = base_sintetica(3000, semente=4, excluir_uf=13)
    m = ModeloLogistico("ocasiao", C=1.0).fit(X, y, w)
    assert m.n_parametros == graus_de_liberdade("ocasiao", "B")


def test_coeficiente_de_tem_pizza_em_log_chances(base_p2) -> None:
    X, y, w = base_p2
    m = ModeloLogistico("preferencia", C=100.0).fit(X, y, w)
    # a base sintética soma 0,8 em log-chances quando há pizza
    assert m.coeficiente("tem_pizza") == pytest.approx(0.8, abs=0.25)
    with pytest.raises(KeyError):
        m.coeficiente("nao_existe")


def test_fit_sem_peso_equivale_a_pesos_iguais(base_p1) -> None:
    X, y, _ = base_p1
    sem = ModeloLogistico("ocasiao", C=1.0).fit(X, y, None)
    iguais = ModeloLogistico("ocasiao", C=1.0).fit(X, y, pd.Series(np.full(len(y), 7.0)))
    np.testing.assert_allclose(sem.predict_proba(X), iguais.predict_proba(X), atol=1e-8)


def test_peso_muda_o_ajuste(base_p1) -> None:
    X, y, w = base_p1
    pesado = w.where(y == 1, 1.0).where(y == 0, 20.0)
    a = ModeloLogistico("ocasiao", C=1.0).fit(X, y, None)
    b = ModeloLogistico("ocasiao", C=1.0).fit(X, y, pesado)
    assert b.predict_proba(X).mean() > a.predict_proba(X).mean()


def test_regularizacao_mais_forte_encolhe_os_coeficientes(base_p1) -> None:
    X, y, w = base_p1
    forte = ModeloLogistico("ocasiao", C=0.001).fit(X, y, w)
    fraca = ModeloLogistico("ocasiao", C=100.0).fit(X, y, w)
    assert np.std(forte.predict_proba(X)) < np.std(fraca.predict_proba(X))


def test_predict_proba_e_unidimensional_e_nao_altera_a_entrada(base_p1) -> None:
    X, y, w = base_p1
    antes = X.copy()
    m = ModeloLogistico("ocasiao", C=1.0, com_curvas=True, interacoes=PRE).fit(X, y, w)
    p = m.predict_proba(X)
    assert p.ndim == 1 and len(p) == len(X)
    assert ((p > 0) & (p < 1)).all()
    pd.testing.assert_frame_equal(X, antes)


def test_ajuste_nao_emite_avisos(base_p1) -> None:
    X, y, w = base_p1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ModeloLogistico("ocasiao", C=1.0, com_curvas=True, interacoes=PRE).fit(X, y, w)


def test_interacao_com_variavel_fora_da_parte_e_recusada(base_p1) -> None:
    X, y, w = base_p1
    with pytest.raises(ValueError, match="interação"):
        ModeloLogistico("ocasiao", C=1.0, interacoes=[("tem_pizza", "local")]).fit(X, y, w)
