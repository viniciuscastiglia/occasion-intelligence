import warnings

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from occasion.config import SEMENTE
from occasion.models.base import logit
from occasion.score.score import CORTE_ALTA
from occasion.validacao.metricas import (
    ajustar_sigmoide,
    brier_ponderado,
    calibracao,
    concentracao_topo,
    log_loss_ponderado,
    metricas_no_corte,
    posicoes_ponderadas,
    pr_auc_ponderada,
    resumo,
    roc_auc_ponderada,
)


@pytest.fixture
def dados() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEMENTE)
    p = rng.uniform(0.02, 0.6, 500)
    y = rng.binomial(1, p)
    w = rng.uniform(0.5, 3.0, 500)
    return y, p, w


def test_log_loss_igual_ao_sklearn(dados):
    y, p, w = dados
    assert log_loss_ponderado(y, p, w) == pytest.approx(log_loss(y, p, sample_weight=w))


def test_brier_e_aucs_iguais_ao_sklearn(dados):
    y, p, w = dados
    assert brier_ponderado(y, p, w) == pytest.approx(brier_score_loss(y, p, sample_weight=w))
    assert pr_auc_ponderada(y, p, w) == pytest.approx(
        average_precision_score(y, p, sample_weight=w)
    )
    assert roc_auc_ponderada(y, p, w) == pytest.approx(roc_auc_score(y, p, sample_weight=w))


def test_probabilidade_zero_ou_um_nao_da_infinito():
    y, p, w = np.array([1, 0]), np.array([0.0, 1.0]), np.ones(2)
    assert np.isfinite(log_loss_ponderado(y, p, w))
    assert np.isfinite(calibracao(y, p, w)["inclinacao"])


def test_sigmoide_recupera_parametros():
    rng = np.random.default_rng(SEMENTE)
    p = rng.uniform(0.02, 0.6, 200_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-(2 * logit(p) - 1))))
    a, b = ajustar_sigmoide(p, y, np.ones_like(p))
    assert a == pytest.approx(2, abs=0.1) and b == pytest.approx(-1, abs=0.1)


def test_calibracao_de_modelo_calibrado_da_o_e_proximo_de_um():
    rng = np.random.default_rng(SEMENTE)
    p = rng.uniform(0.02, 0.6, 100_000)
    y = rng.binomial(1, p)
    w = np.ones_like(p)
    c = calibracao(y, p, w)
    assert c["o_e"] == pytest.approx(1.0, abs=0.05)
    assert c["inclinacao"] == pytest.approx(1.0, abs=0.1)
    assert c["intercepto_cil"] == pytest.approx(0.0, abs=0.1)


def test_calibracao_curva_agrupa_valores_iguais_e_omite_faixas_vazias():
    p = np.repeat([0.1, 0.2, 0.3], [50, 30, 20])
    y = np.zeros(100, dtype=int)
    y[:5] = 1
    c = calibracao(y, p, np.ones(100), n_faixas=10)
    curva = c["curva"]
    assert 1 <= len(curva) <= 3
    assert sum(f["n"] for f in curva) == 100
    assert sum(f["positivos"] for f in curva) == 5
    # cada valor de p cai inteiro em uma faixa: n de cada faixa é soma de grupos de 50, 30, 20
    assert all(f["n"] in (20, 30, 50, 50, 80, 100) for f in curva)
    medias = [f["p_media"] for f in curva]
    assert medias == sorted(medias)


def test_calibracao_curva_usa_pesos():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([1, 0, 1, 1])
    w = np.array([3.0, 1.0, 1.0, 1.0])
    curva = calibracao(y, p, w, n_faixas=2)["curva"]
    assert len(curva) == 2
    assert curva[0]["taxa_observada"] == pytest.approx(0.75)
    assert curva[0]["p_media"] == pytest.approx(0.1)
    assert curva[0]["n"] == 2 and curva[0]["positivos"] == 1
    assert curva[1]["taxa_observada"] == pytest.approx(1.0)


def test_auc_com_uma_classe_vira_none():
    assert pr_auc_ponderada(np.zeros(5), np.linspace(0, 1, 5), np.ones(5)) is None
    assert roc_auc_ponderada(np.ones(5), np.linspace(0, 1, 5), np.ones(5)) is None


def test_posicoes_batem_com_a_referencia_do_score():
    rng = np.random.default_rng(SEMENTE)
    v = np.round(rng.uniform(0, 1, 200), 1)  # muitos empates
    w = rng.uniform(0.5, 3.0, 200)
    obtido = posicoes_ponderadas(v, w)
    ordem = np.argsort(v, kind="stable")
    ordenados = v[ordem]
    acumulado = np.cumsum(w[ordem]) / w.sum()
    esperado = np.empty(200)
    for i, valor in enumerate(v):
        k = int(np.searchsorted(ordenados, valor, side="right"))
        esperado[i] = acumulado[k - 1] * 100.0
    assert obtido == pytest.approx(esperado)
    assert obtido.max() == pytest.approx(100.0)


def test_posicoes_com_a_referencia_real():
    from occasion.score.score import ReferenciaScore

    v = np.array([0.1, 0.2, 0.2, 0.5])
    w = np.array([1.0, 2.0, 1.0, 4.0])
    ordem = np.argsort(v, kind="stable")
    ac = np.cumsum(w[ordem]) / w.sum()
    ref = ReferenciaScore(
        ocasiao=v, preferencia=v, produto=v[ordem], peso_acumulado={"produto": ac}
    )
    obtido = posicoes_ponderadas(v, w)
    assert obtido == pytest.approx([ref.posicao(x, "produto") for x in v])


def test_concentracao_topo_quando_positivos_estao_no_topo():
    y = np.array([0] * 90 + [1] * 10)
    s = np.arange(100.0)
    assert concentracao_topo(y, s, np.ones(100)) == pytest.approx(1.0)


def test_concentracao_sem_positivos_da_none():
    assert concentracao_topo(np.zeros(10), np.arange(10.0), np.ones(10)) is None


def test_concentracao_com_empate_na_fronteira_entra_em_proporcao():
    # 20 linhas, 10% = peso 2. Score 1 empata em 4 linhas (peso 4) e atravessa a fronteira:
    # entra metade (2/4) dos positivos do grupo. Positivos: 2 no grupo, 2 fora (total 4).
    s = np.array([1.0] * 4 + [0.0] * 16)
    y = np.array([1, 1, 0, 0] + [1, 1] + [0] * 14)
    assert concentracao_topo(y, s, np.ones(20)) == pytest.approx(0.5 * 2 / 4)


def test_corte_sem_positivos_previstos_da_precision_none():
    y = np.array([1, 0, 1, 0])
    alta = np.zeros(4, dtype=bool)
    r = metricas_no_corte(y, alta, np.ones(4))
    assert r["precision"] is None
    assert r["recall"] == 0.0
    assert r["f1"] == 0.0
    assert r["matriz"] == {"vp": 0.0, "fp": 0.0, "fn": 2.0, "vn": 2.0}


def test_corte_ponderado():
    y = np.array([1, 1, 0, 0])
    alta = np.array([True, False, True, False])
    w = np.array([2.0, 1.0, 1.0, 4.0])
    r = metricas_no_corte(y, alta, w)
    assert r["matriz"] == {"vp": 2.0, "fp": 1.0, "fn": 1.0, "vn": 4.0}
    assert r["precision"] == pytest.approx(2 / 3)
    assert r["recall"] == pytest.approx(2 / 3)
    assert r["f1"] == pytest.approx(2 / 3)


def test_corte_sem_positivos_reais_da_recall_none():
    r = metricas_no_corte(np.zeros(3), np.array([True, False, False]), np.ones(3))
    assert r["recall"] is None
    assert r["f1"] == 0.0  # 2·0 / (2·0 + fp + 0): definido, só o recall não é
    r0 = metricas_no_corte(np.zeros(3), np.zeros(3, dtype=bool), np.ones(3))
    assert r0["f1"] is None


def test_resumo_sem_e_com_posicao(dados):
    y, p, w = dados
    r = resumo(y, p, w)
    assert set(r) == {
        "log_loss", "brier", "pr_auc", "roc_auc", "calibracao", "prevalencia", "n", "positivos",
        "concentracao_topo",
    }  # fmt: skip
    assert r["n"] == 500 and r["positivos"] == int(y.sum())
    assert r["prevalencia"] == pytest.approx(np.sum(w * y) / np.sum(w))
    assert r["concentracao_topo"] == pytest.approx(concentracao_topo(y, p, w))
    pos = posicoes_ponderadas(p, w)
    r2 = resumo(y, p, w, posicao=pos)
    assert r2["corte_alta"] == metricas_no_corte(y, pos >= CORTE_ALTA, w)
    assert r2["concentracao_topo"] == pytest.approx(concentracao_topo(y, pos, w))
    assert type(r2["log_loss"]) is float and type(r2["n"]) is int


def test_resumo_vai_para_json(dados):
    import json

    y, p, w = dados
    json.dumps(resumo(y, p, w, posicao=posicoes_ponderadas(p, w)))


def _pesos_com_zeros(w: np.ndarray) -> np.ndarray:
    w = w.copy()
    w[::3] = 0.0
    w[[0, 1, 2, 3]] = 0.0
    return w


def test_concentracao_topo_aceita_grupo_de_peso_zero():
    y = np.array([1, 0, 1, 0, 1])
    score = np.array([0.9, 0.9, 0.5, 0.5, 0.1])
    w = np.array([0.0, 0.0, 1.0, 1.0, 2.0])  # o grupo de maior score tem peso zero
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        topo = concentracao_topo(y, score, w, fracao=0.25)
    assert topo is not None and np.isfinite(topo)
    # orçamento 1,0 de 4,0: metade do grupo 0,5, que tem 1 de 2 positivos ponderados (1 de 3)
    assert topo == pytest.approx(0.5 * 1.0 / 3.0)


def test_resumo_aceita_pesos_zero_sem_aviso(dados):
    y, p, w = dados
    w = _pesos_com_zeros(w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        saida = resumo(y, p, w, posicao=posicoes_ponderadas(p, w))
    for chave in ("log_loss", "brier", "pr_auc", "roc_auc", "prevalencia", "concentracao_topo"):
        assert saida[chave] is None or np.isfinite(saida[chave])
    for faixa in saida["calibracao"]["curva"]:
        assert np.isfinite(faixa["p_media"]) and np.isfinite(faixa["taxa_observada"])


def test_auc_sem_positivo_com_peso_devolve_none():
    y = np.array([1, 0, 0, 1])
    p = np.array([0.8, 0.2, 0.3, 0.6])
    w = np.array([0.0, 1.0, 1.0, 0.0])  # os positivos têm peso zero
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pr_auc_ponderada(y, p, w) is None
        assert roc_auc_ponderada(y, p, w) is None
