import numpy as np
import pandas as pd
import pytest

from occasion.avaliacao.segmentos import (
    MINIMO_POSITIVOS,
    colunas_de_segmento,
    metricas_por_segmento,
)
from occasion.validacao.metricas import log_loss_ponderado, pr_auc_ponderada


def test_segmento_sem_positivos_sai_none_sem_excecao():
    linhas = metricas_por_segmento(
        np.zeros(20), np.full(20, 0.1), np.ones(20), {"local": pd.Series(["em_casa"] * 20)}
    )
    assert linhas[0]["pr_auc"] is None and linhas[0]["motivo"] == "positivos insuficientes"
    assert linhas[0]["log_loss"] is None and linhas[0]["o_e"] is None
    assert linhas[0]["n"] == 20 and linhas[0]["positivos"] == 0


def test_segmento_com_positivos_calcula_as_metricas_ponderadas():
    rng = np.random.default_rng(0)
    n = 400
    p = rng.uniform(0.05, 0.5, n)
    y = (rng.random(n) < p).astype(float)
    w = rng.uniform(0.5, 3.0, n)
    local = pd.Series(np.where(np.arange(n) % 2 == 0, "em_casa", "fora_de_casa"))
    linhas = metricas_por_segmento(y, p, w, {"local": local})
    assert [l["valor"] for l in linhas] == ["em_casa", "fora_de_casa"]
    em_casa = local.to_numpy() == "em_casa"
    l0 = linhas[0]
    assert l0["segmento"] == "local" and l0["motivo"] is None
    assert l0["n"] == em_casa.sum() and l0["positivos"] == int(y[em_casa].sum())
    assert l0["log_loss"] == pytest.approx(log_loss_ponderado(y[em_casa], p[em_casa], w[em_casa]))
    assert l0["pr_auc"] == pytest.approx(pr_auc_ponderada(y[em_casa], p[em_casa], w[em_casa]))
    assert l0["o_e"] == pytest.approx(
        np.sum(w[em_casa] * y[em_casa]) / np.sum(w[em_casa] * p[em_casa])
    )


def test_limite_de_positivos_e_inclusivo():
    n = 50
    y = np.zeros(n)
    y[: MINIMO_POSITIVOS - 1] = 1
    seg = {"s": pd.Series(["a"] * n)}
    abaixo = metricas_por_segmento(y, np.full(n, 0.2), np.ones(n), seg)[0]
    assert abaixo["motivo"] == "positivos insuficientes"
    y[MINIMO_POSITIVOS - 1] = 1
    no_limite = metricas_por_segmento(y, np.full(n, 0.2), np.ones(n), seg)[0]
    assert no_limite["motivo"] is None and no_limite["log_loss"] is not None


def test_colunas_de_segmento_usa_as_faixas_da_tarefa_5():
    base = pd.DataFrame(
        {
            "dia_semana": ["segunda", "domingo"],
            "faixa_horario": ["noite", "tarde"],
            "regiao": ["sudeste", "sul"],
            "idade": [15, 62],
            "sexo": ["F", "M"],
            "renda_pc_sm": [0.3, 6.0],
            "local": ["em_casa", "fora_de_casa"],
        }
    )
    seg = colunas_de_segmento(base)
    assert list(seg) == [
        "dia_semana", "faixa_horario", "regiao", "faixa_etaria", "sexo", "faixa_renda", "local",
    ]  # fmt: skip
    assert seg["faixa_etaria"].tolist() == ["10-17", "60+"]
    assert seg["faixa_renda"].iloc[0] != seg["faixa_renda"].iloc[1]
    assert seg["regiao"].tolist() == ["sudeste", "sul"]
