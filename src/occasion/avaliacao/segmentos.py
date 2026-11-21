"""Métricas por segmento: dia, horário, região, faixa etária, sexo, faixa de renda e local.

Todas as métricas são ponderadas por `peso_final` (o `w` que o chamador passa). Um segmento com
poucos positivos não tem métrica estável: sai com `None` e o motivo, sem exceção.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.eda.descritivas import faixa_de_renda, faixa_etaria
from occasion.validacao.metricas import log_loss_ponderado, pr_auc_ponderada

__all__ = ["MINIMO_POSITIVOS", "colunas_de_segmento", "metricas_por_segmento"]

MINIMO_POSITIVOS = 10


def colunas_de_segmento(base: pd.DataFrame) -> dict[str, pd.Series]:
    """Os sete cortes da avaliação; idade e renda entram pelas faixas da tarefa 5."""

    def coluna(nome: str) -> pd.Series:
        return cast(pd.Series, base[nome])

    return {
        "dia_semana": coluna("dia_semana"),
        "faixa_horario": coluna("faixa_horario"),
        "regiao": coluna("regiao"),
        "faixa_etaria": faixa_etaria(coluna("idade")),
        "sexo": coluna("sexo"),
        "faixa_renda": faixa_de_renda(coluna("renda_pc_sm")),
        "local": coluna("local"),
    }


def metricas_por_segmento(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, segmentos: dict[str, pd.Series]
) -> list[dict[str, Any]]:
    """Uma linha por segmento e valor: `n`, `positivos`, `log_loss`, `pr_auc` e `o_e`.

    `o_e` é a razão entre o observado e o esperado ponderados (soma de `w·y` sobre soma de `w·p`).
    Com menos de `MINIMO_POSITIVOS` positivos (contagem simples), as métricas saem `None` e
    `motivo = "positivos insuficientes"`.
    """
    y, p, w = (np.asarray(v, dtype=float) for v in (y, p, w))
    linhas: list[dict[str, Any]] = []
    for nome, coluna in segmentos.items():
        valores = coluna.to_numpy()
        for valor in sorted(pd.unique(valores).tolist(), key=str):
            dentro = valores == valor
            positivos = int(np.sum(y[dentro] == 1))
            linha: dict[str, Any] = {
                "segmento": nome,
                "valor": valor,
                "n": int(dentro.sum()),
                "positivos": positivos,
                "log_loss": None,
                "pr_auc": None,
                "o_e": None,
                "motivo": None,
            }
            if positivos < MINIMO_POSITIVOS:
                linha["motivo"] = "positivos insuficientes"
            else:
                yk, pk, wk = y[dentro], p[dentro], w[dentro]
                esperado = float(np.sum(wk * pk))
                linha["log_loss"] = log_loss_ponderado(yk, pk, wk)
                linha["pr_auc"] = pr_auc_ponderada(yk, pk, wk)
                linha["o_e"] = float(np.sum(wk * yk) / esperado) if esperado > 0 else None
            linhas.append(linha)
    return linhas
