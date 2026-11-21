"""Contextos em que o modelo mais erra: o observado contra o previsto, com peso e um escore z."""

from typing import Any

import numpy as np
import pandas as pd

from occasion.models.base import EPS

__all__ = ["POR_PADRAO", "contextos_com_mais_erro"]

POR_PADRAO: list[str] = ["regiao", "dia_semana", "faixa_horario", "local"]


def contextos_com_mais_erro(
    base: pd.DataFrame,
    y: np.ndarray,
    p: np.ndarray,
    por: list[str] | None = None,
    n: int = 10,
    minimo_linhas: int = 500,
) -> dict[str, list[dict[str, Any]]]:
    """Os `n` contextos mais subestimados e os `n` mais superestimados.

    O contexto é a combinação das colunas de `por` (padrão: `POR_PADRAO`). Com `w` igual a
    `peso_final` normalizado para média 1 na base passada, `O = Σ w·y`, `E = Σ w·p` e
    `z = (O − E) / sqrt(Σ w² p (1 − p))`. `subestimados` têm o maior `z` (observado acima do
    previsto) e `superestimados`, o menor. Contextos com menos de `minimo_linhas` linhas ficam de
    fora; um contexto não aparece nas duas listas. As taxas são ponderadas.
    """
    por = POR_PADRAO if por is None else por
    y, p = np.asarray(y, dtype=float), np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    w = base["peso_final"].to_numpy(dtype=float)
    w = w / w.mean()
    chaves = base[por].reset_index(drop=True)
    contextos: list[dict[str, Any]] = []
    for chave, indices in chaves.groupby(por, sort=True, observed=True).indices.items():
        if len(indices) < minimo_linhas:
            continue
        valores = chave if isinstance(chave, tuple) else (chave,)
        wk, yk, pk = w[indices], y[indices], p[indices]
        variancia = float(np.sum(wk**2 * pk * (1.0 - pk)))
        if variancia <= 0:
            continue
        observado, esperado, peso = float(np.sum(wk * yk)), float(np.sum(wk * pk)), float(wk.sum())
        contextos.append(
            dict(zip(por, valores, strict=True))
            | {
                "n": len(indices),
                "positivos": int(np.sum(yk == 1)),
                "taxa_observada": observado / peso,
                "taxa_prevista": esperado / peso,
                "z": (observado - esperado) / np.sqrt(variancia),
            }
        )
    contextos.sort(key=lambda c: c["z"], reverse=True)
    subestimados = contextos[:n]
    superestimados = [c for c in reversed(contextos[n:])][:n]
    return {"subestimados": subestimados, "superestimados": superestimados}
