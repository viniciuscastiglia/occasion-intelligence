"""Explicabilidade: importância global e estabilidade das contribuições entre as dobras.

As contribuições vêm de `ModeloParte.contribuicoes`, em log-chances e em relação ao contexto
médio. Descrevem o que o modelo faz, não um efeito causal de cada variável.
"""

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.eda.descritivas import faixa_de_renda, faixa_etaria
from occasion.models.base import ModeloParte

__all__ = ["LIMITE_INSTAVEL", "estabilidade", "importancia_global"]

LIMITE_INSTAVEL = 0.05  # log-chances


def importancia_global(modelo: ModeloParte, X: pd.DataFrame, w: np.ndarray) -> dict[str, float]:
    """Média ponderada de `|contribuição|` de cada variável, da maior para a menor."""
    w = np.asarray(w, dtype=float)
    contribuicoes = modelo.contribuicoes(X)
    importancia = {
        str(v): float(np.sum(w * np.abs(contribuicoes[v].to_numpy())) / np.sum(w))
        for v in contribuicoes.columns
    }
    return dict(sorted(importancia.items(), key=lambda par: par[1], reverse=True))


def _niveis(X: pd.DataFrame, variavel: str) -> pd.Series:
    """Nível de cada linha: faixas da tarefa 5 para idade e renda, o próprio valor nas demais."""
    coluna = cast(pd.Series, X[variavel])
    if variavel == "idade":
        return faixa_etaria(coluna)
    if variavel == "renda_pc_sm":
        return faixa_de_renda(coluna)
    return coluna


def estabilidade(
    modelos: Sequence[ModeloParte], X: pd.DataFrame, w: np.ndarray
) -> list[dict[str, Any]]:
    """A contribuição média de cada variável e nível em cada modelo (um por dobra externa).

    A média é ponderada por `w` sobre as linhas do nível. `instavel` vale quando as dobras
    discordam no sinal (há média positiva e negativa) e o maior valor absoluto passa de
    `LIMITE_INSTAVEL` log-chances.
    """
    w = np.asarray(w, dtype=float)
    por_modelo = [m.contribuicoes(X) for m in modelos]
    linhas: list[dict[str, Any]] = []
    for variavel in por_modelo[0].columns:
        niveis = _niveis(X, str(variavel)).to_numpy()
        for nivel in sorted(pd.unique(niveis).tolist(), key=str):
            dentro = niveis == nivel
            peso = float(np.sum(w[dentro]))
            if peso <= 0:
                continue
            medias = [
                float(np.sum(w[dentro] * c[variavel].to_numpy()[dentro]) / peso) for c in por_modelo
            ]
            linhas.append(
                {
                    "variavel": str(variavel),
                    "nivel": str(nivel),
                    "n": int(dentro.sum()),
                    "contribuicao_por_modelo": medias,
                    "instavel": bool(
                        min(medias) < 0 < max(medias)
                        and max(abs(m) for m in medias) > LIMITE_INSTAVEL
                    ),
                }
            )
    return linhas
