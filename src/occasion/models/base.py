"""Funções numéricas e a interface comum dos modelos de cada parte do score."""

from typing import Protocol, Self

import numpy as np
import pandas as pd

EPS = 1e-15


def logit(p: np.ndarray | float) -> np.ndarray:
    """Log-odds de `p`, com `p` cortado em `[EPS, 1 - EPS]` para não dar infinito."""
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray | float) -> np.ndarray:
    """Inversa do logit, estável para valores grandes em módulo."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))


class ModeloParte(Protocol):
    """Interface dos modelos de uma parte do score: tabela de taxas, logística e LightGBM.

    Os três recebem o contexto como `DataFrame` e chamam `adicionar_derivadas` por conta própria.
    `contribuicoes` devolve uma coluna por item de `variaveis`, em log-chances e em relação ao
    contexto médio. Nos degraus B, C e D, `contribuicoes(X).sum(axis=1) + base_` é igual a
    `logit(predict_proba(X))`. Na tabela de taxas (degrau A) isso não vale: a previsão vem da
    tabela conjunta e as contribuições são as marginais de cada dimensão (seção 10 do design).
    """

    @property
    def variaveis(self) -> list[str]: ...

    @property
    def base_(self) -> float: ...

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> Self: ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame: ...
