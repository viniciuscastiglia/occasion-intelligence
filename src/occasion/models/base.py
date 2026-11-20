"""Funções numéricas compartilhadas pelos modelos (a tarefa 7 completa este módulo)."""

import numpy as np

EPS = 1e-15


def logit(p: np.ndarray | float) -> np.ndarray:
    """Log-odds de `p`, com `p` cortado em `[EPS, 1 - EPS]` para não dar infinito."""
    q = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return np.log(q / (1.0 - q))


def sigmoide(z: np.ndarray | float) -> np.ndarray:
    """Inversa do logit, estável para valores grandes em módulo."""
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=float)))
