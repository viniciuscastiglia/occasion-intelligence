"""Calibração de Platt sobre o logit de um modelo de parte.

`p = sigmoide(inclinacao · logit(p_base) + intercepto)`. A calibração é uma transformação afim do
logit, então as contribuições escalam pela inclinação e o intercepto vai para a base: a identidade
`contribuicoes(X).sum(axis=1) + base_ == logit(predict_proba(X))` vale quando vale no modelo base
(degraus B, C e D). Sobre a tabela de taxas (degrau A) ela não vale, como na própria tabela.
Os coeficientes saem de `ajustar_sigmoide`; esta classe só aplica.
"""

import numpy as np
import pandas as pd

from occasion.models.base import ModeloParte, logit, sigmoide


class ModeloCalibrado:
    """Modelo de parte já ajustado, com calibração afim no logit; sem `fit` próprio."""

    def __init__(self, base: ModeloParte, inclinacao: float, intercepto: float) -> None:
        if not (np.isfinite(inclinacao) and np.isfinite(intercepto)):
            raise ValueError("inclinação e intercepto devem ser finitos")
        self.modelo_base = base
        self.inclinacao = float(inclinacao)
        self.intercepto = float(intercepto)

    @property
    def variaveis(self) -> list[str]:
        return self.modelo_base.variaveis

    @property
    def base_(self) -> float:
        return self.inclinacao * self.modelo_base.base_ + self.intercepto

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade calibrada por linha (vetor 1-D); `p_base` 0 ou 1 não dá infinito."""
        z = self.inclinacao * logit(self.modelo_base.predict_proba(X)) + self.intercepto
        return sigmoide(z)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.inclinacao * self.modelo_base.contribuicoes(X)
