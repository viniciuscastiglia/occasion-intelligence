"""Degrau D: LightGBM de uma parte do score.

As categóricas entram como `pd.Categorical` com as categorias fixas de `DOMINIOS` (o LightGBM as
trata como categóricas nativas); as numéricas entram sem transformação. Valor fora do domínio dá
`ValueError`; valor do domínio ausente do treino é aceito (cai no ramo "outras categorias").
`fim_de_semana` não é variável do degrau D: ele vê `dia_semana`.

As contribuições são os valores SHAP exatos das árvores (`pred_contrib=True`), em log-chances; a
última coluna é a base, igual em todas as linhas. Por isso, aqui,
`contribuicoes(X).sum(axis=1) + base_ == logit(predict_proba(X))`.
"""

from typing import Any, Self, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.features.variaveis import (
    DOMINIOS,
    NUMERICAS,
    adicionar_derivadas,
    checar_dominios,
)


class ModeloLightGBM:
    """LightGBM binário de uma parte; interface `ModeloParte`. `params` vão ao `LGBMClassifier`."""

    def __init__(self, parte: ParteModelo, **params: Any) -> None:
        self.parte: ParteModelo = parte
        self.params: dict[str, Any] = dict(params)
        self._variaveis = variaveis_da_parte(parte)
        self.base_: float = float("nan")

    @property
    def variaveis(self) -> list[str]:
        return list(self._variaveis)

    def _preparar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Só as variáveis do modelo, na ordem fixa; recusa coluna ausente, NaN e fora do domínio."""
        com = adicionar_derivadas(X)
        faltam = [v for v in self._variaveis if v not in com.columns]
        if faltam:
            raise ValueError(f"faltam colunas em X: {faltam}")
        checar_dominios(com)
        numericas = [v for v in self._variaveis if v in NUMERICAS]
        if bool(np.asarray(com[numericas].isna()).any()):
            raise ValueError(f"NaN nas variáveis numéricas {numericas}: o contexto está incompleto")
        saida = pd.DataFrame(index=X.index)
        for v in self._variaveis:
            coluna = cast(pd.Series, com[v])
            if v in NUMERICAS:
                saida[v] = coluna.astype(float)
            else:
                saida[v] = pd.Categorical(coluna, categories=DOMINIOS[v])
        return saida

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Ajusta o modelo; sem `sample_weight`, pesos iguais. Peso zero é aceito; média vira 1."""
        Xp = self._preparar(X)
        alvo = np.asarray(y, dtype=float)
        if sample_weight is None:
            pesos = np.ones(len(Xp))
        else:
            pesos = np.asarray(sample_weight, dtype=float)
        if not (len(Xp) == len(alvo) == len(pesos)) or len(Xp) == 0:
            raise ValueError("X, y e sample_weight precisam ter o mesmo tamanho (e não ser vazios)")
        if not np.isfinite(pesos).all() or (pesos < 0).any() or not pesos.sum() > 0:
            raise ValueError(
                "peso inválido: pesos finitos, não negativos e com soma positiva (zero é aceito)"
            )
        pesos = pesos / pesos.mean()

        self._lgbm = LGBMClassifier(**self.params)
        self._lgbm.fit(Xp, alvo, sample_weight=pesos)
        base = self._contribuicoes_brutas(Xp)[:, -1]
        self.base_ = float(np.average(base, weights=pesos))
        return self

    def _exigir_ajuste(self) -> None:
        if not hasattr(self, "_lgbm"):
            raise RuntimeError("o modelo ainda não foi ajustado")

    def _contribuicoes_brutas(self, Xp: pd.DataFrame) -> np.ndarray:
        return np.asarray(self._lgbm.booster_.predict(Xp, pred_contrib=True), dtype=float)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade por linha (vetor 1-D)."""
        self._exigir_ajuste()
        return np.asarray(self._lgbm.predict_proba(self._preparar(X)))[:, 1]

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada variável em log-chances (valores SHAP das árvores)."""
        self._exigir_ajuste()
        bruto = self._contribuicoes_brutas(self._preparar(X))
        return pd.DataFrame(bruto[:, :-1], columns=self._variaveis, index=X.index)
