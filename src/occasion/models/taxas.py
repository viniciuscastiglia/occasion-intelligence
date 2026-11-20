"""Degrau A: baseline sem ML, uma tabela de taxas ponderadas e suavizadas.

A taxa de cada combinação de dimensões é suavizada em direção à taxa geral:
`(soma dos pesos dos positivos + m x taxa geral) / (soma dos pesos + m)`. Combinação sem caso no
treino recebe a taxa geral. As contribuições por dimensão (log-chances da taxa marginal suavizada
menos log-chances da taxa geral) são os fatores que a aplicação mostra.
"""

from typing import Self, cast

import numpy as np
import pandas as pd

from occasion.features.variaveis import adicionar_derivadas
from occasion.models.base import logit

DIMENSOES_PARTE_1: list[str] = ["dia_semana", "faixa_horario", "local"]
# A parte 2 não usa `tem_pizza` na tabela: com poucos casos de pizza ela ficaria vazia.
DIMENSOES_PARTE_2: list[str] = ["regiao", "local"]


class TabelaDeTaxas:
    """Tabela de taxas ponderadas e suavizadas por combinação de dimensões.

    Segue a interface `ModeloParte`, com uma exceção: `contribuicoes(X).sum(axis=1) + base_` não é
    o logit de `predict_proba(X)`. A previsão vem da tabela conjunta e cada contribuição é a taxa
    marginal da dimensão em log-chances menos a `base_`, então a soma é uma aproximação aditiva.
    """

    def __init__(self, dimensoes: list[str], forca_suavizacao: float = 50.0) -> None:
        if not dimensoes:
            raise ValueError("é preciso ao menos uma dimensão")
        if forca_suavizacao < 0:
            raise ValueError("a força de suavização não pode ser negativa")
        self.dimensoes = list(dimensoes)
        self.forca_suavizacao = float(forca_suavizacao)
        self.taxa_geral_: float = float("nan")
        self._conjunta: pd.Series = pd.Series(dtype=float)
        self._marginais: dict[str, pd.Series] = {}

    @property
    def variaveis(self) -> list[str]:
        """As dimensões da tabela, que são os fatores que ela explica."""
        return list(self.dimensoes)

    @property
    def base_(self) -> float:
        """Log-chances da taxa geral do treino."""
        return float(logit(self.taxa_geral_))

    def _checar_colunas(self, X: pd.DataFrame) -> None:
        faltam = [d for d in self.dimensoes if d not in X.columns]
        if faltam:
            raise ValueError(f"faltam dimensões em X: {faltam}")

    def _recusar_nan(self, X: pd.DataFrame) -> None:
        """Valor ausente nas dimensões é erro de montagem; valor não visto é outra coisa."""
        com_nan = [d for d in self.dimensoes if bool(X[d].isna().any())]
        if com_nan:
            raise ValueError(f"NaN nas dimensões {com_nan}: o contexto está incompleto")

    def _taxas(
        self, chaves: pd.DataFrame, y: pd.Series, w: pd.Series, colunas: list[str]
    ) -> pd.Series:
        base = cast(pd.DataFrame, chaves[colunas]).assign(_w=w.to_numpy(), _wy=(w * y).to_numpy())
        soma = cast(pd.DataFrame, base.groupby(colunas, observed=True)[["_w", "_wy"]].sum())
        soma_w = cast(pd.Series, soma["_w"])
        soma_wy = cast(pd.Series, soma["_wy"])
        m = self.forca_suavizacao
        return (soma_wy + m * self.taxa_geral_) / (soma_w + m)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Ajusta a tabela; sem `sample_weight`, todas as linhas pesam o mesmo."""
        X = adicionar_derivadas(X)
        self._checar_colunas(X)
        if sample_weight is None:
            sample_weight = pd.Series(np.ones(len(X)))
        if not (len(X) == len(y) == len(sample_weight)):
            raise ValueError("X, y e sample_weight precisam ter o mesmo tamanho")
        pesos = pd.Series(np.asarray(sample_weight, dtype=float))
        if len(pesos) == 0 or not np.isfinite(pesos).all() or (pesos <= 0).any():
            raise ValueError("peso inválido: todos os pesos devem ser finitos e positivos")
        pesos = pesos / pesos.mean()
        alvo = pd.Series(np.asarray(y, dtype=float))
        chaves = cast(pd.DataFrame, X[self.dimensoes]).reset_index(drop=True)
        self.taxa_geral_ = float((pesos * alvo).sum() / pesos.sum())
        self._conjunta = self._taxas(chaves, alvo, pesos, self.dimensoes)
        self._marginais = {d: self._taxas(chaves, alvo, pesos, [d]) for d in self.dimensoes}
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade por linha (vetor 1-D); combinação não vista recebe a taxa geral."""
        X = adicionar_derivadas(X)
        self._checar_colunas(X)
        self._recusar_nan(X)
        indice = pd.MultiIndex.from_frame(
            cast(pd.DataFrame, X[self.dimensoes]).reset_index(drop=True)
        )
        taxas = self._conjunta.reindex(indice).to_numpy(dtype=float)
        return np.where(np.isnan(taxas), self.taxa_geral_, taxas)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada dimensão em log-chances; valor não visto contribui 0."""
        X = adicionar_derivadas(X)
        self._checar_colunas(X)
        self._recusar_nan(X)
        base = self.base_
        saida: dict[str, np.ndarray] = {}
        for d in self.dimensoes:
            valores = cast(pd.Series, X[d]).reset_index(drop=True)
            taxas = self._marginais[d].reindex(valores).to_numpy(dtype=float)
            taxas = np.where(np.isnan(taxas), self.taxa_geral_, taxas)
            saida[d] = logit(taxas) - base
        return pd.DataFrame(saida, index=X.index)
