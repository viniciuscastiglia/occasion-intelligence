"""Degraus B e C: regressão logística regularizada de uma parte do score.

O degrau B tem efeitos principais lineares (categóricas em one-hot, numéricas padronizadas). O
degrau C troca idade e renda por curvas (`SplineTransformer`) e soma interações declaradas no
pré-registro (`occasion.experimentos.especificacao`). A regularização é L2, sobre todas as
colunas menos o intercepto, e nenhum nível de categórica é descartado: a regularização puxa cada
UF para perto da média e `regiao` com `uf` dão o efeito de "UF perto da média da região".

Domínios fechados: valor fora de `DOMINIOS` dá `ValueError`; valor do domínio ausente do treino
é aceito (o one-hot tem categorias fixas e o coeficiente dele fica perto de zero).

Declarado, não corrigido: os nós das curvas saem sem peso, porque o `sample_weight` não chega ao
`SplineTransformer` dentro do `ColumnTransformer`.
"""

from collections.abc import Sequence
from typing import Any, Self, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    SplineTransformer,
    StandardScaler,
)

from occasion.experimentos.especificacao import CURVAS, SPLINE, ParteModelo, variaveis_da_parte
from occasion.features.variaveis import (
    DOMINIOS,
    NUMERICAS,
    adicionar_derivadas,
    checar_dominios,
)
from occasion.models.base import sigmoide

_PIZZA = "tem_pizza"
_DERIVADA_DE: dict[str, str] = {"fim_de_semana": "dia_semana"}


def _cruzada(a: str, b: str) -> str:
    return f"{a}|{b}"


class ModeloLogistico:
    """Logística L2 de uma parte (`ocasiao`, `preferencia` ou `direto`); interface `ModeloParte`."""

    def __init__(
        self,
        parte: ParteModelo,
        C: float,
        com_curvas: bool = False,
        interacoes: Sequence[tuple[str, str]] = (),
    ) -> None:
        if not C > 0:
            raise ValueError(f"C deve ser positivo: {C!r}")
        self.parte: ParteModelo = parte
        self.C = float(C)
        self.com_curvas = com_curvas
        self.interacoes: list[tuple[str, str]] = [(a, b) for a, b in interacoes]
        self._variaveis = variaveis_da_parte(parte)
        for par in self.interacoes:
            for v in par:
                categorica = v in DOMINIOS and v not in NUMERICAS
                if not categorica or (v not in self._variaveis and v not in _DERIVADA_DE):
                    raise ValueError(f"interação {par} usa {v!r}, que não é categórica da parte")
                if v in _DERIVADA_DE and _DERIVADA_DE[v] not in self._variaveis:
                    raise ValueError(f"interação {par}: falta {_DERIVADA_DE[v]!r} na parte")
        self.base_: float = float("nan")
        self.n_parametros: int = 0

    # ------------------------------------------------------------------ montagem

    @property
    def variaveis(self) -> list[str]:
        return list(self._variaveis)

    def _categoricas(self) -> list[str]:
        return [v for v in self._variaveis if v not in NUMERICAS and v != _PIZZA]

    def _preparar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Cópia só com o que o modelo usa; recusa coluna ausente, NaN e valor fora do domínio."""
        com = adicionar_derivadas(X)
        usadas = list(
            dict.fromkeys([*self._variaveis, *(v for par in self.interacoes for v in par)])
        )
        faltam = [c for c in usadas if c not in com.columns]
        if faltam:
            raise ValueError(f"faltam colunas em X: {faltam}")
        checar_dominios(com)
        numericas = [v for v in self._variaveis if v in NUMERICAS]
        if bool(np.asarray(com[numericas].isna()).any()):
            raise ValueError(f"NaN nas variáveis numéricas {numericas}: o contexto está incompleto")
        if bool((com["renda_pc_sm"] < 0).any()):
            raise ValueError("renda_pc_sm negativa")
        for a, b in self.interacoes:
            # chave pelas posições no domínio, para não depender do dtype (6 e 6.0 dão a mesma)
            ia = pd.Index(DOMINIOS[a]).get_indexer(cast(pd.Series, com[a]))
            ib = pd.Index(DOMINIOS[b]).get_indexer(cast(pd.Series, com[b]))
            com[_cruzada(a, b)] = [f"{i}|{j}" for i, j in zip(ia, ib, strict=True)]
        return cast(pd.DataFrame, com[[*usadas, *(_cruzada(a, b) for a, b in self.interacoes)]])

    def _transformador(self) -> ColumnTransformer:
        blocos: list[tuple[str, Any, list[str]]] = []
        for v in self._categoricas():
            codificador = OneHotEncoder(categories=cast(Any, [DOMINIOS[v]]), sparse_output=True)
            blocos.append((f"cat:{v}", codificador, [v]))
        for v in (v for v in self._variaveis if v in NUMERICAS):
            curva = self.com_curvas and v in CURVAS
            passos: list[tuple[str, Any]] = []
            if v == "renda_pc_sm":
                passos.append(("log1p", FunctionTransformer(np.log1p)))
            passos.append(
                ("curva", SplineTransformer(**SPLINE)) if curva else ("z", StandardScaler())
            )
            blocos.append((f"num:{v}", Pipeline(passos), [v]))
        if _PIZZA in self._variaveis:
            blocos.append((f"bin:{_PIZZA}", "passthrough", [_PIZZA]))
        for a, b in self.interacoes:
            niveis = [f"{i}|{j}" for i in range(len(DOMINIOS[a])) for j in range(len(DOMINIOS[b]))]
            blocos.append(
                (
                    f"inter:{a}|{b}",
                    OneHotEncoder(categories=cast(Any, [niveis]), sparse_output=True),
                    [_cruzada(a, b)],
                )
            )
        return ColumnTransformer(blocos, sparse_threshold=1.0)

    # ------------------------------------------------------------------ ajuste

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None = None) -> Self:
        """Ajusta o modelo; sem `sample_weight`, pesos iguais. Os pesos viram média 1."""
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

        self._ct = self._transformador()
        Xt = sparse.csr_matrix(self._ct.fit_transform(Xp))
        self._lr = LogisticRegression(
            C=self.C, l1_ratio=0.0, solver="newton-cholesky", max_iter=200, tol=1e-8
        )
        self._lr.fit(Xt, alvo, sample_weight=pesos)

        coef = self._lr.coef_.ravel()
        self._coef = coef
        self._media = np.asarray(Xt.T @ pesos, dtype=float).ravel() / pesos.sum()
        self.base_ = float(np.ravel(self._lr.intercept_)[0] + float(self._media @ coef))
        self._indices: dict[str, slice] = {
            nome: cast(slice, fatia)
            for nome, fatia in self._ct.output_indices_.items()
            if nome != "remainder"
        }
        self.n_parametros = self._contar_parametros()
        return self

    def _contar_parametros(self) -> int:
        """Parâmetros do que foi ajustado, sem o intercepto (um por coluna livre)."""
        total = 0
        transformadores = {nome: t for nome, t, _ in self._ct.transformers_}
        for nome, fatia in self._indices.items():
            colunas = fatia.stop - fatia.start
            if nome.startswith("cat:"):
                niveis = len(cast(OneHotEncoder, transformadores[nome]).categories_[0])
                total += niveis - 1
            elif nome.startswith("inter:"):
                a, b = nome.removeprefix("inter:").split("|")
                if colunas != len(DOMINIOS[a]) * len(DOMINIOS[b]):
                    raise ValueError(f"interação {nome}: colunas não batem com os domínios")
                total += (len(DOMINIOS[a]) - 1) * (len(DOMINIOS[b]) - 1)
            else:  # numérica (1 coluna ou as colunas da curva) e `tem_pizza` (1 coluna)
                total += colunas
        return total

    def _exigir_ajuste(self) -> None:
        if not hasattr(self, "_lr"):
            raise RuntimeError("o modelo ainda não foi ajustado")

    # ------------------------------------------------------------------ previsão

    def _transformar(self, X: pd.DataFrame) -> Any:
        self._exigir_ajuste()
        return sparse.csr_matrix(self._ct.transform(self._preparar(X)))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probabilidade por linha (vetor 1-D)."""
        return sigmoide(self._lr.decision_function(self._transformar(X)))

    def coeficiente(self, nome: str) -> float:
        """Coeficiente de uma variável de uma só coluna (numérica linear ou `tem_pizza`).

        `tem_pizza` passa sem escala, então o valor é em log-chances; as numéricas lineares estão
        padronizadas (por desvio-padrão do treino). Variável com várias colunas dá `KeyError`.
        """
        self._exigir_ajuste()
        for prefixo in ("bin:", "num:"):
            fatia = self._indices.get(prefixo + nome)
            if fatia is not None and fatia.stop - fatia.start == 1:
                return float(self._coef[fatia.start])
        raise KeyError(f"{nome!r} não tem um coeficiente único neste modelo")

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada variável em log-chances, em relação ao contexto médio do treino.

        Para cada termo, `coef · (x − média ponderada do treino)` nas colunas dele. A coluna de
        uma interação vai meio a meio para as duas variáveis; `fim_de_semana` conta como
        `dia_semana`, de onde deriva. Vale `contribuicoes(X).sum(axis=1) + base_ ==
        logit(predict_proba(X))`.
        """
        Xt = self._transformar(X).tocsc()
        saida = {v: np.zeros(Xt.shape[0]) for v in self._variaveis}
        for nome, fatia in self._indices.items():
            coef = self._coef[fatia]
            valor = np.asarray(Xt[:, fatia] @ coef).ravel() - float(self._media[fatia] @ coef)
            origem = nome.split(":", 1)[1]
            if nome.startswith("inter:"):
                for v in origem.split("|"):
                    saida[_DERIVADA_DE.get(v, v)] += valor / 2.0
            else:
                saida[origem] += valor
        return pd.DataFrame(saida, index=X.index)
