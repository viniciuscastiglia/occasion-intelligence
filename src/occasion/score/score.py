"""Score: produto das duas partes, posição na população de referência e classificação.

A posição de um valor é a fração ponderada (por `peso_final`) dos contextos da população de
referência com valor menor ou igual a ele, de 0 a 100. Empates contam como "menor ou igual".
"""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import pandas as pd

from occasion.inference.contract import Classificacao

CORTE_ALTA = 80.0
CORTE_MEDIA = 50.0

Parte = Literal["ocasiao", "preferencia", "produto"]


class ModeloProba(Protocol):
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class ReferenciaScore:
    """Distribuição ponderada das probabilidades na população de referência (valores ordenados)."""

    ocasiao: np.ndarray
    preferencia: np.ndarray
    produto: np.ndarray
    peso_acumulado: dict[str, np.ndarray]  # mesma ordem dos valores, de 0 a 1

    def posicao(self, valor: float, qual: Parte) -> float:
        """Fração ponderada da referência com valor menor ou igual ao informado, de 0 a 100."""
        valores: np.ndarray = getattr(self, qual)
        indice = int(np.searchsorted(valores, valor, side="right"))
        if indice == 0:
            return 0.0
        return float(self.peso_acumulado[qual][indice - 1] * 100.0)


def probabilidades(X: pd.DataFrame, modelo_p1: ModeloProba, modelo_p2: ModeloProba) -> pd.DataFrame:
    """Probabilidade de cada parte e o produto; a parte 2 usa `tem_pizza = 1` em todas as linhas."""
    X2 = X.assign(tem_pizza=1)
    p1 = np.asarray(modelo_p1.predict_proba(X), dtype=float)
    p2 = np.asarray(modelo_p2.predict_proba(X2), dtype=float)
    return pd.DataFrame({"p_ocasiao": p1, "p_preferencia": p2, "p_produto": p1 * p2}, index=X.index)


def construir_referencia(
    celulas: pd.DataFrame, modelo_p1: ModeloProba, modelo_p2: ModeloProba
) -> ReferenciaScore:
    """Referência a partir de todas as linhas da base de células, ponderada por `peso_final`."""
    p = probabilidades(celulas, modelo_p1, modelo_p2)
    pesos = np.asarray(celulas["peso_final"], dtype=float)
    series: dict[str, np.ndarray] = {}
    acumulados: dict[str, np.ndarray] = {}
    for qual, coluna in (
        ("ocasiao", "p_ocasiao"),
        ("preferencia", "p_preferencia"),
        ("produto", "p_produto"),
    ):
        valores = p[coluna].to_numpy(dtype=float)
        ordem = np.argsort(valores, kind="stable")
        series[qual] = valores[ordem]
        acumulado = np.cumsum(pesos[ordem])
        acumulados[qual] = acumulado / acumulado[-1]
    return ReferenciaScore(
        ocasiao=series["ocasiao"],
        preferencia=series["preferencia"],
        produto=series["produto"],
        peso_acumulado=acumulados,
    )


def classificar(score: float) -> Classificacao:
    """`alta` a partir de 80; `média` de 50 até abaixo de 80; `baixa` abaixo de 50."""
    if score >= CORTE_ALTA:
        return "alta"
    if score >= CORTE_MEDIA:
        return "média"
    return "baixa"
