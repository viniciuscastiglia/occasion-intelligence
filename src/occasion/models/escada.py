"""Fábrica da escada de modelos (degraus A a D) e seus espaços de busca pré-registrados."""

from itertools import product
from typing import Any, Literal

from occasion.experimentos.especificacao import (
    ESPACO_A,
    GRADE_C_LOGISTICA,
    INTERACOES,
    ParteModelo,
    espaco_lightgbm,
)
from occasion.models.arvores import ModeloLightGBM
from occasion.models.base import ModeloParte
from occasion.models.logistica import ModeloLogistico
from occasion.models.taxas import TabelaDeTaxas

Degrau = Literal["A", "B", "C", "D"]


def construir(degrau: Degrau, parte: ParteModelo, params: dict[str, Any]) -> ModeloParte:
    """Modelo do degrau, ainda sem ajuste, com uma configuração de `espaco(degrau, parte)`."""
    if degrau == "A":
        return TabelaDeTaxas(params["dimensoes"], params["suavizacao"])
    if degrau == "B":
        return ModeloLogistico(parte, C=params["C"])
    if degrau == "C":
        return ModeloLogistico(
            parte,
            C=params["C"],
            com_curvas=True,
            interacoes=INTERACOES[parte][params["interacoes"]],
        )
    if degrau == "D":
        return ModeloLightGBM(parte, **params)
    raise ValueError(f"degrau desconhecido: {degrau!r}")


def espaco(degrau: Degrau, parte: ParteModelo) -> list[dict[str, Any]]:
    """Configurações do degrau na ordem do desempate (da mais simples à mais complexa)."""
    if degrau == "A":
        return list(ESPACO_A[parte])
    if degrau == "B":
        return [{"C": c} for c in GRADE_C_LOGISTICA]
    if degrau == "C":
        return [
            {"C": c, "interacoes": nome}
            for nome, c in product(INTERACOES[parte], GRADE_C_LOGISTICA)
        ]
    if degrau == "D":
        return espaco_lightgbm()
    raise ValueError(f"degrau desconhecido: {degrau!r}")
