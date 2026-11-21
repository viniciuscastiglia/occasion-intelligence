"""Grade do mapa de calor: o score de cada dia da semana x faixa de horário, demais campos fixos."""

from typing import get_args

import pandas as pd

from occasion.inference.contract import Contexto, DiaSemana, FaixaHorario
from occasion.inference.predict import ArtefatoModelo, prever


def grade_do_mapa(contexto: Contexto) -> list[Contexto]:
    """Os 35 contextos (dia x faixa), na ordem do contrato, mudando só esses dois campos."""
    return [
        contexto.model_copy(update={"dia_semana": dia, "faixa_horario": faixa})
        for dia in get_args(DiaSemana)
        for faixa in get_args(FaixaHorario)
    ]


def tabela_do_mapa(contexto: Contexto, artefato: ArtefatoModelo) -> pd.DataFrame:
    """Uma linha por célula da grade, com o score de `prever` e a marca da célula do contexto."""
    linhas = [
        {
            "dia_semana": c.dia_semana,
            "faixa_horario": c.faixa_horario,
            "score": prever(c, artefato).occasion_score,
            "atual": c.dia_semana == contexto.dia_semana
            and c.faixa_horario == contexto.faixa_horario,
        }
        for c in grade_do_mapa(contexto)
    ]
    return pd.DataFrame(linhas, columns=["dia_semana", "faixa_horario", "score", "atual"])
