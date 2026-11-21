"""Testes da grade do mapa de calor (dia da semana x faixa de horário)."""

from typing import get_args

import pytest
from tests.inference.test_predict import CONTEXTO_VALIDO, _artefato_unico

from occasion.inference.contract import Contexto, DiaSemana, FaixaHorario
from occasion.inference.grade import grade_do_mapa, tabela_do_mapa
from occasion.inference.predict import prever


@pytest.fixture
def contexto() -> Contexto:
    return Contexto(**CONTEXTO_VALIDO)  # type: ignore[arg-type]


def test_grade_tem_35_contextos_e_so_muda_dia_e_faixa(contexto: Contexto) -> None:
    grade = grade_do_mapa(contexto)
    assert len(grade) == 35
    esperado = [(d, f) for d in get_args(DiaSemana) for f in get_args(FaixaHorario)]
    assert [(c.dia_semana, c.faixa_horario) for c in grade] == esperado
    base = contexto.model_dump(exclude={"dia_semana", "faixa_horario"})
    for c in grade:
        assert c.model_dump(exclude={"dia_semana", "faixa_horario"}) == base


def test_tabela_marca_so_o_contexto_atual(contexto: Contexto) -> None:
    tabela = tabela_do_mapa(contexto, _artefato_unico())
    assert list(tabela.columns) == ["dia_semana", "faixa_horario", "score", "atual"]
    atual = tabela[tabela["atual"]]
    assert len(tabela) == 35 and len(atual) == 1
    assert atual.iloc[0]["dia_semana"] == contexto.dia_semana
    assert atual.iloc[0]["faixa_horario"] == contexto.faixa_horario


def test_score_da_tabela_igual_ao_de_prever(contexto: Contexto) -> None:
    artefato = _artefato_unico()
    tabela = tabela_do_mapa(contexto, artefato)
    for score, c in zip(tabela["score"], grade_do_mapa(contexto), strict=True):
        assert score == pytest.approx(prever(c, artefato).occasion_score)
