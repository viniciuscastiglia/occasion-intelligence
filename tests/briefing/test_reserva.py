"""Texto reserva: modelo fixo que sempre passa na conferência."""

from collections.abc import Callable

import pytest

from occasion.briefing.conferencia import conferir
from occasion.briefing.prompt import entrada_do_briefing
from occasion.briefing.reserva import texto_reserva
from occasion.inference.contract import Contexto, Resultado

CATEGORIAS = [
    "Sustentar, via ação na pizzaria",
    "Converter, via influencers e conteúdo digital",
    "Criar ocasião, via combo em app de delivery",
    "Não priorizar",
]


@pytest.mark.parametrize("categoria", CATEGORIAS)
def test_texto_reserva_passa_na_conferencia(
    contexto: Contexto, montar_resultado: Callable[[str], Resultado], categoria: str
) -> None:
    resultado = montar_resultado(categoria)
    texto = texto_reserva(contexto, resultado)
    r = conferir(texto, entrada_do_briefing(contexto, resultado))
    assert r.aprovado, (texto, r.motivos)


def test_texto_reserva_cita_categoria_e_fatores(
    contexto: Contexto, montar_resultado: Callable[[str], Resultado]
) -> None:
    resultado = montar_resultado("Converter, via combo em app de delivery")
    texto = texto_reserva(contexto, resultado)
    assert "Converter, via combo em app de delivery" in texto
    assert "faixa de horário" in texto
    assert "idade" in texto


def test_texto_reserva_mostra_sigla_da_uf_e_nao_o_codigo(
    contexto: Contexto, montar_resultado: Callable[[str], Resultado]
) -> None:
    ctx = contexto.model_copy(update={"uf": "RJ"})
    resultado = montar_resultado(CATEGORIAS[0])
    fatores = [resultado.top_factors[0].model_copy(update={"variavel": "uf", "valor": "33"})]
    resultado = resultado.model_copy(update={"top_factors": fatores})
    texto = texto_reserva(ctx, resultado)
    assert "UF RJ" in texto
    assert "UF 33" not in texto
    assert conferir(texto, entrada_do_briefing(ctx, resultado)).aprovado
