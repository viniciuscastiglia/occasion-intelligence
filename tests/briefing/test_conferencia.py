"""Conferência automática do texto do briefing contra a entrada."""

import pytest

from occasion.briefing.conferencia import conferir
from occasion.briefing.prompt import entrada_do_briefing
from occasion.inference.contract import Contexto, Resultado


def test_aceita_formas_equivalentes(contexto: Contexto, resultado: Resultado) -> None:
    entrada = entrada_do_briefing(contexto, resultado)
    texto = "O score é 87, ou 87,3 com uma casa; a probabilidade é 0,31%; posição 92."
    r = conferir(texto, entrada)
    assert r.aprovado, r.motivos


def test_reprova_numero_inventado(contexto: Contexto, resultado: Resultado) -> None:
    entrada = entrada_do_briefing(contexto, resultado)
    r = conferir("A preferência chega a 45% neste contexto.", entrada)
    assert not r.aprovado
    assert any("45" in m for m in r.motivos)


@pytest.mark.parametrize(
    "texto",
    [
        "Isso vende mais pizza.",
        "ISSO VENDE MAIS PIZZA.",
        "Pelo Guaraná Antarctica.",
        "PELO GUARANA ANTARCTICA.",
        "Há muito impacto aqui.",
        "A eficácia é alta.",
        "Isso causa mais consumo.",
        "Vai garantir o resultado.",
        "Os impactos são grandes.",
        "Vamos impactar o público.",
        "Isso causou mais consumo.",
        "Há muitas causas para isso.",
        "O efeito causado é claro.",
        "Há garantia de resultado.",
        "Um resultado garantido.",
        "Os combos vendem mais pizza.",
        "Elas aumentam as vendas.",
        "Isso aumentou as vendas.",
        "Ação eficaz neste contexto.",
    ],
)
def test_reprova_termo_proibido(contexto: Contexto, resultado: Resultado, texto: str) -> None:
    r = conferir(texto, entrada_do_briefing(contexto, resultado))
    assert not r.aprovado
    assert r.motivos


@pytest.mark.parametrize(
    "texto",
    [
        "Não há leitura causal aqui.",
        "A causalidade não é estimada.",
        "Só a clausura e a pausa.",
    ],
)
def test_nao_reprova_causal_nem_palavra_parecida(
    contexto: Contexto, resultado: Resultado, texto: str
) -> None:
    r = conferir(texto, entrada_do_briefing(contexto, resultado))
    assert r.aprovado, r.motivos


def test_termo_nao_casa_dentro_de_outra_palavra(contexto: Contexto, resultado: Resultado) -> None:
    r = conferir(
        "A pausa e a causalidade? Não: só a clausura.", entrada_do_briefing(contexto, resultado)
    )
    assert r.aprovado, r.motivos


def test_aprova_texto_sem_numeros(contexto: Contexto, resultado: Resultado) -> None:
    r = conferir(
        "Contexto favorável à ocasião, com preferência baixa.",
        entrada_do_briefing(contexto, resultado),
    )
    assert r.aprovado
    assert r.motivos == ()
