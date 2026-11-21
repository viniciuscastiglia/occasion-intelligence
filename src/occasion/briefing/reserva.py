"""Texto reserva: briefing por modelo fixo, usado quando o LLM falha ou é reprovado."""

from occasion.inference.contract import Contexto, Fator, Resultado
from occasion.rotulos import ROTULOS_DIA, rotulo_da_parte, rotulo_da_variavel, valor_legivel

MAXIMO_FATORES = 3


def _fator(fator: Fator) -> str:
    verbo = "puxa para cima" if fator.direcao == "aumenta" else "puxa para baixo"
    return (
        f"{rotulo_da_variavel(fator.variavel)} {valor_legivel(fator.variavel, fator.valor)} "
        f"({rotulo_da_parte(fator.parte)}, {verbo})"
    )


def texto_reserva(contexto: Contexto, resultado: Resultado) -> str:
    """Três a cinco frases montadas só com a categoria, o score e os fatores recebidos."""
    ocasiao = resultado.score_components["ocasiao"].posicao
    preferencia = resultado.score_components["preferencia"].posicao
    frases = [
        (
            f"Para este contexto ({ROTULOS_DIA[contexto.dia_semana].lower()}, {contexto.uf}), "
            f"o Occasion Score é {resultado.occasion_score:.0f} de 100, classificação "
            f"{resultado.classification}."
        ),
        (
            f"A posição na parte de ocasião é {ocasiao:.0f} e na parte de preferência pelo "
            f"refrigerante de guaraná (o sabor registrado na POF) é {preferencia:.0f}."
        ),
        f"A categoria recomendada é: {resultado.recommendation_category}.",
    ]
    fatores = resultado.top_factors[:MAXIMO_FATORES]
    if fatores:
        frases.append("Fatores do contexto: " + "; ".join(_fator(f) for f in fatores) + ".")
    frases.append(
        "Trata-se de uma associação observada nos dados da POF, e a recomendação "
        "não estima o efeito da ação."
    )
    return " ".join(frases)
