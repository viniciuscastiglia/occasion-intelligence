"""Decision Engine: regras explícitas que transformam as duas posições do score e o contexto
em um objetivo, um canal de ativação, uma categoria e uma justificativa em texto.

Não há modelo aqui. "Preferência alta" significa guaraná acima do contexto mediano da
população de referência, não guaraná à frente da cola. A regra 5 (rural em casa vai para o
varejo) é uma suposição de negócio, não um resultado da POF.
"""

from dataclasses import dataclass
from typing import Literal

from occasion.inference.contract import UF, Fator, Local, Situacao

CORTE_OCASIAO = 75.0
CORTE_PREFERENCIA = 50.0
IDADE_JOVEM = 30
MAXIMO_FATORES = 3

Objetivo = Literal["Sustentar", "Converter", "Criar ocasião", "Não priorizar"]
Canal = Literal[
    "influencers e conteúdo digital",
    "ação na pizzaria",
    "combo em app de delivery",
    "pizza e guaraná juntos no varejo",
]

MIDIA_REGIONAL = "mídia regional"
FRASE_FINAL = (
    "A recomendação indica que a ativação combina com o contexto; ela não estima o efeito da ação."
)


@dataclass(frozen=True)
class Decisao:
    """Resultado das regras: objetivo, canal e a regra de canal que disparou."""

    objetivo: Objetivo
    canal_principal: Canal | None
    canais_complementares: tuple[str, ...]
    categoria: str
    regra: str


def _objetivo(posicao_ocasiao: float, posicao_preferencia: float) -> Objetivo:
    ocasiao_alta = posicao_ocasiao >= CORTE_OCASIAO
    preferencia_alta = posicao_preferencia >= CORTE_PREFERENCIA
    if ocasiao_alta and preferencia_alta:
        return "Sustentar"
    if ocasiao_alta:
        return "Converter"
    if preferencia_alta:
        return "Criar ocasião"
    return "Não priorizar"


def _canal(
    objetivo: Objetivo, local: Local, situacao: Situacao, idade: int
) -> tuple[Canal | None, str]:
    """Primeira regra de canal que se aplica, com o seu identificador."""
    if objetivo == "Não priorizar":
        return None, "canal-1"
    if objetivo == "Converter" and idade < IDADE_JOVEM:
        return "influencers e conteúdo digital", "canal-2"
    if local == "fora_de_casa":
        return "ação na pizzaria", "canal-3"
    if situacao == "urbano":
        return "combo em app de delivery", "canal-4"
    return "pizza e guaraná juntos no varejo", "canal-5"


def decidir(
    posicao_ocasiao: float,
    posicao_preferencia: float,
    local: Local,
    situacao: Situacao,
    idade: int,
    uf: UF,
    ufs_midia_regional: frozenset[str],
) -> Decisao:
    """Aplica as regras da seção 11 do design e devolve a decisão."""
    objetivo = _objetivo(posicao_ocasiao, posicao_preferencia)
    canal, regra = _canal(objetivo, local, situacao, idade)
    if canal is None:
        return Decisao(objetivo, None, (), "Não priorizar", regra)
    complementares = (MIDIA_REGIONAL,) if uf in ufs_midia_regional else ()
    return Decisao(objetivo, canal, complementares, f"{objetivo}, via {canal}", regra)


def _descrever_fator(fator: Fator) -> str:
    verbo = "puxa para cima" if fator.direcao == "aumenta" else "puxa para baixo"
    return f"{fator.variavel} = {fator.valor} ({verbo})"


def justificar(decisao: Decisao, fatores: list[Fator]) -> str:
    """Texto por modelo fixo: objetivo, regra, canais e até três fatores, na ordem recebida."""
    if decisao.canal_principal is None:
        partes = [f"Objetivo: {decisao.objetivo} (regra {decisao.regra})."]
    else:
        partes = [
            f"Objetivo: {decisao.objetivo}; canal principal: {decisao.canal_principal} (regra {decisao.regra})."
        ]
    if decisao.canais_complementares:
        partes.append(f"Canal complementar: {', '.join(decisao.canais_complementares)}.")
    selecionados = fatores[:MAXIMO_FATORES]
    if selecionados:
        partes.append(
            "Fatores do contexto: " + "; ".join(_descrever_fator(f) for f in selecionados) + "."
        )
    partes.append(FRASE_FINAL)
    return " ".join(partes)
