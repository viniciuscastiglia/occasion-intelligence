"""Regras de decisão das hipóteses e da escada de modelos, todas sobre intervalos de bootstrap."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

from occasion.validacao.bootstrap import Intervalo

__all__ = [
    "correlacao_positiva",
    "melhor_logistico",
    "padrao",
    "sinal_negativo",
    "superioridade",
    "vencedor_escada",
]


def superioridade(ic: Intervalo) -> Literal["confirmada", "refutada"]:
    """Confirmada se o intervalo do ganho exclui zero a favor do candidato."""
    return "confirmada" if ic.ic_inf > 0 else "refutada"


def padrao(
    ic_ganho_da_alternativa: Intervalo,
) -> Literal["sem evidência contra o padrão", "padrão refutado"]:
    """H5, H6 e H9: o padrão cai só se a alternativa tiver ganho com intervalo acima de zero."""
    return (
        "padrão refutado" if ic_ganho_da_alternativa.ic_inf > 0 else "sem evidência contra o padrão"
    )


def sinal_negativo(
    ic: Intervalo,
) -> Literal["confirmada", "efeito no sentido contrário", "os dados não mostram o efeito"]:
    """H4: efeito esperado negativo."""
    if ic.ic_sup < 0:
        return "confirmada"
    if ic.ic_inf > 0:
        return "efeito no sentido contrário"
    return "os dados não mostram o efeito"


def correlacao_positiva(ics: Sequence[Intervalo]) -> Literal["confirmada", "refutada"]:
    """H10: confirmada se todos os intervalos ficam acima de zero."""
    return "confirmada" if all(ic.ic_inf > 0 for ic in ics) else "refutada"


def melhor_logistico(ic_c_contra_b: Intervalo) -> Literal["B", "C"]:
    """Referência de H3: C só se superar B; no empate, o mais simples."""
    return "C" if superioridade(ic_c_contra_b) == "confirmada" else "B"


def vencedor_escada(comparar: Callable[[str, str], Intervalo]) -> tuple[str, list[dict[str, Any]]]:
    """Campeão sequencial: começa em A; B, C e D enfrentam o campeão do momento.

    Devolve o vencedor e as comparações feitas (candidato, referência, intervalo, veredito).
    """
    campeao = "A"
    comparacoes: list[dict[str, Any]] = []
    for candidato in ("B", "C", "D"):
        ic = comparar(candidato, campeao)
        veredito = superioridade(ic)
        comparacoes.append(
            {"candidato": candidato, "referencia": campeao, "intervalo": ic, "veredito": veredito}
        )
        if veredito == "confirmada":
            campeao = candidato
    return campeao, comparacoes
