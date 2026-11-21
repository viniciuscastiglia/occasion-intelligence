"""Serviço do briefing: cache, limite por sessão, chamada ao LLM e conferência."""

import hashlib
import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Literal

from occasion.briefing.cliente import ClienteLLM, FalhaDoLLM
from occasion.briefing.conferencia import conferir
from occasion.briefing.prompt import entrada_do_briefing, montar_mensagens
from occasion.briefing.reserva import texto_reserva
from occasion.inference.contract import Contexto, Resultado

LIMITE_POR_SESSAO = 5
TAMANHO_MAXIMO_DO_CACHE = 1000

MOTIVOS_DA_FALHA = {
    "tempo_esgotado": "tempo esgotado no provedor",
    "cota_esgotada": "cota do provedor esgotada",
    "resposta_invalida": "resposta inválida do provedor",
    "erro_do_provedor": "erro do provedor",
}


@dataclass(frozen=True)
class Briefing:
    texto: str
    fonte: Literal["llm", "reserva"]
    modelo: str | None
    motivo: str | None


def chave_do_cache(contexto: Contexto, resultado: Resultado) -> str:
    """SHA-256 da entrada do briefing, em JSON com chaves ordenadas."""
    entrada = entrada_do_briefing(contexto, resultado)
    serial = json.dumps(entrada, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serial.encode("utf-8")).hexdigest()


def _reserva(contexto: Contexto, resultado: Resultado, motivo: str) -> Briefing:
    return Briefing(texto_reserva(contexto, resultado), "reserva", None, motivo)


def gerar_briefing(
    contexto: Contexto,
    resultado: Resultado,
    cliente: ClienteLLM | None,
    cache: MutableMapping[str, Briefing],
    chamadas_na_sessao: int,
) -> tuple[Briefing, bool]:
    """Devolve o briefing e se o provedor foi chamado.

    O `bool` é True sempre que a chamada ao provedor foi tentada, mesmo que ela tenha falhado
    ou que o texto tenha sido reprovado na conferência (a tentativa gasta cota). É False
    quando o briefing veio do cache, da ausência de cliente ou do limite da sessão.
    Nunca levanta exceção por falha do provedor. Ordem: cache, sem cliente, limite, chamada,
    conferência. Só texto aprovado entra no cache.
    """
    chave = chave_do_cache(contexto, resultado)
    if chave in cache:
        return cache[chave], False
    if cliente is None:
        return _reserva(contexto, resultado, "sem chave configurada"), False
    if chamadas_na_sessao >= LIMITE_POR_SESSAO:
        return _reserva(contexto, resultado, "limite de briefings desta sessão atingido"), False
    try:
        texto = cliente.completar(montar_mensagens(contexto, resultado))
    except FalhaDoLLM as falha:
        return _reserva(contexto, resultado, MOTIVOS_DA_FALHA[falha.motivo]), True
    conferencia = conferir(texto, entrada_do_briefing(contexto, resultado))
    if not conferencia.aprovado:
        motivo = "texto reprovado na conferência: " + "; ".join(conferencia.motivos)
        return _reserva(contexto, resultado, motivo), True
    briefing = Briefing(texto, "llm", cliente.modelo, None)
    cache[chave] = briefing
    while len(cache) > TAMANHO_MAXIMO_DO_CACHE:
        del cache[next(iter(cache))]
    return briefing, True
