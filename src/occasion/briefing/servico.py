"""Serviço do briefing: cache, limites (sessão e processo), chamada ao LLM e conferência."""

import hashlib
import json
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Literal

from occasion.briefing.cliente import ClienteLLM, FalhaDoLLM
from occasion.briefing.conferencia import conferir
from occasion.briefing.limite import LimiteGlobal
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
MOTIVO_LIMITE_GLOBAL = "limite de briefings da demonstração atingido; tente mais tarde"


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
    limite_global: LimiteGlobal | None = None,
) -> tuple[Briefing, bool]:
    """Devolve o briefing e se o provedor foi chamado.

    O `bool` é True sempre que a chamada ao provedor foi tentada, mesmo que ela tenha falhado
    ou que o texto tenha sido reprovado na conferência (a tentativa gasta cota). É False
    quando o briefing veio do cache, da ausência de cliente ou de um dos limites.
    Nunca levanta exceção por falha do provedor. Ordem: cache, sem cliente, limite da sessão,
    limite global do processo (se houver), chamada, conferência. Só texto aprovado entra no
    cache, e o cache não gasta nenhum dos limites.
    """
    chave = chave_do_cache(contexto, resultado)
    if chave in cache:
        return cache[chave], False
    if cliente is None:
        return _reserva(contexto, resultado, "sem chave configurada"), False
    if chamadas_na_sessao >= LIMITE_POR_SESSAO:
        return _reserva(contexto, resultado, "limite de briefings desta sessão atingido"), False
    if limite_global is not None and not limite_global.reservar():
        return _reserva(contexto, resultado, MOTIVO_LIMITE_GLOBAL), False
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
    _descartar_excesso(cache)
    return briefing, True


def _descartar_excesso(cache: MutableMapping[str, Briefing]) -> None:
    """Descarta as entradas mais antigas acima do tamanho máximo, sem nunca levantar exceção.

    O cache é compartilhado entre as threads das sessões: outra sessão pode ter removido a
    chave mais antiga entre a leitura e a remoção (`pop` com padrão ignora a ausência), ou
    mudado o dicionário durante a iteração (RuntimeError, que encerra o descarte desta vez).
    """
    while len(cache) > TAMANHO_MAXIMO_DO_CACHE:
        try:
            mais_antiga = next(iter(cache), None)
        except RuntimeError:
            return
        if mais_antiga is None:
            return
        cache.pop(mais_antiga, None)
