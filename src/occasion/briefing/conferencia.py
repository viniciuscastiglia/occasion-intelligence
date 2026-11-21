"""Conferência automática do briefing: números que existem na entrada e termos proibidos."""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Termos que o briefing não pode conter. A lista é rigorosa de propósito: "por causa de" também
# reprova. Cada termo vale também para as formas derivadas, pelo padrão de `_FORMAS` (impacto,
# impactos, impactar; causa, causas, causar, causado; garante, garantia, garantido; vendem mais).
# "causal" e "causalidade" não reprovam: o aviso do modelo diz "efeito causal" e o texto pode
# repeti-lo para negar o efeito. A comparação é sem acento e sem diferença de maiúsculas.
TERMOS_PROIBIDOS: tuple[str, ...] = (
    "vende mais",
    "aumenta as vendas",
    "aumentar as vendas",
    "causa",
    "garante",
    "garantir",
    "impacto",
    "eficácia",
    "antarctica",
    "coca-cola",
    "coca cola",
    "pepsi",
    "kuat",
)

# Padrão (já sem acento, em minúsculas) de cada termo, com as formas derivadas.
# O que não está aqui vale como a palavra inteira.
_FORMAS: dict[str, str] = {
    "vende mais": r"vend(?:e|em|eu|endo)\s+mais",
    "aumenta as vendas": r"aument(?:a|am|ou|ando)\s+(?:as\s+)?vendas",
    "aumentar as vendas": r"aumentar\s+(?:as\s+)?vendas",
    "causa": r"caus(?:a|as|ar|ara|aria|ou|am|ando|ado|ada|ados|adas|e|em)",
    "garante": r"garant\w*",
    "garantir": r"garant\w*",
    "impacto": r"impact\w*",
    "eficácia": r"eficaci\w*|eficaz(?:es)?",
}

# Número com vírgula ou ponto decimal; o `%` não entra no padrão, porque só muda a escala
# e a escala já é tratada na comparação (x ou 100·x). Separador de milhar ("1.234") não é
# tratado: lê-se como 1,234. Os números do contexto e do resultado são pequenos, e um milhar
# no texto, sem equivalente na entrada, reprova (falha para o lado seguro).
_NUMERO = re.compile(r"\d+(?:[.,]\d+)?")
_TOLERANCIA = 1e-9


@dataclass(frozen=True)
class Conferencia:
    aprovado: bool
    motivos: tuple[str, ...]


def _sem_acento(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c)).lower()


def _para_float(token: str) -> float:
    return float(token.replace(",", "."))


def _casas(token: str) -> int:
    separador = max(token.rfind(","), token.rfind("."))
    return 0 if separador < 0 else len(token) - separador - 1


def _numeros_da_entrada(valor: Any) -> list[float]:
    """Todos os números da entrada, recursivamente, inclusive os embutidos em strings."""
    if isinstance(valor, bool) or valor is None:
        return []
    if isinstance(valor, int | float):
        return [float(valor)]
    if isinstance(valor, str):
        return [_para_float(t) for t in _NUMERO.findall(valor)]
    if isinstance(valor, dict):
        return [n for v in valor.values() for n in _numeros_da_entrada(v)]
    if isinstance(valor, list | tuple):
        return [n for v in valor for n in _numeros_da_entrada(v)]
    return []


def _existe_na_entrada(token: str, candidatos: list[float]) -> bool:
    alvo = _para_float(token)
    casas = _casas(token)
    for x in candidatos:
        formas = [x]
        if 0 <= x <= 1:
            formas.append(100 * x)
        if any(abs(round(f, casas) - alvo) < _TOLERANCIA for f in formas):
            return True
    return False


def _padrao_do_termo(termo: str) -> re.Pattern[str]:
    corpo = _FORMAS.get(termo)
    if corpo is None:
        corpo = r"\s+".join(re.escape(p) for p in _sem_acento(termo).split())
    corpo = f"(?:{corpo})"
    return re.compile(rf"(?<![a-z0-9]){corpo}(?![a-z0-9])")


_PADROES_PROIBIDOS = tuple((t, _padrao_do_termo(t)) for t in TERMOS_PROIBIDOS)


def conferir(texto: str, entrada: dict[str, Any]) -> Conferencia:
    """Reprova número do texto sem correspondente na entrada e termo proibido."""
    motivos: list[str] = []
    candidatos = _numeros_da_entrada(entrada)
    vistos: set[str] = set()
    for token in _NUMERO.findall(texto):
        if token in vistos:
            continue
        vistos.add(token)
        if not _existe_na_entrada(token, candidatos):
            motivos.append(f"número {token} não existe na entrada")
    normalizado = _sem_acento(texto)
    for termo, padrao in _PADROES_PROIBIDOS:
        if padrao.search(normalizado):
            motivos.append(f"termo proibido: {termo}")
    return Conferencia(aprovado=not motivos, motivos=tuple(motivos))
