"""Cliente HTTP de um provedor compatível com a API `chat/completions` da OpenAI."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import httpx

MotivoDaFalha = Literal["tempo_esgotado", "cota_esgotada", "resposta_invalida", "erro_do_provedor"]

TEMPERATURA = 0.2
MAXIMO_DE_TOKENS = 400


@dataclass(frozen=True)
class ConfiguracaoLLM:
    base_url: str
    modelo: str
    chave: str


def configuracao_de(fonte: Mapping[str, str]) -> ConfiguracaoLLM | None:
    """Lê os três `BRIEFING_*`; devolve `None` se faltar algum (ou se estiver vazio)."""
    base_url = fonte.get("BRIEFING_BASE_URL", "").strip()
    modelo = fonte.get("BRIEFING_MODELO", "").strip()
    chave = fonte.get("BRIEFING_CHAVE", "").strip()
    if not (base_url and modelo and chave):
        return None
    return ConfiguracaoLLM(base_url=base_url, modelo=modelo, chave=chave)


class FalhaDoLLM(Exception):
    """Falha da chamada ao provedor; `motivo` diz o tipo, sem expor detalhe da resposta."""

    def __init__(self, motivo: MotivoDaFalha) -> None:
        super().__init__(motivo)
        self.motivo: MotivoDaFalha = motivo


class ClienteLLM(Protocol):
    modelo: str

    def completar(self, mensagens: list[dict[str, str]]) -> str: ...


class ClienteHTTP:
    """Chama `POST {base_url}/chat/completions`. Toda falha vira `FalhaDoLLM`."""

    def __init__(
        self,
        config: ConfiguracaoLLM,
        tempo_limite: float = 15.0,
        transporte: httpx.BaseTransport | None = None,
    ) -> None:
        self._config = config
        self._tempo_limite = tempo_limite
        self._transporte = transporte
        self.modelo = config.modelo

    def completar(self, mensagens: list[dict[str, str]]) -> str:
        url = self._config.base_url.rstrip("/") + "/chat/completions"
        corpo = {
            "model": self._config.modelo,
            "messages": mensagens,
            "temperature": TEMPERATURA,
            "max_tokens": MAXIMO_DE_TOKENS,
        }
        cabecalhos = {"Authorization": f"Bearer {self._config.chave}"}
        try:
            with httpx.Client(timeout=self._tempo_limite, transport=self._transporte) as http:
                resposta = http.post(url, json=corpo, headers=cabecalhos)
        except httpx.TimeoutException as erro:
            raise FalhaDoLLM("tempo_esgotado") from erro
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as erro:
            # URL malformada (sem esquema, porta inválida) não deriva de HTTPError
            raise FalhaDoLLM("erro_do_provedor") from erro
        if resposta.status_code == 429:
            raise FalhaDoLLM("cota_esgotada")
        if resposta.status_code != 200:
            raise FalhaDoLLM("erro_do_provedor")
        return _conteudo(resposta)


def _conteudo(resposta: httpx.Response) -> str:
    try:
        conteudo = resposta.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as erro:
        raise FalhaDoLLM("resposta_invalida") from erro
    if _truncada(resposta):
        raise FalhaDoLLM("resposta_invalida")
    if not isinstance(conteudo, str) or not conteudo.strip():
        raise FalhaDoLLM("resposta_invalida")
    return conteudo.strip()


def _truncada(resposta: httpx.Response) -> bool:
    """Resposta cortada pelo limite de tokens (`finish_reason == "length"`)."""
    return resposta.json()["choices"][0].get("finish_reason") == "length"
