"""Serviço do briefing: cache, limite por sessão, conferência e texto reserva (sem rede)."""

from collections.abc import Callable

import pytest

from occasion.briefing.cliente import FalhaDoLLM
from occasion.briefing.limite import LimiteGlobal
from occasion.briefing.reserva import texto_reserva
from occasion.briefing.servico import (
    LIMITE_POR_SESSAO,
    MOTIVO_LIMITE_GLOBAL,
    TAMANHO_MAXIMO_DO_CACHE,
    Briefing,
    chave_do_cache,
    gerar_briefing,
)
from occasion.inference.contract import Contexto, Resultado

TEXTO_BOM = "O Occasion Score é 87 de 100 e a categoria é converter."
TEXTO_RUIM = "O Occasion Score é 55 de 100."


class ClienteFalso:
    modelo = "modelo-falso"

    def __init__(self, texto: str = TEXTO_BOM, falha: str | None = None) -> None:
        self.texto = texto
        self.falha = falha
        self.chamadas = 0

    def completar(self, mensagens: list[dict[str, str]]) -> str:
        self.chamadas += 1
        if self.falha:
            raise FalhaDoLLM(self.falha)  # type: ignore[arg-type]
        return self.texto


def test_sem_cliente_usa_reserva_com_motivo(contexto: Contexto, resultado: Resultado) -> None:
    cache: dict[str, Briefing] = {}
    b, chamou = gerar_briefing(contexto, resultado, None, cache, 0)
    assert (b.fonte, b.modelo, b.motivo) == ("reserva", None, "sem chave configurada")
    assert b.texto == texto_reserva(contexto, resultado)
    assert chamou is False
    assert cache == {}


def test_texto_aprovado_vem_do_llm_e_entra_no_cache(
    contexto: Contexto, resultado: Resultado
) -> None:
    cliente, cache = ClienteFalso(), {}
    b, chamou = gerar_briefing(contexto, resultado, cliente, cache, 0)
    assert (b.fonte, b.modelo, b.motivo, b.texto) == ("llm", "modelo-falso", None, TEXTO_BOM)
    assert chamou is True
    assert cache[chave_do_cache(contexto, resultado)] == b


def test_segunda_chamada_igual_nao_chama_o_provedor(
    contexto: Contexto, resultado: Resultado
) -> None:
    cliente, cache = ClienteFalso(), {}
    primeiro, _ = gerar_briefing(contexto, resultado, cliente, cache, 0)
    segundo, chamou = gerar_briefing(contexto, resultado, cliente, cache, 1)
    assert segundo == primeiro
    assert chamou is False
    assert cliente.chamadas == 1


def test_limite_corta_a_sexta_chamada(contexto: Contexto, resultado: Resultado) -> None:
    cliente = ClienteFalso()
    b, chamou = gerar_briefing(contexto, resultado, cliente, {}, LIMITE_POR_SESSAO)
    assert LIMITE_POR_SESSAO == 5
    assert (b.fonte, b.motivo) == ("reserva", "limite de briefings desta sessão atingido")
    assert chamou is False
    assert cliente.chamadas == 0


def test_cache_atende_mesmo_com_limite_atingido(contexto: Contexto, resultado: Resultado) -> None:
    cliente, cache = ClienteFalso(), {}
    gerar_briefing(contexto, resultado, cliente, cache, 0)
    b, chamou = gerar_briefing(contexto, resultado, cliente, cache, LIMITE_POR_SESSAO)
    assert b.fonte == "llm"
    assert chamou is False


def test_texto_reprovado_vira_reserva_e_nao_entra_no_cache(
    contexto: Contexto, resultado: Resultado
) -> None:
    cliente, cache = ClienteFalso(TEXTO_RUIM), {}
    b, chamou = gerar_briefing(contexto, resultado, cliente, cache, 0)
    assert b.fonte == "reserva"
    assert b.motivo is not None
    assert b.motivo.startswith("texto reprovado na conferência: ")
    assert "55" in b.motivo
    assert b.texto == texto_reserva(contexto, resultado)
    assert chamou is True
    assert cache == {}


@pytest.mark.parametrize(
    ("falha", "motivo"),
    [
        ("tempo_esgotado", "tempo esgotado no provedor"),
        ("cota_esgotada", "cota do provedor esgotada"),
        ("resposta_invalida", "resposta inválida do provedor"),
        ("erro_do_provedor", "erro do provedor"),
    ],
)
def test_falha_do_llm_vira_reserva_com_motivo(
    contexto: Contexto, resultado: Resultado, falha: str, motivo: str
) -> None:
    cache: dict[str, Briefing] = {}
    b, chamou = gerar_briefing(contexto, resultado, ClienteFalso(falha=falha), cache, 0)
    assert (b.fonte, b.modelo, b.motivo) == ("reserva", None, motivo)
    assert chamou is True
    assert cache == {}


def test_chave_do_cache_muda_com_o_resultado(
    contexto: Contexto, montar_resultado: Callable[[str], Resultado]
) -> None:
    a = chave_do_cache(contexto, montar_resultado("Não priorizar"))
    b = chave_do_cache(contexto, montar_resultado("Sustentar, via ação na pizzaria"))
    assert a != b
    assert a == chave_do_cache(contexto, montar_resultado("Não priorizar"))


def test_cache_descarta_a_entrada_mais_antiga(
    contexto: Contexto, resultado: Resultado, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("occasion.briefing.servico.TAMANHO_MAXIMO_DO_CACHE", 2)
    assert TAMANHO_MAXIMO_DO_CACHE == 1000
    cache: dict[str, Briefing] = {
        "a": Briefing("x", "llm", "m", None),
        "b": Briefing("y", "llm", "m", None),
    }
    gerar_briefing(contexto, resultado, ClienteFalso(), cache, 0)
    assert "a" not in cache
    assert len(cache) == 2
    assert "b" in cache


class CacheComChaveJaRemovida(dict[str, Briefing]):
    """Simula outra sessão que removeu a chave mais antiga entre a leitura e a remoção."""

    def __init__(self, *args: object) -> None:
        super().__init__(*args)  # type: ignore[arg-type]
        self.remover_na_leitura = True

    def __iter__(self):  # type: ignore[override]
        chaves = list(super().__iter__())
        if self.remover_na_leitura and chaves:
            self.remover_na_leitura = False
            super().__delitem__(chaves[0])
        return iter(chaves)


def test_descarte_tolera_chave_ja_removida_por_outra_sessao(
    contexto: Contexto, resultado: Resultado, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("occasion.briefing.servico.TAMANHO_MAXIMO_DO_CACHE", 2)
    cache = CacheComChaveJaRemovida(
        {"a": Briefing("x", "llm", "m", None), "b": Briefing("y", "llm", "m", None)}
    )
    b, chamou = gerar_briefing(contexto, resultado, ClienteFalso(), cache, 0)
    assert (b.fonte, chamou) == ("llm", True)
    assert "a" not in cache
    assert len(cache) == 2


def test_limite_global_corta_depois_de_n_chamadas_entre_sessoes(
    contexto: Contexto, montar_resultado: Callable[[str], Resultado]
) -> None:
    limite = LimiteGlobal(maximo=2, relogio=lambda: 0.0)
    cliente, cache = ClienteFalso(), {}
    categorias = ["Não priorizar", "Sustentar, via ação na pizzaria", "Converter"]
    saidas = [
        # Cada chamada vem de uma sessão nova (zero chamadas na sessão).
        gerar_briefing(contexto, montar_resultado(c), cliente, cache, 0, limite)
        for c in categorias
    ]
    assert [s[0].fonte for s in saidas] == ["llm", "llm", "reserva"]
    assert saidas[2][0].motivo == MOTIVO_LIMITE_GLOBAL
    assert saidas[2][1] is False
    assert cliente.chamadas == 2


def test_cache_nao_gasta_o_limite_global(contexto: Contexto, resultado: Resultado) -> None:
    limite = LimiteGlobal(maximo=1, relogio=lambda: 0.0)
    cliente, cache = ClienteFalso(), {}
    gerar_briefing(contexto, resultado, cliente, cache, 0, limite)
    for _ in range(3):
        b, chamou = gerar_briefing(contexto, resultado, cliente, cache, 0, limite)
        assert (b.fonte, chamou) == ("llm", False)
    assert limite.usadas() == 1
    assert cliente.chamadas == 1


def test_limite_da_sessao_vem_antes_e_nao_gasta_o_global(
    contexto: Contexto, resultado: Resultado
) -> None:
    limite = LimiteGlobal(maximo=1, relogio=lambda: 0.0)
    b, _ = gerar_briefing(contexto, resultado, ClienteFalso(), {}, LIMITE_POR_SESSAO, limite)
    assert b.motivo == "limite de briefings desta sessão atingido"
    assert limite.usadas() == 0
