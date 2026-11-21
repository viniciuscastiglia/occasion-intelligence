"""Cliente HTTP do provedor compatível com a OpenAI, testado com transporte falso (sem rede)."""

import json
from collections.abc import Callable

import httpx
import pytest

from occasion.briefing.cliente import (
    ClienteHTTP,
    ConfiguracaoLLM,
    FalhaDoLLM,
    configuracao_de,
)

CONFIG = ConfiguracaoLLM(base_url="https://provedor.exemplo/v1", modelo="modelo-x", chave="segredo")
MENSAGENS = [{"role": "user", "content": "oi"}]


def _cliente(resposta: Callable[[httpx.Request], httpx.Response]) -> ClienteHTTP:
    return ClienteHTTP(CONFIG, transporte=httpx.MockTransport(resposta))


def test_cliente_devolve_conteudo() -> None:
    vistos: list[httpx.Request] = []

    def resposta(req: httpx.Request) -> httpx.Response:
        vistos.append(req)
        return httpx.Response(200, json={"choices": [{"message": {"content": " Texto. "}}]})

    assert _cliente(resposta).completar(MENSAGENS) == "Texto."
    req = vistos[0]
    assert str(req.url) == "https://provedor.exemplo/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer segredo"
    corpo = json.loads(req.content)
    assert corpo["model"] == "modelo-x"
    assert corpo["temperature"] == 0.2
    assert corpo["max_tokens"] == 400
    assert corpo["messages"] == MENSAGENS


def test_base_url_com_barra_final() -> None:
    config = ConfiguracaoLLM(base_url="https://p.exemplo/v1/", modelo="m", chave="c")
    vistos: list[str] = []

    def resposta(req: httpx.Request) -> httpx.Response:
        vistos.append(str(req.url))
        return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

    ClienteHTTP(config, transporte=httpx.MockTransport(resposta)).completar(MENSAGENS)
    assert vistos == ["https://p.exemplo/v1/chat/completions"]


def test_modelo_exposto() -> None:
    assert _cliente(lambda r: httpx.Response(200)).modelo == "modelo-x"


def test_429_vira_cota_esgotada() -> None:
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(429)).completar(MENSAGENS)
    assert erro.value.motivo == "cota_esgotada"


def test_timeout_vira_tempo_esgotado() -> None:
    def resposta(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("lento", request=req)

    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(resposta).completar(MENSAGENS)
    assert erro.value.motivo == "tempo_esgotado"


def test_erro_de_conexao_vira_erro_do_provedor() -> None:
    def resposta(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("sem rede", request=req)

    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(resposta).completar(MENSAGENS)
    assert erro.value.motivo == "erro_do_provedor"


@pytest.mark.parametrize(
    "corpo",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": 3}}]},
        [1],
    ],
)
def test_json_sem_choices_vira_resposta_invalida(corpo: object) -> None:
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(200, json=corpo)).completar(MENSAGENS)
    assert erro.value.motivo == "resposta_invalida"


def test_corpo_que_nao_e_json_vira_resposta_invalida() -> None:
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(200, content=b"<html>")).completar(MENSAGENS)
    assert erro.value.motivo == "resposta_invalida"


@pytest.mark.parametrize("conteudo", ["", "   \n"])
def test_conteudo_vazio_vira_resposta_invalida(conteudo: str) -> None:
    corpo = {"choices": [{"message": {"content": conteudo}}]}
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(200, json=corpo)).completar(MENSAGENS)
    assert erro.value.motivo == "resposta_invalida"


@pytest.mark.parametrize("status", [500, 503, 401, 400])
def test_status_de_erro_vira_erro_do_provedor(status: int) -> None:
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(status)).completar(MENSAGENS)
    assert erro.value.motivo == "erro_do_provedor"


def test_configuracao_de_exige_os_tres_campos() -> None:
    completa = {"BRIEFING_BASE_URL": "https://x/v1", "BRIEFING_MODELO": "m", "BRIEFING_CHAVE": "c"}
    assert configuracao_de(completa) == ConfiguracaoLLM("https://x/v1", "m", "c")
    for faltante in completa:
        parcial = {k: v for k, v in completa.items() if k != faltante}
        assert configuracao_de(parcial) is None
        assert configuracao_de({**completa, faltante: ""}) is None
        assert configuracao_de({**completa, faltante: "  "}) is None


def test_resposta_truncada_vira_resposta_invalida() -> None:
    corpo = {"choices": [{"message": {"content": "Texto cortado"}, "finish_reason": "length"}]}
    with pytest.raises(FalhaDoLLM) as erro:
        _cliente(lambda r: httpx.Response(200, json=corpo)).completar(MENSAGENS)
    assert erro.value.motivo == "resposta_invalida"


def test_finish_reason_stop_e_aceito() -> None:
    corpo = {"choices": [{"message": {"content": "Ok."}, "finish_reason": "stop"}]}
    assert _cliente(lambda r: httpx.Response(200, json=corpo)).completar(MENSAGENS) == "Ok."


@pytest.mark.parametrize("base_url", ["sem-esquema/v1", "", "http://host:porta/v1"])
def test_base_url_malformada_vira_erro_do_provedor(base_url: str) -> None:
    config = ConfiguracaoLLM(base_url=base_url, modelo="m", chave="c")
    cliente = ClienteHTTP(config, transporte=httpx.MockTransport(lambda r: httpx.Response(200)))
    with pytest.raises(FalhaDoLLM) as erro:
        cliente.completar(MENSAGENS)
    assert erro.value.motivo == "erro_do_provedor"
