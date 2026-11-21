"""Teste de contrato da API: o corpo e a resposta seguem o contrato de entrada e saída."""

from fastapi.testclient import TestClient
from tests.inference.test_predict import CONTEXTO_VALIDO, _artefato_unico

from occasion.api.app import criar_app
from occasion.inference.contract import Contexto, Resultado
from occasion.inference.predict import prever

CAMPOS_DO_TAPI = {
    "occasion_score",
    "classification",
    "top_factors",
    "recommendation_category",
    "recommendation_reason",
    "model_metadata",
}


def _cliente() -> TestClient:
    return TestClient(criar_app(_artefato_unico()))


def test_predict_devolve_o_contrato() -> None:
    resposta = _cliente().post("/predict", json=CONTEXTO_VALIDO)
    assert resposta.status_code == 200
    corpo = resposta.json()
    assert CAMPOS_DO_TAPI | {"score_components"} <= set(corpo)
    assert isinstance(corpo["occasion_score"], float)
    assert isinstance(corpo["classification"], str)
    assert isinstance(corpo["top_factors"], list)
    assert isinstance(corpo["recommendation_category"], str)
    assert isinstance(corpo["recommendation_reason"], str)
    assert isinstance(corpo["model_metadata"], dict)
    assert set(corpo["score_components"]) == {"ocasiao", "preferencia"}
    Resultado.model_validate(corpo)


def test_predict_igual_a_prever() -> None:
    artefato = _artefato_unico()
    esperado = prever(Contexto.model_validate(CONTEXTO_VALIDO), artefato)
    resposta = TestClient(criar_app(artefato)).post("/predict", json=CONTEXTO_VALIDO)
    assert Resultado.model_validate(resposta.json()) == esperado


def test_idade_menor_que_18_da_422() -> None:
    resposta = _cliente().post("/predict", json={**CONTEXTO_VALIDO, "idade": 17})
    assert resposta.status_code == 422


def test_uf_invalida_da_422() -> None:
    resposta = _cliente().post("/predict", json={**CONTEXTO_VALIDO, "uf": "XX"})
    assert resposta.status_code == 422


def test_idade_em_texto_nao_numerico_da_422() -> None:
    resposta = _cliente().post("/predict", json={**CONTEXTO_VALIDO, "idade": "vinte"})
    assert resposta.status_code == 422


def test_health_traz_o_modelo() -> None:
    artefato = _artefato_unico()
    resposta = TestClient(criar_app(artefato)).get("/health")
    assert resposta.status_code == 200
    meta = artefato.metadados
    assert resposta.json() == {
        "status": "ok",
        "modelo": meta.modelo,
        "degrau": meta.degrau,
        "treinado_em": meta.treinado_em,
        "receita_commit": meta.receita_commit,
        "versao_contrato": meta.versao_contrato,
    }
