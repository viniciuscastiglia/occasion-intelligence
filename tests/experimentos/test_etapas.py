"""Teste de integração das etapas de experimento em base sintética, sem tocar `data/` nem `reports/`."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.config import EDICAO_TREINO, SEMENTE
from occasion.experimentos import etapas, registro
from occasion.experimentos.especificacao import espaco_lightgbm
from occasion.experimentos.regras import melhor_logistico, padrao, sinal_negativo, superioridade
from occasion.models import escada
from occasion.models.base import sigmoide
from occasion.validacao.bootstrap import Intervalo
from occasion.validacao.dobras import dobra_das_linhas
from occasion.validacao.metricas import log_loss_ponderado

CHAVES = ("configuracao", "metricas", "commit")


def _com_setores(n: int, semente: int) -> pd.DataFrame:
    X = contexto_sintetico(n, semente)
    rng = np.random.default_rng(semente + 5)
    X["cod_upa"] = [f"{uf}-{rng.integers(0, 5)}" for uf in X["uf"]]
    X["estrato"] = X["cod_upa"]
    X["peso_final"] = rng.uniform(0.5, 3.0, size=n)
    return X


def _celulas() -> pd.DataFrame:
    X = _com_setores(8000, 0)
    rng = np.random.default_rng(1)
    z = -2.2 + 1.2 * (X["faixa_horario"] == "noite") - 0.02 * (X["idade"] - 40)
    X["y_pizza_refri"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    X["y_pizza_guarana"] = ((rng.random(len(X)) < 0.35) & (X["y_pizza_refri"] == 1)).astype(int)
    return X


def _ocasioes() -> pd.DataFrame:
    X = _com_setores(3000, 2)
    rng = np.random.default_rng(3)
    X["tem_pizza"] = (rng.random(len(X)) < 0.4).astype(int)
    z = -1.0 - 0.6 * X["tem_pizza"] + 0.5 * (X["local"] == "fora_de_casa")
    X["y_guarana"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    return X


@pytest.fixture
def pastas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aponta as três pastas das etapas para `tmp_path`."""
    processados = tmp_path / "processed"
    processados.mkdir()
    _celulas().to_parquet(processados / f"celulas_{EDICAO_TREINO}.parquet")
    _ocasioes().to_parquet(processados / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    for modulo in (etapas, registro):
        monkeypatch.setattr(modulo, "DADOS_EXPERIMENTOS", tmp_path / "experimentos")
        monkeypatch.setattr(modulo, "RELATORIOS", tmp_path / "reports")
    monkeypatch.setattr(etapas, "DADOS_PROCESSADOS", processados)
    # o espaço do D, só neste teste, tem duas configurações
    monkeypatch.setattr(escada, "espaco_lightgbm", lambda: espaco_lightgbm()[:2])
    return tmp_path


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _ic(d: dict) -> Intervalo:
    return Intervalo(**d)


def test_dobras_peso_e_escada_da_preferencia(pastas: Path) -> None:
    # as pastas lidas e gravadas pelas etapas são as de `tmp_path`, nunca as do repositório
    assert etapas._pasta_relatorios().is_relative_to(pastas)
    assert etapas.DADOS_EXPERIMENTOS.is_relative_to(pastas)
    assert registro.DADOS_EXPERIMENTOS.is_relative_to(pastas)

    etapas.etapa_dobras()
    dobras = _json(pastas / "reports" / "dobras.json")
    assert all(c in dobras for c in CHAVES)
    assert (pastas / "experimentos" / "dobras.parquet").exists()
    assert dobras["semente"] == dobras["metricas"]["semente_usada"]
    assert dobras["metricas"]["status"] == "ok"
    hash_dobras = dobras["dados"]["hash_dobras"]

    etapas.etapa_peso()
    h5 = _json(pastas / "reports" / "experimentos" / "h5_peso.json")
    assert all(c in h5 for c in CHAVES)
    assert h5["metricas"]["status"] == "ok"
    assert {"ocasiao", "preferencia"} <= set(h5["metricas"])
    assert h5["dados"]["dobras"] == {"hash": hash_dobras, "semente": dobras["semente"]}
    for parte in ("ocasiao", "preferencia"):
        m = h5["metricas"][parte]
        # o veredito segue a regra aplicada ao intervalo gravado
        assert m["usar_peso"] == (padrao(_ic(m["ganho_sem_peso"])) != "padrão refutado")

    etapas.etapa_escada("preferencia")
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert all(c in e for c in CHAVES)
    assert e["dados"]["dobras"]["hash"] == hash_dobras
    m = e["metricas"]
    assert m["status"] == "ok"
    assert set(m["degraus"]) == {"A", "B", "C", "D"}
    for d in m["degraus"].values():
        assert d["tempo_s"] >= 0
        assert len(d["parametros_por_dobra"]) == 5
        assert d["resumo"]["log_loss"] > 0
        assert d["resumo_com_pizza"]["n"] > 0
    for h in m["comparacoes"].values():
        assert h["veredito"] == superioridade(_ic(h["intervalo"]))
    assert m["comparacoes"]["H1"]["candidato"] == "B"
    assert m["comparacoes"]["H1"]["referencia"] == "A"
    assert m["comparacoes"]["H3"]["referencia"] == melhor_logistico(
        _ic(m["comparacoes"]["H2"]["intervalo"])
    )
    confirmados = [
        c["candidato"] for c in m["comparacoes_sequenciais"] if c["veredito"] == "confirmada"
    ]
    assert m["vencedor"] == (confirmados[-1] if confirmados else "A")
    assert (pastas / "experimentos" / "aninhada_preferencia_D.joblib").exists()
    # o B da escada é o B de H5 (o resultado salvo por H5, com o hash das dobras)
    assert (pastas / "experimentos" / "aninhada_preferencia_B.joblib").exists()


def test_escada_falha_registra_o_parcial_e_relanca(
    pastas: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    real = etapas.validacao_aninhada

    def com_falha(base, alvo, degrau, *args, **kwargs):  # type: ignore[no-untyped-def]
        if degrau == "C":
            raise RuntimeError("falha forçada no degrau C")
        return real(base, alvo, degrau, *args, **kwargs)

    monkeypatch.setattr(etapas, "validacao_aninhada", com_falha)
    with pytest.raises(RuntimeError, match="forçada"):
        etapas.etapa_escada("preferencia")
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert all(c in e for c in CHAVES)
    assert e["metricas"]["status"] == "falhou"
    assert e["metricas"]["erro"] == {
        "tipo": "RuntimeError",
        "mensagem": "falha forçada no degrau C",
    }
    assert set(e["metricas"]["degraus_concluidos"]) == {"A", "B"}
    assert e["configuracao"]["parte"] == "preferencia"


def test_h5_com_previsao_nan_registra_a_falha(
    pastas: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    etapas.etapa_dobras()

    def nan(p: np.ndarray, o_que: str) -> np.ndarray:
        raise ValueError(f"previsão fora da dobra com NaN em {o_que}")

    monkeypatch.setattr(etapas, "_sem_nan", nan)
    with pytest.raises(ValueError, match="NaN"):
        etapas.etapa_peso()
    h5 = _json(pastas / "reports" / "experimentos" / "h5_peso.json")
    assert h5["metricas"]["status"] == "falhou"
    assert h5["metricas"]["erro"]["tipo"] == "ValueError"
    # com H5 falho, as etapas seguintes recusam
    with pytest.raises(ValueError, match="falhou"):
        etapas.etapa_escada("preferencia")


def test_escada_recusa_dobras_diferentes_das_de_h5(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    caminho = pastas / "experimentos" / "dobras.parquet"
    dobras = pd.read_parquet(caminho)
    dobras["dobra"] = (dobras["dobra"] + 1) % 5
    dobras.to_parquet(caminho)
    with pytest.raises(ValueError, match="dobras"):
        etapas.etapa_escada("preferencia")
    with pytest.raises(ValueError, match="dobras"):
        etapas.etapa_h4(b=5)
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert e["metricas"]["status"] == "falhou"


def test_h4_da_preferencia(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_h4(b=20)
    h4 = _json(pastas / "reports" / "experimentos" / "h4_tem_pizza.json")
    assert all(c in h4 for c in CHAVES)
    m = h4["metricas"]
    assert m["status"] == "ok"
    assert m["veredito"] == sinal_negativo(_ic(m["intervalo"]))
    assert m["razao_de_chances"] == pytest.approx(np.exp(m["coeficiente_tem_pizza"]))
    assert m["intervalo"]["ic_inf"] <= m["intervalo"]["ic_sup"]
    assert m["intervalo"]["b"] > 0
    assert h4["configuracao"]["C"] > 0
    assert "regulariza" in h4["configuracao"]["aviso"]
    assert (
        h4["dados"]["dobras"]["hash"]
        == _json(pastas / "reports" / "dobras.json")["dados"]["hash_dobras"]
    )


def test_embaralhado_em_base_sintetica(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_embaralhado(partes=("preferencia",), usar_peso={"preferencia": True})
    r = _json(pastas / "reports" / "experimentos" / "rotulos_embaralhados.json")
    assert all(c in r for c in CHAVES)
    m = r["metricas"]["preferencia"]
    assert r["metricas"]["status"] == "ok"
    assert m["veredito"] == superioridade(_ic(m["ganho_do_modelo_sobre_a_constante"]))
    # a constante é a prevalência ponderada do treino de cada dobra, conferida à parte
    base = pd.read_parquet(pastas / "processed" / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    y = np.random.default_rng(SEMENTE).permutation(base["y_guarana"].to_numpy())
    w = base["peso_final"].to_numpy()
    por_linha = dobra_das_linhas(base, pd.read_parquet(pastas / "experimentos" / "dobras.parquet"))
    p = np.empty(len(base))
    for k in range(5):
        dentro = por_linha != k
        p[~dentro] = np.sum(w[dentro] * y[dentro]) / np.sum(w[dentro])
    assert m["log_loss_constante"] == pytest.approx(log_loss_ponderado(y, p, w))


@pytest.mark.dados
def test_rotulos_embaralhados_na_preferencia_nao_vencem_a_constante(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for modulo in (etapas, registro):
        monkeypatch.setattr(modulo, "DADOS_EXPERIMENTOS", tmp_path / "experimentos")
        monkeypatch.setattr(modulo, "RELATORIOS", tmp_path / "reports")
    etapas.etapa_dobras()
    etapas.etapa_embaralhado(partes=("preferencia",), usar_peso={"preferencia": True})
    r = _json(tmp_path / "reports" / "experimentos" / "rotulos_embaralhados.json")
    assert r["metricas"]["preferencia"]["veredito"] == "refutada"
