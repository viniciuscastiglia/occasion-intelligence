"""Teste de integração das etapas de experimento em base sintética, sem tocar `data/` nem `reports/`."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.config import EDICAO_TREINO
from occasion.experimentos import etapas, registro
from occasion.experimentos.especificacao import espaco_lightgbm
from occasion.models import escada
from occasion.models.base import sigmoide

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


def test_dobras_peso_e_escada_da_preferencia(pastas: Path) -> None:
    etapas.etapa_dobras()
    dobras = _json(pastas / "reports" / "dobras.json")
    assert all(c in dobras for c in CHAVES)
    assert (pastas / "experimentos" / "dobras.parquet").exists()

    etapas.etapa_peso()
    h5 = _json(pastas / "reports" / "experimentos" / "h5_peso.json")
    assert all(c in h5 for c in CHAVES)
    assert set(h5["metricas"]) == {"ocasiao", "preferencia"}
    assert isinstance(h5["metricas"]["preferencia"]["usar_peso"], bool)

    etapas.etapa_escada("preferencia")
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert all(c in e for c in CHAVES)
    m = e["metricas"]
    assert set(m["degraus"]) == {"A", "B", "C", "D"}
    for d in m["degraus"].values():
        assert d["tempo_s"] >= 0
        assert len(d["parametros_por_dobra"]) == 5
        assert d["resumo"]["log_loss"] > 0
        assert d["resumo_com_pizza"]["n"] > 0
    assert {"H1", "H2", "H3"} <= set(m["comparacoes"])
    assert m["vencedor"] in {"A", "B", "C", "D"}
    assert m["comparacoes"]["H3"]["referencia"] in {"B", "C"}
    # sem escrita fora do tmp_path: os resultados aninhados estão na pasta redirecionada
    assert (pastas / "experimentos" / "aninhada_preferencia_D.joblib").exists()
    # o B da escada é o B de H5, não um recálculo
    assert (pastas / "experimentos" / "aninhada_preferencia_B.joblib").exists()


def test_h4_da_preferencia(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_h4(b=20)
    h4 = _json(pastas / "reports" / "experimentos" / "h4_tem_pizza.json")
    assert all(c in h4 for c in CHAVES)
    assert h4["metricas"]["veredito"] in {
        "confirmada",
        "efeito no sentido contrário",
        "os dados não mostram o efeito",
    }
    assert h4["configuracao"]["C"] > 0
    assert "regulariza" in h4["configuracao"]["aviso"]
    assert h4["metricas"]["intervalo"]["b"] > 0


def test_embaralhado_em_base_sintetica(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_embaralhado(partes=("preferencia",), usar_peso={"preferencia": True})
    r = _json(pastas / "reports" / "experimentos" / "rotulos_embaralhados.json")
    assert all(c in r for c in CHAVES)
    assert r["metricas"]["preferencia"]["veredito"] in {"confirmada", "refutada"}


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
