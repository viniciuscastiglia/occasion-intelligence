"""Testes do registro de experimentos em JSON e do salvamento de resultados."""

import json
from pathlib import Path

import numpy as np
import pytest

from occasion.config import SEMENTE
from occasion.experimentos import registro
from occasion.validacao.bootstrap import Intervalo


def test_registro_grava_campos_obrigatorios(tmp_path: Path) -> None:
    caminho = registro.registrar(
        "exp_teste",
        "confirmatório",
        {"degrau": "B", "C": np.float64(0.5), "dimensoes": ("a", "b")},
        {
            "log_loss": np.float64(0.25),
            "n": np.int64(7),
            "ok": np.bool_(True),
            "vetor": np.array([1.0, 2.0]),
            "ic": Intervalo(0.1, 0.0, 0.2, 1000),
            "nan": float("nan"),
        },
        {"base": "pessoa_dia_2008"},
        pasta=tmp_path,
    )
    assert caminho == tmp_path / "exp_teste.json"
    d = json.loads(caminho.read_text(encoding="utf-8"))
    for chave in (
        "nome", "natureza", "configuracao", "metricas", "dados", "commit", "semente",
        "data_hora", "versoes",
    ):  # fmt: skip
        assert chave in d
    assert d["semente"] == SEMENTE
    assert d["natureza"] == "confirmatório"
    assert set(d["versoes"]) == {"python", "pandas", "scikit-learn", "lightgbm"}
    assert d["metricas"]["ic"] == {
        "estimativa": 0.1,
        "ic_inf": 0.0,
        "ic_sup": 0.2,
        "b": 1000,
        "nivel": 0.95,
    }
    assert d["metricas"]["vetor"] == [1.0, 2.0]
    assert d["metricas"]["nan"] is None
    assert d["metricas"]["ok"] is True
    assert d["configuracao"]["dimensoes"] == ["a", "b"]


def test_registro_cria_a_pasta(tmp_path: Path) -> None:
    caminho = registro.registrar("x", "exploratório", {}, {}, {}, pasta=tmp_path / "novo" / "p")
    assert caminho.exists()


def test_salvar_e_carregar_resultado(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registro, "DADOS_EXPERIMENTOS", tmp_path / "pasta_nova")
    caminho = registro.salvar_resultado("r", {"a": np.arange(3)})
    assert caminho == tmp_path / "pasta_nova" / "r.joblib"
    np.testing.assert_array_equal(registro.carregar_resultado("r")["a"], np.arange(3))


def test_registro_grava_a_semente_usada(tmp_path: Path) -> None:
    padrao = registro.registrar("a", "exploratório", {}, {}, {}, pasta=tmp_path)
    outra = registro.registrar(
        "b", "exploratório", {}, {}, {}, pasta=tmp_path, semente=SEMENTE + 300
    )
    assert json.loads(padrao.read_text(encoding="utf-8"))["semente"] == SEMENTE
    assert json.loads(outra.read_text(encoding="utf-8"))["semente"] == SEMENTE + 300


def test_conjuntos_viram_listas_ordenadas() -> None:
    assert registro._para_json({"b", "a", "c"}) == ["a", "b", "c"]
    assert registro._para_json(frozenset({"SP", "AC", "RJ"})) == ["AC", "RJ", "SP"]
    assert registro._para_json({"x": frozenset({2, 10, 1})}) == {"x": [1, 10, 2]}  # por str
