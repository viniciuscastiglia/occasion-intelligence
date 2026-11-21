"""Diagnóstico depois do teste e H9 em bases sintéticas, sem tocar `data/` nem `reports/`.

As linhas de 2008-2009 levam `n_moradores_uc == 7` (as de 2017-2018 têm no máximo 6): o construtor
espião lê essa marca na entrada de cada `fit` para provar quem entrou no treino.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, ClassVar, Self, cast

import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.bases import acesso
from occasion.bases.acesso import exigir_teste_commitado
from occasion.config import EDICAO_TESTE, EDICAO_TREINO
from occasion.experimentos import pos_teste
from occasion.experimentos.especificacao import variaveis_da_parte
from occasion.experimentos.registro import _para_json
from occasion.models.base import ModeloParte, sigmoide
from occasion.models.calibracao import ModeloCalibrado
from occasion.models.escada import construir
from occasion.receita import ReceitaParte
from occasion.validacao.dobras import (
    atribuir_dobras,
    dobra_das_linhas,
    dobras_internas,
    tabela_de_upas,
)

_GIT = [
    "git", "-c", "user.name=teste", "-c", "user.email=teste@exemplo.invalid",
    "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false",
]  # fmt: skip


# ------------------------------------------------------------------------------ trava


def _repo(pasta: Path) -> Path:
    pasta.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=pasta, check=True)
    subprocess.run([*_GIT, "commit", "-q", "--allow-empty", "-m", "inicial"], cwd=pasta, check=True)
    return pasta


def _commitar(repo: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run([*_GIT, "commit", "-q", "-m", "teste final"], cwd=repo, check=True)


def test_teste_commitado_bloqueia_sem_o_arquivo(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(acesso.TesteBloqueado):
        exigir_teste_commitado(repo)


def test_teste_commitado_bloqueia_com_arquivo_nao_rastreado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    (repo / "reports").mkdir()
    (repo / "reports" / "teste_final.json").write_text("{}", encoding="utf-8")
    with pytest.raises(acesso.TesteBloqueado):
        exigir_teste_commitado(repo)


def test_teste_commitado_bloqueia_com_arquivo_alterado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    (repo / "reports").mkdir()
    arquivo = repo / "reports" / "teste_final.json"
    arquivo.write_text("{}", encoding="utf-8")
    _commitar(repo)
    exigir_teste_commitado(repo)  # commitado e sem alteração: passa
    arquivo.write_text('{"mudou": 1}', encoding="utf-8")
    with pytest.raises(acesso.TesteBloqueado):
        exigir_teste_commitado(repo)


def test_rodar_pos_teste_recusa_sem_teste_commitado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(acesso.TesteBloqueado):
        pos_teste.rodar_pos_teste(raiz=repo)


# ------------------------------------------------------------------------------ bases sintéticas


def _base(n: int, semente: int, edicao: str, tipo: str) -> pd.DataFrame:
    X = contexto_sintetico(n, semente)
    rng = np.random.default_rng(semente + 5)
    X["cod_upa"] = [f"{edicao[:4]}-{uf}-{rng.integers(0, 4)}" for uf in X["uf"]]
    X["estrato"] = X["cod_upa"]
    X["edicao"] = edicao
    X["peso_final"] = rng.uniform(0.5, 3.0, size=n)
    X["n_moradores_uc"] = 7 if edicao == EDICAO_TREINO else rng.integers(1, 7, size=n)
    if tipo == "celulas":
        z = -0.8 + 1.0 * (X["faixa_horario"] == "noite")
        X["y_pizza_refri"] = (rng.random(n) < sigmoide(np.asarray(z, float))).astype(int)
        X["y_pizza_guarana"] = ((rng.random(n) < 0.4) & (X["y_pizza_refri"] == 1)).astype(int)
    else:
        X["tem_pizza"] = (rng.random(n) < 0.5).astype(int)
        z = -0.6 - 0.5 * X["tem_pizza"] + 0.5 * (X["local"] == "fora_de_casa")
        X["y_guarana"] = (rng.random(n) < sigmoide(np.asarray(z, float))).astype(int)
    return X


@pytest.fixture(scope="module")
def bases() -> dict[str, pd.DataFrame]:
    return {
        "celulas08": _base(1500, 1, EDICAO_TREINO, "celulas"),
        "ocasioes08": _base(1200, 2, EDICAO_TREINO, "ocasioes"),
        "celulas17": _base(2500, 3, EDICAO_TESTE, "celulas"),
        "ocasioes17": _base(1800, 4, EDICAO_TESTE, "ocasioes"),
    }


class Espiao:
    """Embrulha um modelo real e anota, em cada `fit`, quantas linhas de 2008-2009 entraram."""

    chamadas: ClassVar[list[tuple[int, int]]] = []

    def __init__(self, real: ModeloParte) -> None:
        self.real = real

    @property
    def variaveis(self) -> list[str]:
        return self.real.variaveis

    @property
    def base_(self) -> float:
        return self.real.base_

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> Self:
        antigas = int(np.sum(X["n_moradores_uc"].to_numpy() == 7))
        Espiao.chamadas.append((antigas, len(X)))
        self.real.fit(X, y, sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.real.predict_proba(X)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.real.contribuicoes(X)


@pytest.fixture
def espiao(monkeypatch: pytest.MonkeyPatch) -> type[Espiao]:
    Espiao.chamadas = []
    monkeypatch.setattr(
        pos_teste,
        "construir",
        lambda degrau, parte, params: Espiao(construir(degrau, parte, params)),
    )
    return Espiao


def _receita_parte(calibrar: bool) -> ReceitaParte:
    return ReceitaParte(degrau="B", parametros={"C": 1.0}, usar_peso=True, calibrar=calibrar)


def _treino17_e_interna(base17: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    upas = tabela_de_upas(base17)
    dobras = atribuir_dobras(upas, 5, 7)
    mapa = dobra_das_linhas(base17, dobras)
    treino = cast(pd.DataFrame, base17[mapa != 0]).reset_index(drop=True)
    return treino, dobras_internas(dobras, 0)


def test_braco_recente_nunca_ve_2008(bases: dict[str, pd.DataFrame], espiao: type[Espiao]) -> None:
    treino17, interna = _treino17_e_interna(bases["celulas17"])
    pos_teste._ajustar_braco(
        "recente", "ocasiao", _receita_parte(True), "y_pizza_refri",
        bases["celulas08"], treino17, interna,
    )  # fmt: skip
    assert espiao.chamadas  # o ajuste final e os ajustes internos da calibração
    assert all(antigas == 0 for antigas, _ in espiao.chamadas)
    assert espiao.chamadas[0][1] == len(treino17)


def test_braco_base_inteira_tem_toda_2008_em_todo_ajuste(
    bases: dict[str, pd.DataFrame], espiao: type[Espiao]
) -> None:
    treino17, interna = _treino17_e_interna(bases["celulas17"])
    pos_teste._ajustar_braco(
        "inteira", "ocasiao", _receita_parte(True), "y_pizza_refri",
        bases["celulas08"], treino17, interna,
    )  # fmt: skip
    n08 = len(bases["celulas08"])
    assert all(antigas == n08 for antigas, _ in espiao.chamadas)
    assert espiao.chamadas[0][1] == n08 + len(treino17)


@pytest.mark.parametrize("braco", ["recente", "inteira"])
def test_calibracao_so_usa_linhas_de_2017(
    bases: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch, braco: str
) -> None:
    treino17, interna = _treino17_e_interna(bases["celulas17"])
    vistos: list[tuple[int, int]] = []
    original = pos_teste.ajustar_sigmoide

    def espia(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
        vistos.append((len(p), len(w)))
        # os pesos recebidos são os `peso_final` das linhas de 2017-2018, na ordem do treino
        assert np.allclose(np.sort(w), np.sort(treino17["peso_final"].to_numpy()))
        return original(p, y, w)

    monkeypatch.setattr(pos_teste, "ajustar_sigmoide", espia)
    modelo = pos_teste._ajustar_braco(
        braco, "ocasiao", _receita_parte(True), "y_pizza_refri",  # type: ignore[arg-type]
        bases["celulas08"], treino17, interna,
    )  # fmt: skip
    assert vistos == [(len(treino17), len(treino17))]
    assert isinstance(modelo, ModeloCalibrado)


def test_sem_calibrar_nao_chama_a_sigmoide(
    bases: dict[str, pd.DataFrame], monkeypatch: pytest.MonkeyPatch
) -> None:
    treino17, interna = _treino17_e_interna(bases["celulas17"])
    monkeypatch.setattr(pos_teste, "ajustar_sigmoide", lambda *a: pytest.fail("não devia calibrar"))
    modelo = pos_teste._ajustar_braco(
        "inteira", "ocasiao", _receita_parte(False), "y_pizza_refri",
        bases["celulas08"], treino17, interna,
    )  # fmt: skip
    assert not isinstance(modelo, ModeloCalibrado)


# ------------------------------------------------------------------------------ o experimento


def _teste_final_ficticio() -> dict[str, Any]:
    def resumo(ll: float) -> dict[str, Any]:
        return {"log_loss": ll, "brier": 0.1, "pr_auc": 0.2, "roc_auc": 0.6, "prevalencia": 0.3}

    nomes = ("ocasiao", "preferencia_todas", "preferencia_com_pizza", "score")
    return {
        "metricas": {
            "resumos": {n: {"receita": resumo(0.5)} for n in nomes},
            "queda": {n: {"receita": {"validacao": resumo(0.4)}} for n in nomes},
        }
    }


def test_executar_grava_h9_e_diagnostico(bases: dict[str, pd.DataFrame]) -> None:
    from occasion.receita import carregar_receita

    receita = carregar_receita().model_copy(
        update={
            "ocasiao": _receita_parte(False),
            "preferencia": _receita_parte(False),
        }
    )
    resultado = pos_teste.executar(
        bases["celulas08"], bases["ocasioes08"], bases["celulas17"], bases["ocasioes17"],
        receita, _teste_final_ficticio(), b=20,
    )  # fmt: skip
    h9 = resultado["H9"]
    assert resultado["H9"]["natureza"] == "confirmatório"
    assert resultado["diagnostico_6_8"]["natureza"] == "exploratório"
    assert set(h9["comparacoes"]) == {"ocasiao", "preferencia", "preferencia_com_pizza"}
    vereditos = [c["veredito"] for c in h9["comparacoes"].values()]
    esperado = "2017-2018" if "padrão refutado" in vereditos else "base_inteira"
    assert h9["treino_servido"] == esperado
    for c in h9["comparacoes"].values():
        assert c["ganho"].ic_inf <= c["ganho"].estimativa <= c["ganho"].ic_sup
        assert np.isclose(c["log_loss_inteira"] - c["log_loss_recente"], c["ganho"].estimativa)
    diag = resultado["diagnostico_6_8"]["partes"]
    assert set(diag) == {"ocasiao", "preferencia_todas", "preferencia_com_pizza", "score"}
    assert np.isclose(
        diag["ocasiao"]["diferenca"]["log_loss"],
        diag["ocasiao"]["recente"]["log_loss"] - 0.5,
    )
    resultado.pop("_dobras")
    json.dumps(_para_json(resultado))  # serializa


@pytest.mark.parametrize(
    ("vereditos", "esperado"),
    [
        (["sem evidência contra o padrão"] * 3, "base_inteira"),
        (["sem evidência contra o padrão", "padrão refutado", "sem evidência contra o padrão"], "2017-2018"),
        (["padrão refutado"] * 3, "2017-2018"),
    ],
)  # fmt: skip
def test_regra_de_h9_qualquer_recusa_leva_a_2017(vereditos: list[str], esperado: str) -> None:
    assert pos_teste.treino_servido(vereditos) == esperado


def test_variaveis_da_receita_usadas(bases: dict[str, pd.DataFrame]) -> None:
    assert "tem_pizza" in variaveis_da_parte("preferencia")
