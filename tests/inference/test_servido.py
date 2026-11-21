"""Modelo servido com bases sintéticas e um repositório git em `tmp_path`."""

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from tests.inference.test_treino import _celulas, _ocasioes, _parte, _receita

from occasion.bases import acesso
from occasion.inference import servido
from occasion.inference.contract import MetadadosModelo
from occasion.inference.servido import treinar_servido

_GIT = [
    "git", "-c", "user.name=teste", "-c", "user.email=teste@exemplo.invalid",
    "-c", "commit.gpgsign=false",
]  # fmt: skip
_B = {"degrau": "B", "parametros": {"C": 1.0}, "usar_peso": True}
TESTE_FINAL: dict[str, Any] = {
    "metricas": {
        "H8": {"veredito": "refutada"},
        "partes": {
            "ocasiao": {"veredito": "confirmada"},
            "preferencia_todas": {"veredito": "refutada"},
        },
        "resumos": {
            "score": {
                "receita": {
                    "pr_auc": 0.0019,
                    "log_loss": 0.0022,
                    "roc_auc": 0.89,
                    "brier": 0.00027,
                },
                "degrau_a": {"pr_auc": 0.0021},
            }
        },
    }
}


def _repo(pasta: Path, com_teste: bool, com_pos_teste: bool = True) -> Path:
    pasta.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=pasta, check=True)
    (pasta / "reports").mkdir()
    (pasta / "reports" / "teste_final.json").write_text("{}", encoding="utf-8")
    if com_pos_teste:
        (pasta / "reports" / "pos_teste.json").write_text("{}", encoding="utf-8")
    if com_teste:
        subprocess.run(["git", "add", "-A"], cwd=pasta, check=True)
    subprocess.run([*_GIT, "commit", "-q", "--allow-empty", "-m", "x"], cwd=pasta, check=True)
    return pasta


def _com_setores(base: pd.DataFrame, n_setores: int, edicao: str) -> pd.DataFrame:
    base = base.copy()
    base["edicao"] = edicao
    base["cod_upa"] = [f"{edicao}-{i % n_setores}" for i in range(len(base))]
    base["estrato"] = (base["uf"].astype(int) % 3).astype(str) + "e"
    return base


def _bases(n_setores: int = 40) -> dict[str, pd.DataFrame]:
    c08, o08 = _celulas(300, "2008-2009"), _ocasioes(300, "2008-2009")
    c17, o17 = _celulas(1500, "2017-2018"), _ocasioes(1200, "2017-2018")
    rng = np.random.default_rng(9)
    c17["y_pizza_guarana"] = (rng.random(len(c17)) < 0.15).astype(int)
    return {
        "celulas_2008-2009": _com_setores(c08, n_setores, "2008-2009"),
        "ocasioes_refri_2008-2009": _com_setores(o08, n_setores, "2008-2009"),
        "celulas_2017-2018": _com_setores(c17, n_setores, "2017-2018"),
        "ocasioes_refri_2017-2018": _com_setores(o17, n_setores, "2017-2018"),
    }


def _pos_teste(treino: str) -> dict[str, Any]:
    return {"metricas": {"H9": {"treino_servido": treino}}}


def _receita_simples() -> Any:
    return _receita(
        ocasiao=_parte("B", calibrar=False),
        preferencia=_parte("B", calibrar=False),
        calibrar_score=False,
    )


def test_sem_teste_final_commitado_levanta_teste_bloqueado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=False)
    with pytest.raises(acesso.TesteBloqueado):
        treinar_servido(
            _receita_simples(), _bases(), _pos_teste("2017-2018"), TESTE_FINAL, "abc", repo
        )


def test_sem_pos_teste_commitado_levanta_teste_bloqueado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True, com_pos_teste=False)
    with pytest.raises(acesso.TesteBloqueado, match="pos_teste.json"):
        treinar_servido(
            _receita_simples(), _bases(), _pos_teste("2017-2018"), TESTE_FINAL, "abc", repo
        )


def test_pos_teste_alterado_depois_do_commit_levanta_teste_bloqueado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    (repo / "reports" / "pos_teste.json").write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(acesso.TesteBloqueado, match="pos_teste.json"):
        treinar_servido(
            _receita_simples(), _bases(), _pos_teste("2017-2018"), TESTE_FINAL, "abc", repo
        )


def test_descricao_diz_o_veredito_de_h8_lido_do_json(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    artefato = treinar_servido(
        _receita_simples(), _bases(), _pos_teste("base_inteira"), TESTE_FINAL, "abc", repo
    )
    texto = artefato.metadados.descricao_metricas
    assert "a receita não superou o produto das tabelas de taxas" in texto
    assert "H8 refutada" in texto
    assert "a parte de ocasião superou a tabela" in texto
    assert "preferência não superou a tabela" in texto


def test_descricao_segue_o_veredito_confirmado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    metricas = {**TESTE_FINAL["metricas"], "H8": {"veredito": "confirmada"}}
    artefato = treinar_servido(
        _receita_simples(),
        _bases(),
        _pos_teste("base_inteira"),
        {"metricas": metricas},
        "abc",
        repo,
    )
    texto = artefato.metadados.descricao_metricas
    assert "a receita superou o produto das tabelas de taxas" in texto
    assert "H8 confirmada" in texto


def test_h9_2017_nenhuma_linha_de_2008_entra_no_treino(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    treinos: list[set[str]] = []
    original = servido.ajustar_parte

    def espia(base: pd.DataFrame, *args: Any, **kw: Any) -> Any:
        treinos.append(set(base["edicao"]))
        return original(base, *args, **kw)

    monkeypatch.setattr(servido, "ajustar_parte", espia)
    artefato = treinar_servido(
        _receita_simples(), _bases(), _pos_teste("2017-2018"), TESTE_FINAL, "abc", repo
    )
    assert treinos and all(t == {"2017-2018"} for t in treinos)
    assert artefato.metadados.treinado_em == ["2017-2018"]
    assert artefato.metadados.modelo == "servido"


def test_h9_base_inteira_usa_as_duas_edicoes(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    artefato = treinar_servido(
        _receita_simples(), _bases(), _pos_teste("base_inteira"), TESTE_FINAL, "abc", repo
    )
    assert artefato.metadados.treinado_em == ["2008-2009", "2017-2018"]
    assert "H9" in artefato.metadados.descricao_metricas


def test_metricas_da_receita_iguais_as_do_json(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    artefato = treinar_servido(
        _receita_simples(), _bases(), _pos_teste("base_inteira"), TESTE_FINAL, "abc", repo
    )
    meta: MetadadosModelo = artefato.metadados
    score = TESTE_FINAL["metricas"]["resumos"]["score"]
    assert meta.metricas_da_receita == {
        "pr_auc": score["receita"]["pr_auc"],
        "log_loss": score["receita"]["log_loss"],
        "roc_auc": score["receita"]["roc_auc"],
        "brier": score["receita"]["brier"],
        "pr_auc_degrau_a": score["degrau_a"]["pr_auc"],
    }


@pytest.mark.parametrize("treino", ["base_inteira", "2017-2018"])
def test_referencia_so_com_linhas_de_2017(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, treino: str
) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    vistas: list[set[str]] = []
    original = servido.construir_referencia

    def espia(celulas: pd.DataFrame, *args: Any) -> Any:
        vistas.append(set(celulas["edicao"]))
        return original(celulas, *args)

    monkeypatch.setattr(servido, "construir_referencia", espia)
    treinar_servido(_receita_simples(), _bases(), _pos_teste(treino), TESTE_FINAL, "abc", repo)
    assert vistas == [{"2017-2018"}]


@pytest.mark.parametrize("treino", ["base_inteira", "2017-2018"])
def test_calibracao_so_com_linhas_de_2017(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, treino: str
) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    comprimentos: list[int] = []
    original = servido.ajustar_sigmoide

    def espia(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> Any:
        comprimentos.append(len(p))
        return original(p, y, w)

    monkeypatch.setattr(servido, "ajustar_sigmoide", espia)
    bases = _bases()
    receita = _receita(ocasiao=_parte("B"), preferencia=_parte("B"), calibrar_score=True)
    artefato = treinar_servido(receita, bases, _pos_teste(treino), TESTE_FINAL, "abc", repo)
    n17 = [len(bases["celulas_2017-2018"]), len(bases["ocasioes_refri_2017-2018"])]
    assert sorted(comprimentos) == sorted([n17[0], n17[1], n17[0]])
    assert artefato.calibracao_score is not None and artefato.calibracao_score[0] > 0


def test_direto_e_reajustado_e_calibrado(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r", com_teste=True)
    a_ocasiao = {"dimensoes": ["faixa_horario", "local"], "suavizacao": 100}
    receita = _receita(
        ocasiao=_parte("B", calibrar=False),
        preferencia=_parte("B", calibrar=False),
        score="direto",
        direto=_parte("B"),
        calibrar_score=True,
        referencia_a={
            "ocasiao": a_ocasiao,
            "preferencia": {"dimensoes": ["regiao"], "suavizacao": 1000},
            "direto": a_ocasiao,
        },
    )
    artefato = treinar_servido(receita, _bases(), _pos_teste("2017-2018"), TESTE_FINAL, "abc", repo)
    assert artefato.modelo_direto is not None
    assert artefato.calibracao_score is not None
