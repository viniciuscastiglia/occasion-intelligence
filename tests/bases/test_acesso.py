"""Testes da porta única de leitura das bases e da trava do alvo de 2017-2018."""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from occasion.bases import acesso
from occasion.bases.acesso import COLUNAS_DE_ALVO, carregar_base
from occasion.config import TAG_RECEITA

_GIT = ["git", "-c", "user.name=teste", "-c", "user.email=teste@exemplo.invalid"]


def _criar_repo(pasta: Path) -> Path:
    pasta.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=pasta, check=True)
    subprocess.run([*_GIT, "commit", "-q", "--allow-empty", "-m", "inicial"], cwd=pasta, check=True)
    return pasta


@pytest.fixture
def repo_sem_tag(tmp_path: Path) -> Path:
    return _criar_repo(tmp_path / "sem_tag")


@pytest.fixture
def repo_com_tag(tmp_path: Path) -> Path:
    repo = _criar_repo(tmp_path / "com_tag")
    subprocess.run([*_GIT, "tag", TAG_RECEITA], cwd=repo, check=True)
    return repo


@pytest.fixture
def pasta_bases(tmp_path: Path) -> Path:
    pasta = tmp_path / "bases"
    pasta.mkdir()
    for edicao in ("2008-2009", "2017-2018"):
        pd.DataFrame(
            {
                "uf": ["SP", "RJ"],
                "peso_final": [1.0, 2.0],
                "y_pizza_refri": [0, 1],
                "y_pizza_guarana": [0, 0],
            }
        ).to_parquet(pasta / f"celulas_{edicao}.parquet")
        pd.DataFrame(
            {"uf": ["SP", "RJ"], "peso_final": [1.0, 2.0], "y_guarana": [0, 1], "tem_pizza": [1, 0]}
        ).to_parquet(pasta / f"ocasioes_refri_{edicao}.parquet")
    return pasta


def test_alvo_de_2017_bloqueado_sem_tag(repo_sem_tag: Path, pasta_bases: Path) -> None:
    with pytest.raises(acesso.TesteBloqueado):
        carregar_base("celulas", "2017-2018", pasta=pasta_bases, raiz=repo_sem_tag)


def test_contexto_de_2017_sai_sem_colunas_de_alvo(repo_sem_tag: Path, pasta_bases: Path) -> None:
    df = carregar_base(
        "ocasioes_refri", "2017-2018", so_contexto=True, pasta=pasta_bases, raiz=repo_sem_tag
    )
    assert not (set(df.columns) & COLUNAS_DE_ALVO)
    assert {"uf", "peso_final"} <= set(df.columns)


def test_alvo_de_2017_liberado_com_tag(repo_com_tag: Path, pasta_bases: Path) -> None:
    df = carregar_base("celulas", "2017-2018", pasta=pasta_bases, raiz=repo_com_tag)
    assert "y_pizza_guarana" in df.columns


def test_2008_sempre_liberado(repo_sem_tag: Path, pasta_bases: Path) -> None:
    df = carregar_base("ocasioes_refri", "2008-2009", pasta=pasta_bases, raiz=repo_sem_tag)
    assert "y_guarana" in df.columns


def test_teste_liberado_reflete_a_tag(repo_sem_tag: Path, repo_com_tag: Path) -> None:
    assert not acesso.teste_liberado(repo_sem_tag)
    assert acesso.teste_liberado(repo_com_tag)


def test_teste_liberado_sem_git_ou_fora_de_repo_e_falso(tmp_path: Path) -> None:
    assert not acesso.teste_liberado(tmp_path / "inexistente")
    assert not acesso.teste_liberado(tmp_path)
