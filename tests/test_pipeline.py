"""Testes do commit da receita, com repositórios temporários criados no próprio teste."""

import subprocess
from pathlib import Path

import pytest

from occasion.pipeline import commit_da_receita


def _git(pasta: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=pasta,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repositorio(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("um\n")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "commit", "-q", "-m", "inicial")
    return tmp_path


def test_repositorio_limpo_nao_tem_sufixo(repositorio: Path) -> None:
    commit = commit_da_receita(repositorio)
    assert commit and not commit.endswith("-dirty")


def test_repositorio_com_alteracao_tem_sufixo_dirty(repositorio: Path) -> None:
    (repositorio / "a.txt").write_text("dois\n")
    assert commit_da_receita(repositorio).endswith("-dirty")


def test_pasta_sem_git_devolve_desconhecido(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    pasta = tmp_path / "sem_git"
    pasta.mkdir()
    assert commit_da_receita(pasta) == "desconhecido"
    assert "aviso" in capsys.readouterr().err


def test_sem_binario_do_git_devolve_desconhecido(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))
    assert commit_da_receita(tmp_path) == "desconhecido"
