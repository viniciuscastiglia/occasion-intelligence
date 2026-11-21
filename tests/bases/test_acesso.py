"""Testes da porta única de leitura das bases e da trava do alvo de 2017-2018."""

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from occasion.bases import acesso
from occasion.bases.acesso import COLUNAS_DE_ALVO, carregar_base
from occasion.config import TAG_RECEITA

# identidade local e sem assinatura: os commits e tags de teste não dependem da máquina
_GIT = [
    "git",
    "-c",
    "user.name=teste",
    "-c",
    "user.email=teste@exemplo.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "tag.gpgsign=false",
]


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


# ------------------------------------------------------------------ assinatura da receita

_DA_RECEITA = {
    "src/occasion/config.py": "SEMENTE = 1\n",
    "src/occasion/models/taxas.py": "class Tabela: ...\n",
    "src/occasion/validacao/metricas.py": "def resumo(): ...\n",
    "reports/receita.json": '{"degrau": "B"}\n',
    "pyproject.toml": "[project]\n",
    "uv.lock": "versao = 1\n",
}
_FORA_DA_RECEITA = {
    "src/occasion/trends/coleta.py": "def coletar(): ...\n",
    "src/occasion/pipeline.py": "def main(): ...\n",
}


def _escrever(repo: Path, arquivos: dict[str, str]) -> None:
    for caminho, conteudo in arquivos.items():
        destino = repo / caminho
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(conteudo, encoding="utf-8")


def _commitar(repo: Path, mensagem: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run([*_GIT, "commit", "-q", "-m", mensagem], cwd=repo, check=True)


def _repo_da_receita(pasta: Path, arquivos: dict[str, str] | None = None) -> Path:
    repo = _criar_repo(pasta)
    _escrever(repo, arquivos if arquivos is not None else {**_DA_RECEITA, **_FORA_DA_RECEITA})
    _commitar(repo, "receita")
    return repo


def test_arquivos_da_receita_sao_os_do_brief() -> None:
    assert acesso.ARQUIVOS_DA_RECEITA == (
        "src/occasion/config.py",
        "src/occasion/features/",
        "src/occasion/bases/modelagem.py",
        "src/occasion/models/",
        "src/occasion/score/",
        "src/occasion/validacao/",
        "src/occasion/experimentos/especificacao.py",
        "src/occasion/experimentos/teste_final.py",
        "src/occasion/inference/treino.py",
        "src/occasion/inference/predict.py",
        "reports/receita.json",
        "pyproject.toml",
        "uv.lock",
    )


def test_assinatura_nao_depende_da_ordem_de_criacao(tmp_path: Path) -> None:
    direto = _repo_da_receita(tmp_path / "direto")
    invertido = _criar_repo(tmp_path / "invertido")
    for caminho, conteudo in reversed(list({**_DA_RECEITA, **_FORA_DA_RECEITA}.items())):
        _escrever(invertido, {caminho: conteudo})
        _commitar(invertido, caminho)
    assinatura = acesso.assinatura_da_receita(direto)
    assert len(assinatura) == 64
    assert assinatura == acesso.assinatura_da_receita(direto)
    assert assinatura == acesso.assinatura_da_receita(invertido)
    assert assinatura == acesso.assinatura_da_receita(direto, "HEAD")


def test_assinatura_muda_com_conteudo_e_com_nome_de_arquivo(tmp_path: Path) -> None:
    base = acesso.assinatura_da_receita(_repo_da_receita(tmp_path / "base"))
    outro_conteudo = {**_DA_RECEITA, "uv.lock": "versao = 2\n"}
    renomeado = {k.replace("taxas", "tabela"): v for k, v in _DA_RECEITA.items()}
    for nome, arquivos in (("conteudo", outro_conteudo), ("nome", renomeado)):
        assert acesso.assinatura_da_receita(_repo_da_receita(tmp_path / nome, arquivos)) != base


def test_assinatura_ignora_o_que_esta_fora_da_receita(tmp_path: Path) -> None:
    repo = _repo_da_receita(tmp_path / "repo")
    antes = acesso.assinatura_da_receita(repo)
    _escrever(repo, {"src/occasion/trends/coleta.py": "mudou\n", "src/occasion/trends/novo.py": ""})
    assert acesso.assinatura_da_receita(repo) == antes  # sujo e não rastreado, mas fora
    _commitar(repo, "trends")
    assert acesso.assinatura_da_receita(repo) == antes


def test_assinatura_com_ref_le_o_conteudo_do_git(tmp_path: Path) -> None:
    repo = _repo_da_receita(tmp_path / "repo")
    antes = acesso.assinatura_da_receita(repo)
    subprocess.run([*_GIT, "tag", "marco"], cwd=repo, check=True)
    _escrever(repo, {"src/occasion/models/taxas.py": "class Tabela: pass\n"})
    _commitar(repo, "muda o modelo")
    depois = acesso.assinatura_da_receita(repo)
    assert depois != antes
    assert acesso.assinatura_da_receita(repo, "marco") == antes
    # com `ref`, a árvore de trabalho suja não importa
    _escrever(repo, {"src/occasion/models/taxas.py": "sujo\n"})
    assert acesso.assinatura_da_receita(repo, "marco") == antes
    assert acesso.assinatura_da_receita(repo, "HEAD") == depois


@pytest.mark.parametrize(
    "mudanca",
    [
        {"src/occasion/models/taxas.py": "alterado sem commit\n"},
        {"src/occasion/models/novo.py": "não rastreado\n"},
        {"src/occasion/features/nova/variavel.py": "pasta nova não rastreada\n"},
        {"reports/receita.json": '{"degrau": "D"}\n'},
    ],
)
def test_assinatura_sem_ref_recusa_arvore_suja_na_receita(
    tmp_path: Path, mudanca: dict[str, str]
) -> None:
    repo = _repo_da_receita(tmp_path / "repo")
    _escrever(repo, mudanca)
    with pytest.raises(acesso.TesteBloqueado, match="make avaliado"):
        acesso.assinatura_da_receita(repo)


def test_assinatura_sem_ref_recusa_alteracao_so_no_indice_e_arquivo_apagado(
    tmp_path: Path,
) -> None:
    repo = _repo_da_receita(tmp_path / "repo")
    _escrever(repo, {"pyproject.toml": "[project]\nname = 'x'\n"})
    subprocess.run(["git", "add", "pyproject.toml"], cwd=repo, check=True)
    with pytest.raises(acesso.TesteBloqueado):
        acesso.assinatura_da_receita(repo)
    subprocess.run(["git", "reset", "-q", "--hard"], cwd=repo, check=True)
    (repo / "uv.lock").unlink()
    with pytest.raises(acesso.TesteBloqueado):
        acesso.assinatura_da_receita(repo)


def test_assinatura_fora_de_repositorio_levanta(tmp_path: Path) -> None:
    with pytest.raises(acesso.TesteBloqueado):
        acesso.assinatura_da_receita(tmp_path)


def test_exigir_receita_intacta(tmp_path: Path) -> None:
    repo = _repo_da_receita(tmp_path / "repo")
    with pytest.raises(acesso.TesteBloqueado, match=TAG_RECEITA):
        acesso.exigir_receita_intacta(repo)
    subprocess.run([*_GIT, "tag", TAG_RECEITA], cwd=repo, check=True)
    assinatura = acesso.exigir_receita_intacta(repo)
    assert assinatura == acesso.assinatura_da_receita(repo, TAG_RECEITA)

    _escrever(repo, {"src/occasion/trends/coleta.py": "depois da tag\n"})
    _commitar(repo, "trends depois da tag")
    assert acesso.exigir_receita_intacta(repo) == assinatura

    _escrever(repo, {"src/occasion/score/score.py": "CORTE_ALTA = 70\n"})
    _commitar(repo, "score depois da tag")
    with pytest.raises(acesso.TesteBloqueado, match="mudou"):
        acesso.exigir_receita_intacta(repo)
