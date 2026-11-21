"""Testes da porta única de leitura das bases e da trava do alvo de 2017-2018."""

import ast
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from occasion.bases import acesso, assinatura
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
    "src/occasion/versoes_da_receita.py": 'VERSOES_DA_RECEITA = {"numpy": "1"}\n',
}
_FORA_DA_RECEITA = {
    "src/occasion/trends/coleta.py": "def coletar(): ...\n",
    "src/occasion/pipeline.py": "def main(): ...\n",
    "pyproject.toml": "[project]\n",
    "uv.lock": "versao = 1\n",
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
    # o brief, mais as Rulings R-p (regras, receita, avaliação, registro), R-q (assinatura.py),
    # R-s (definição do alvo e faixas dos segmentos) e R-t (versões fixadas e `artefato.py` no
    # lugar de `pyproject.toml`, `uv.lock` e `predict.py`)
    assert acesso.ARQUIVOS_DA_RECEITA == (
        "src/occasion/config.py",
        "src/occasion/versoes_da_receita.py",
        "src/occasion/ingest/",
        "src/occasion/bases/ocasioes.py",
        "src/occasion/features/",
        "src/occasion/bases/modelagem.py",
        "src/occasion/bases/assinatura.py",
        "src/occasion/eda/descritivas.py",
        "src/occasion/models/",
        "src/occasion/score/",
        "src/occasion/validacao/",
        "src/occasion/avaliacao/",
        "src/occasion/receita.py",
        "src/occasion/experimentos/especificacao.py",
        "src/occasion/experimentos/regras.py",
        "src/occasion/experimentos/registro.py",
        "src/occasion/experimentos/teste_final.py",
        "src/occasion/inference/treino.py",
        "src/occasion/inference/artefato.py",
        "reports/receita.json",
    )
    assert acesso.ARQUIVOS_DA_RECEITA is assinatura.ARQUIVOS_DA_RECEITA


def test_acesso_reexporta_a_assinatura() -> None:
    assert acesso.TesteBloqueado is assinatura.TesteBloqueado
    assert acesso.assinatura_da_receita is assinatura.assinatura_da_receita
    assert acesso.exigir_receita_intacta is assinatura.exigir_receita_intacta
    assert acesso.teste_liberado is assinatura.teste_liberado
    assert "src/occasion/bases/acesso.py" not in acesso.ARQUIVOS_DA_RECEITA


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
    outro_conteudo = {**_DA_RECEITA, "reports/receita.json": '{"degrau": "C"}\n'}
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
    _escrever(repo, {"src/occasion/config.py": "SEMENTE = 2\n"})
    subprocess.run(["git", "add", "src/occasion/config.py"], cwd=repo, check=True)
    with pytest.raises(acesso.TesteBloqueado):
        acesso.assinatura_da_receita(repo)
    subprocess.run(["git", "reset", "-q", "--hard"], cwd=repo, check=True)
    (repo / "src/occasion/versoes_da_receita.py").unlink()
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


@pytest.mark.parametrize(
    ("arquivo", "bloqueia"),
    [
        ("src/occasion/bases/assinatura.py", True),
        ("src/occasion/experimentos/regras.py", True),
        ("src/occasion/receita.py", True),
        ("src/occasion/avaliacao/segmentos.py", True),
        ("src/occasion/experimentos/registro.py", True),
        # R-s: a definição do alvo e as faixas dos segmentos
        ("src/occasion/bases/ocasioes.py", True),
        ("src/occasion/ingest/layouts.py", True),
        ("src/occasion/ingest/parse.py", True),
        ("src/occasion/ingest/schema.py", True),
        ("src/occasion/eda/descritivas.py", True),
        # R-t: as versões fixadas e o artefato, no lugar das dependências e da resposta da tela
        ("src/occasion/versoes_da_receita.py", True),
        ("src/occasion/inference/artefato.py", True),
        ("src/occasion/bases/acesso.py", False),
        ("src/occasion/inference/predict.py", False),
        ("src/occasion/inference/contract.py", False),
        ("pyproject.toml", False),
        ("uv.lock", False),
    ],
)
def test_guarda_e_codigo_que_mede_depois_da_tag(
    tmp_path: Path, arquivo: str, bloqueia: bool
) -> None:
    """Mudar a assinatura ou o código que mede depois da tag bloqueia; mudar `acesso.py`, não."""
    repo = _repo_da_receita(tmp_path / "repo", {**_DA_RECEITA, arquivo: "antes\n"})
    subprocess.run([*_GIT, "tag", TAG_RECEITA], cwd=repo, check=True)
    assinatura_da_tag = acesso.exigir_receita_intacta(repo)
    _escrever(repo, {arquivo: "depois da tag\n"})
    _commitar(repo, f"muda {arquivo}")
    if bloqueia:
        with pytest.raises(acesso.TesteBloqueado, match="mudou"):
            acesso.exigir_receita_intacta(repo)
    else:
        assert acesso.exigir_receita_intacta(repo) == assinatura_da_tag


# ------------------------------------------------------------------ fechamento da assinatura

_RAIZ_REAL = Path(__file__).resolve().parents[2]
# Módulos fora da assinatura que os arquivos assinados podem importar, e por quê.
_IMPORTACOES_FORA_PERMITIDAS: dict[str, str] = {
    "src/occasion/bases/acesso.py": "a porta de leitura das bases; a tarefa 17 a modifica, e as "
    "contagens de referência do teste final pegam uma leitura de outro arquivo",
    "src/occasion/inference/contract.py": "só tipos do contrato, `MetadadosModelo` e a versão do "
    "contrato; nenhum número sai dele",
    "src/occasion/pipeline.py": "só `commit_da_receita`, para registro",
}


def _assinado(relativo: str) -> bool:
    return any(
        relativo == a or (a.endswith("/") and relativo.startswith(a))
        for a in acesso.ARQUIVOS_DA_RECEITA
    )


def _arquivo_do_modulo(modulo: str) -> str | None:
    base = Path("src", *modulo.split("."))
    for candidato in (base.with_suffix(".py"), base / "__init__.py"):
        if (_RAIZ_REAL / candidato).exists():
            return candidato.as_posix()
    return None


def _importados(relativo: str) -> set[str]:
    arvore = ast.parse((_RAIZ_REAL / relativo).read_text(encoding="utf-8"))
    modulos: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.ImportFrom) and no.module and no.module.startswith("occasion"):
            modulos.add(no.module)
            # `from occasion.pacote import modulo` também importa o submódulo
            modulos |= {f"{no.module}.{a.name}" for a in no.names}
        elif isinstance(no, ast.Import):
            modulos |= {a.name for a in no.names if a.name.startswith("occasion")}
    return modulos


def _so_docstring(relativo: str) -> bool:
    corpo = ast.parse((_RAIZ_REAL / relativo).read_text(encoding="utf-8")).body
    return all(isinstance(no, ast.Expr) and isinstance(no.value, ast.Constant) for no in corpo)


def test_assinatura_fecha_as_importacoes_da_receita() -> None:
    """Todo módulo `occasion.*` importado por um arquivo assinado também é assinado.

    As exceções ficam em `_IMPORTACOES_FORA_PERMITIDAS`; os `__init__.py` dos pacotes de arquivos
    assinados, que rodam na importação, precisam ser assinados ou ter só a docstring.
    """
    assinados = sorted(
        p.relative_to(_RAIZ_REAL).as_posix()
        for p in (_RAIZ_REAL / "src").rglob("*.py")
        if _assinado(p.relative_to(_RAIZ_REAL).as_posix())
    )
    assert assinados
    fora: dict[str, list[str]] = {}
    for relativo in assinados:
        for modulo in _importados(relativo):
            alvo = _arquivo_do_modulo(modulo)
            if alvo is None or _assinado(alvo) or alvo in _IMPORTACOES_FORA_PERMITIDAS:
                continue
            if alvo.endswith("__init__.py") and _so_docstring(alvo):
                continue
            fora.setdefault(alvo, []).append(relativo)
        for pacote in Path(relativo).parents:
            init = (pacote / "__init__.py").as_posix()
            if (_RAIZ_REAL / init).exists() and not _assinado(init) and not _so_docstring(init):
                fora.setdefault(init, []).append(relativo)
    assert fora == {}
