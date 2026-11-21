"""Assinatura da receita congelada: o que a tag `receita-congelada` protege.

`assinatura_da_receita` é o SHA-256 dos arquivos que definem a receita, o modelo avaliado e o
código que mede o teste final (`ARQUIVOS_DA_RECEITA`). O teste final exige que a assinatura da
árvore de trabalho seja a da tag (`exigir_receita_intacta`); o resto de `src/` pode mudar depois
da tag. Este módulo está na própria lista: encolher a lista ou mudar a regra do hash depois da
tag muda a assinatura e bloqueia o teste. `occasion.bases.acesso` reexporta os nomes públicos
(e `acesso.py` fica fora da lista, porque a tarefa 17 o modifica depois da tag).
"""

import hashlib
import subprocess
from pathlib import Path

from occasion.config import RAIZ, TAG_RECEITA

__all__ = [
    "ARQUIVOS_DA_RECEITA",
    "TesteBloqueado",
    "assinatura_da_receita",
    "exigir_receita_intacta",
    "teste_liberado",
]


class TesteBloqueado(RuntimeError):
    """Leitura do alvo da edição de teste antes de a receita estar congelada."""


# O que define a receita, o modelo avaliado e o código que mede o teste final, relativo à raiz
# do repositório (pasta com `/` no fim). O resto de `src/` (linha de comando, pós-teste, servido,
# Trends, hipóteses, `bases/acesso.py`) pode mudar depois da tag sem mudar a assinatura.
ARQUIVOS_DA_RECEITA: tuple[str, ...] = (
    "src/occasion/config.py",
    "src/occasion/features/",
    "src/occasion/bases/modelagem.py",
    "src/occasion/bases/assinatura.py",
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
    "src/occasion/inference/predict.py",
    "reports/receita.json",
    "pyproject.toml",
    "uv.lock",
)

_COMO_LIMPAR = (
    "A assinatura da receita (usada por `make avaliado` e `make teste-final`) só é calculada "
    "com a árvore limpa nesses caminhos: commite (ou descarte) as alterações e os arquivos novos "
    "antes de rodar `make avaliado`."
)


def teste_liberado(raiz: Path = RAIZ) -> bool:
    """Devolve True se a tag da receita congelada existe no repositório em `raiz`.

    Git ausente, pasta inexistente ou qualquer falha do comando contam como não liberado.
    """
    try:
        resultado = subprocess.run(
            ["git", "tag", "-l", TAG_RECEITA],
            cwd=raiz,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return TAG_RECEITA in resultado.stdout.split()


def _git(raiz: Path, *args: str) -> bytes:
    """Saída de `git <args>` em `raiz`; falha do git vira `TesteBloqueado`."""
    try:
        return subprocess.run(["git", *args], cwd=raiz, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as erro:
        detalhe = getattr(erro, "stderr", b"") or b""
        raise TesteBloqueado(
            f"git falhou em {raiz} (`git {' '.join(args)}`): "
            f"{detalhe.decode('utf-8', 'replace').strip() or erro}"
        ) from erro


def _nomes(saida: bytes) -> list[str]:
    return sorted(nome.decode("utf-8") for nome in saida.split(b"\0") if nome)


def assinatura_da_receita(raiz: Path = RAIZ, ref: str | None = None) -> str:
    """SHA-256 dos arquivos rastreados sob `ARQUIVOS_DA_RECEITA`, em ordem de caminho.

    Cada arquivo entra como `caminho NUL sha256(conteúdo) LF`, então renomear também muda a
    assinatura. Com `ref`, lista e lê os arquivos nessa referência do git (`git ls-tree` e
    `git show`), sem olhar a árvore de trabalho. Sem `ref`, lê a árvore de trabalho e levanta
    `TesteBloqueado` se `git status` mostrar alteração (no índice ou não), arquivo apagado ou
    arquivo não rastreado sob esses caminhos: por isso `make avaliado` exige árvore limpa neles.
    """
    if ref is None:
        sujos = _git(
            raiz, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--",
            *ARQUIVOS_DA_RECEITA,
        )  # fmt: skip
        if sujos.strip(b"\0"):
            linhas = [x.decode("utf-8") for x in sujos.split(b"\0") if x]
            raise TesteBloqueado(f"árvore suja nos arquivos da receita: {linhas}. {_COMO_LIMPAR}")
        caminhos = _nomes(_git(raiz, "ls-files", "-z", "--", *ARQUIVOS_DA_RECEITA))
    else:
        caminhos = _nomes(
            _git(raiz, "ls-tree", "-r", "-z", "--name-only", ref, "--", *ARQUIVOS_DA_RECEITA)
        )
    if not caminhos:
        raise TesteBloqueado(f"nenhum arquivo da receita rastreado em {raiz} ({ref or 'árvore'})")
    total = hashlib.sha256()
    for caminho in caminhos:
        if ref is None:
            conteudo = (raiz / caminho).read_bytes()
        else:
            conteudo = _git(raiz, "show", f"{ref}:{caminho}")
        total.update(caminho.encode("utf-8") + b"\0")
        total.update(hashlib.sha256(conteudo).hexdigest().encode("ascii") + b"\n")
    return total.hexdigest()


def exigir_receita_intacta(raiz: Path = RAIZ) -> str:
    """Exige a tag e a assinatura da árvore de trabalho igual à da tag; devolve a assinatura.

    Compara a assinatura, e não o commit: o trabalho fora da receita continua depois da tag.
    """
    if not teste_liberado(raiz):
        raise TesteBloqueado(
            f"a tag `{TAG_RECEITA}` não existe em {raiz}: a receita ainda não foi congelada."
        )
    atual = assinatura_da_receita(raiz)
    da_tag = assinatura_da_receita(raiz, TAG_RECEITA)
    if atual != da_tag:
        raise TesteBloqueado(
            f"a receita mudou depois da tag `{TAG_RECEITA}`: a assinatura da árvore ({atual}) "
            f"não é a da tag ({da_tag}). Confira `git diff {TAG_RECEITA} -- "
            f"{' '.join(ARQUIVOS_DA_RECEITA)}`."
        )
    return atual
