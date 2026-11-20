"""Porta única de leitura das bases processadas, com a trava do alvo da edição de teste.

O alvo de 2017-2018 só pode ser lido depois que a tag `receita-congelada` existir no
repositório. Antes disso, só o contexto (sem as colunas de alvo) sai desta porta, e as
colunas de alvo nem chegam a ser carregadas na memória.
"""

import subprocess
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from occasion.config import DADOS_PROCESSADOS, EDICAO_TESTE, RAIZ, TAG_RECEITA


class TesteBloqueado(RuntimeError):
    """Leitura do alvo da edição de teste antes de a receita estar congelada."""


COLUNAS_DE_ALVO: frozenset[str] = frozenset(
    {"y_pizza_refri", "y_pizza_guarana", "y_guarana", "tem_pizza"}
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


def carregar_base(
    nome: Literal["celulas", "ocasioes_refri"],
    edicao: str,
    *,
    so_contexto: bool = False,
    pasta: Path = DADOS_PROCESSADOS,
    raiz: Path = RAIZ,
) -> pd.DataFrame:
    """Lê `<nome>_<edicao>.parquet`; em 2017-2018 o alvo exige a tag da receita congelada."""
    caminho = pasta / f"{nome}_{edicao}.parquet"
    if edicao != EDICAO_TESTE:
        return pd.read_parquet(caminho)
    if so_contexto:
        colunas = [c for c in pq.read_schema(caminho).names if c not in COLUNAS_DE_ALVO]
        return pd.read_parquet(caminho, columns=colunas)
    if not teste_liberado(raiz):
        raise TesteBloqueado(
            f"O alvo de {EDICAO_TESTE} só pode ser lido depois da tag `{TAG_RECEITA}`."
        )
    return pd.read_parquet(caminho)
