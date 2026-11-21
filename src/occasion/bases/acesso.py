"""Porta única de leitura das bases processadas, com a trava do alvo da edição de teste.

O alvo de 2017-2018 só pode ser lido depois que a tag `receita-congelada` existir no
repositório. Antes disso, só o contexto (sem as colunas de alvo) sai desta porta, e as
colunas de alvo nem chegam a ser carregadas na memória.

A trava (`teste_liberado`, `TesteBloqueado`) e a assinatura da receita moram em
`occasion.bases.assinatura`, que entra na própria assinatura; este módulo os reexporta.
"""

from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.parquet as pq

from occasion.bases.assinatura import (
    ARQUIVOS_DA_RECEITA,
    TesteBloqueado,
    _git,
    assinatura_da_receita,
    exigir_receita_intacta,
    teste_liberado,
)
from occasion.config import DADOS_PROCESSADOS, EDICAO_TESTE, RAIZ, TAG_RECEITA

__all__ = [
    "ARQUIVOS_DA_RECEITA",
    "COLUNAS_DE_ALVO",
    "TesteBloqueado",
    "assinatura_da_receita",
    "carregar_base",
    "exigir_receita_intacta",
    "exigir_teste_commitado",
    "teste_liberado",
]

COLUNAS_DE_ALVO: frozenset[str] = frozenset(
    {"y_pizza_refri", "y_pizza_guarana", "y_guarana", "tem_pizza"}
)


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


ARQUIVO_DO_TESTE_FINAL = "reports/teste_final.json"


def exigir_teste_commitado(raiz: Path = RAIZ, arquivo: str = ARQUIVO_DO_TESTE_FINAL) -> None:
    """Exige `arquivo` (por padrão `reports/teste_final.json`, relativo a `raiz`) no git e sem alteração em relação ao `HEAD`.

    O diagnóstico depois do teste, a decisão do modelo servido e o reajuste só leem o alvo de
    2017-2018 depois de o resultado do teste final estar commitado (seção 12 do design).
    """
    try:
        _git(raiz, "ls-files", "--error-unmatch", "--", arquivo)
    except TesteBloqueado as erro:
        raise TesteBloqueado(f"{arquivo} não está no git: commite o teste final antes.") from erro
    try:
        _git(raiz, "diff", "--quiet", "HEAD", "--", arquivo)
    except TesteBloqueado as erro:
        raise TesteBloqueado(
            f"{arquivo} tem alteração não commitada: o teste final precisa estar "
            "commitado e intacto."
        ) from erro
