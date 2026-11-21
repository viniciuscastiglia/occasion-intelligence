"""O artefato do modelo e a regra que tira o score dele: a parte assinada da inferência.

`ArtefatoModelo` e `pontuar` definem o score a partir do artefato, e `_exigir_so_treino` e
`_ufs_de_midia_regional` entram na montagem do artefato avaliado (`inference.treino`). Por isso
este módulo entra na assinatura da receita congelada (`ARQUIVOS_DA_RECEITA`), e
`occasion.inference.predict`, que monta a resposta da tela e do contrato, fica fora dela e
reexporta estes nomes.

**Pickles.** A classe é definida aqui, e `occasion.inference.predict.ArtefatoModelo` é o mesmo
objeto. Um artefato gravado quando a classe morava em `predict` guarda o caminho
`occasion.inference.predict.ArtefatoModelo`; o `joblib.load` resolve esse caminho pelo atributo do
módulo `predict` e chega a esta mesma classe. Os gravados daqui em diante guardam o caminho deste
módulo.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.config import CODIGO_UF, EDICAO_TREINO
from occasion.inference.contract import MetadadosModelo
from occasion.score.score import ReferenciaScore, probabilidades

__all__ = ["NOME_ARQUIVO", "N_UFS_MIDIA_REGIONAL", "ArtefatoModelo", "pontuar"]

N_UFS_MIDIA_REGIONAL = 7
NOME_ARQUIVO = "artefato.joblib"
_TOLERANCIA_EMPATE = 1e-12
_SIGLA_DA_UF: dict[int, str] = {codigo: sigla for sigla, codigo in CODIGO_UF.items()}


@dataclass
class ArtefatoModelo:
    """Tudo o que a inferência precisa: as duas partes, a referência do score e os metadados."""

    modelo_p1: Any
    modelo_p2: Any
    referencia: ReferenciaScore
    ufs_midia_regional: frozenset[str]
    metadados: MetadadosModelo
    # campos do Plano 2; o padrão é atributo de classe, então artefatos antigos continuam valendo
    calibracao_score: tuple[float, float] | None = None
    modelo_direto: Any = None
    assinatura_receita: str | None = None


def _exigir_so_treino(base: pd.DataFrame, nome: str) -> None:
    edicoes = set(cast(pd.Series, base["edicao"]).unique())
    if edicoes != {EDICAO_TREINO}:
        raise ValueError(
            f"{nome} deve conter só a edição {EDICAO_TREINO}; veio {sorted(map(str, edicoes))}"
        )


def _ufs_de_midia_regional(celulas: pd.DataFrame, modelo_p1: Any, modelo_p2: Any) -> frozenset[str]:
    """As UFs de maior preferência média ponderada, com as empatadas com a última incluídas."""
    p = probabilidades(celulas, modelo_p1, modelo_p2)
    pesos = celulas["peso_final"].to_numpy(dtype=float)
    pref = p["p_preferencia"].to_numpy(dtype=float)
    codigos = celulas["uf"].to_numpy()
    medias: dict[str, float] = {}
    for codigo in np.unique(codigos):
        da_uf = codigos == codigo
        medias[_SIGLA_DA_UF[int(codigo)]] = float(
            (pesos[da_uf] * pref[da_uf]).sum() / pesos[da_uf].sum()
        )
    ordenadas = sorted(medias.items(), key=lambda item: (-item[1], item[0]))
    corte = ordenadas[min(N_UFS_MIDIA_REGIONAL, len(ordenadas)) - 1][1]
    return frozenset(
        sigla for sigla, media in ordenadas if media >= corte * (1 - _TOLERANCIA_EMPATE)
    )


def pontuar(X: pd.DataFrame, artefato: ArtefatoModelo) -> pd.DataFrame:
    """Probabilidades, posições e score (posição do produto) de cada linha de contexto."""
    p = probabilidades(
        X, artefato.modelo_p1, artefato.modelo_p2, artefato.calibracao_score, artefato.modelo_direto
    )
    if p.isna().any().any():
        raise ValueError("probabilidade NaN: o contexto não pôde ser pontuado")
    ref = artefato.referencia
    for qual, coluna in (
        ("ocasiao", "p_ocasiao"),
        ("preferencia", "p_preferencia"),
        ("produto", "p_produto"),
    ):
        p[f"posicao_{qual}"] = [ref.posicao(float(v), qual) for v in p[coluna]]  # type: ignore[arg-type]
    p["score"] = p["posicao_produto"]
    return p
