"""Faixas de idade e de renda e taxas ponderadas por grupo, com intervalo de bootstrap.

As faixas são reusadas pela avaliação por segmentos e pela paridade da inferência. A taxa por
grupo é a razão ponderada por `peso_final`; o intervalo vem do bootstrap de setores dentro da UF
(`occasion.validacao.bootstrap`), sorteando os setores da base inteira, e não só os do grupo:
é a estimação de domínio, em que o tamanho do grupo também varia entre as réplicas.
"""

from typing import Any, cast, get_args

import numpy as np
import pandas as pd

from occasion.config import SEMENTE
from occasion.inference.contract import FaixaRenda
from occasion.validacao.bootstrap import intervalo_estatistica

__all__ = [
    "CORTES_RENDA",
    "FAIXAS_ETARIAS",
    "faixa_de_renda",
    "faixa_etaria",
    "taxa_por_grupo",
]

FAIXAS_ETARIAS: dict[str, tuple[int, int]] = {
    "10-17": (10, 17),
    "18-29": (18, 29),
    "30-44": (30, 44),
    "45-59": (45, 59),
    "60+": (60, 130),
}

# Cortes da renda por pessoa, em salários mínimos, fechados à esquerda: [0; 0,5), [0,5; 1) etc.
CORTES_RENDA: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0)
_CODIGOS_RENDA: tuple[str, ...] = get_args(FaixaRenda)


def faixa_etaria(idade: pd.Series) -> pd.Series:
    """Rótulo de `FAIXAS_ETARIAS` de cada idade em anos; idade fora das faixas é erro."""
    valores = idade.to_numpy()
    rotulos = np.full(len(valores), None, dtype=object)
    for nome, (inicio, fim) in FAIXAS_ETARIAS.items():
        rotulos[(valores >= inicio) & (valores <= fim)] = nome
    if any(r is None for r in rotulos):
        fora = sorted({v for v, r in zip(valores.tolist(), rotulos, strict=True) if r is None})
        raise ValueError(f"idade fora de FAIXAS_ETARIAS: {fora[:5]}")
    return pd.Series(rotulos, index=idade.index, dtype=object, name=idade.name)


def faixa_de_renda(renda_pc_sm: pd.Series) -> pd.Series:
    """Código de `FaixaRenda` do contrato para cada renda por pessoa em salários mínimos."""
    valores = renda_pc_sm.to_numpy(dtype=float)
    if np.isnan(valores).any() or (valores < 0).any():
        raise ValueError("renda ausente ou negativa: a faixa de renda não está definida")
    posicao = np.searchsorted(np.asarray(CORTES_RENDA), valores, side="right")
    rotulos = np.asarray(_CODIGOS_RENDA, dtype=object)[posicao]
    return pd.Series(rotulos, index=renda_pc_sm.index, dtype=object, name=renda_pc_sm.name)


def taxa_por_grupo(
    base: pd.DataFrame,
    alvo: str,
    por: list[str],
    b: int = 200,
    semente: int = SEMENTE,
) -> pd.DataFrame:
    """Taxa ponderada de `alvo` em cada combinação de `por`, com intervalo de 95%.

    `base` precisa de `alvo` (0 ou 1), `peso_final`, `cod_upa` e `uf`. Devolve uma linha por
    combinação observada, com `n` (linhas), `positivos` (contagem simples de `alvo`), `taxa`
    (ponderada) e o intervalo percentil `ic_inf` e `ic_sup` do bootstrap de setores.

    Como o multiplicador do bootstrap é o mesmo em todas as linhas de um setor, a taxa de cada
    réplica só depende das somas por setor; o cálculo usa essas somas, que é idêntico a usar as
    linhas e muito mais rápido.
    """
    if not por:
        raise ValueError("é preciso ao menos uma coluna em `por`")
    peso = base["peso_final"].to_numpy(dtype=float)
    y = base[alvo].to_numpy(dtype=float)
    setores, setor_da_linha = np.unique(base["cod_upa"].to_numpy(), return_inverse=True)
    uf_do_setor = np.empty(len(setores), dtype=base["uf"].to_numpy().dtype)
    uf_do_setor[setor_da_linha] = base["uf"].to_numpy()

    chaves = cast(pd.DataFrame, base[por]).reset_index(drop=True)
    grupos = chaves.groupby(por, sort=True, observed=True, dropna=True).indices
    linhas: list[dict[str, Any]] = []
    for chave, indices in grupos.items():
        valores = chave if isinstance(chave, tuple) else (chave,)
        w_s = np.bincount(setor_da_linha[indices], weights=peso[indices], minlength=len(setores))
        wy_s = np.bincount(
            setor_da_linha[indices], weights=peso[indices] * y[indices], minlength=len(setores)
        )

        def taxa(m: np.ndarray, w_s: np.ndarray = w_s, wy_s: np.ndarray = wy_s) -> float | None:
            denominador = float(m @ w_s)
            return float(m @ wy_s) / denominador if denominador > 0 else None

        intervalo = intervalo_estatistica(taxa, setores, uf_do_setor, b=b, semente=semente)
        linhas.append(
            dict(zip(por, valores, strict=True))
            | {
                "n": len(indices),
                "positivos": int(y[indices].sum()),
                "taxa": intervalo.estimativa,
                "ic_inf": intervalo.ic_inf,
                "ic_sup": intervalo.ic_sup,
            }
        )
    return pd.DataFrame(linhas, columns=[*por, "n", "positivos", "taxa", "ic_inf", "ic_sup"])
