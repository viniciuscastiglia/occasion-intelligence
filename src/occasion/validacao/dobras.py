"""Dobras de validação cruzada que imitam o desenho amostral da POF.

O setor (`cod_upa`) fica inteiro em uma dobra, e os setores de cada estrato são
espalhados entre as dobras. As dobras internas da validação aninhada saem da
tabela de setores, não das linhas de uma base, para serem as mesmas nas duas
partes do modelo.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from occasion.config import SEMENTE

K_EXTERNO = 5
K_INTERNO = 3

# Critério da conferência, declarado antes do uso: cada dobra externa tem entre
# 10% e 30% dos positivos de cada alvo.
FRACAO_MIN = 0.10
FRACAO_MAX = 0.30


def tabela_de_upas(base: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por setor, com as colunas `cod_upa` e `estrato`."""
    return base[["cod_upa", "estrato"]].drop_duplicates("cod_upa")  # pyright: ignore[reportCallIssue].reset_index(drop=True)


def atribuir_dobras(upas: pd.DataFrame, k: int, semente: int) -> pd.DataFrame:
    """Atribui cada setor a uma dobra de 0 a `k-1`.

    Permuta os setores, ordena de forma estável por estrato e atribui
    `dobra = posição % k`: dentro de cada estrato a diferença entre dobras é de
    no máximo um setor e as dobras ficam do mesmo tamanho no total.
    """
    rng = np.random.default_rng(semente)
    permutadas = upas[["cod_upa", "estrato"]].iloc[rng.permutation(len(upas))]
    ordenadas = permutadas.sort_values("estrato", kind="stable").reset_index(drop=True)
    ordenadas["dobra"] = (np.arange(len(ordenadas)) % k).astype(int)
    return ordenadas


def dobras_internas(
    dobras: pd.DataFrame, externa: int, k: int = K_INTERNO, semente: int = SEMENTE
) -> pd.DataFrame:
    """Dobras internas dos setores de treino da dobra externa `externa`.

    Usa a semente `semente + externa`.
    """
    treino = dobras.loc[dobras["dobra"] != externa, ["cod_upa", "estrato"]]
    return atribuir_dobras(treino, k, semente + externa)


def dobra_das_linhas(base: pd.DataFrame, dobras: pd.DataFrame) -> np.ndarray:
    """A dobra de cada linha pelo `cod_upa`; setor fora da tabela levanta `ValueError`."""
    mapa: dict[Any, int] = dict(zip(dobras["cod_upa"], dobras["dobra"], strict=True))
    setores = base["cod_upa"].to_numpy()
    ausentes = sorted({str(u) for u in setores if u not in mapa})
    if ausentes:
        raise ValueError(f"setores ausentes da tabela de dobras: {ausentes[:10]}")
    return np.array([mapa[u] for u in setores], dtype=int)


def conferir_dobras(
    alvos: Mapping[str, tuple[pd.DataFrame, str]], dobras: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    """Confere se cada dobra tem entre 10% e 30% dos positivos de cada alvo.

    `alvos` mapeia o nome a (base, coluna do alvo). Para cada nome devolve os
    positivos por dobra (contagem simples), a fração de cada dobra sobre o total
    de positivos do alvo, `ok` e `motivo`. A chave `geral` traz o `ok` de todos
    os alvos. Alvo sem nenhum positivo reprova, porque a fração é indefinida.
    """
    k = int(np.max(dobras["dobra"].to_numpy())) + 1
    resultado: dict[str, dict[str, Any]] = {}
    for nome, (base, coluna) in alvos.items():
        por_linha = dobra_das_linhas(base, dobras)
        positivo = base[coluna].to_numpy() == 1
        contagem = np.bincount(por_linha[positivo], minlength=k)
        total = int(contagem.sum())
        entrada: dict[str, Any] = {"positivos_por_dobra": {d: int(contagem[d]) for d in range(k)}}
        if total == 0:
            entrada.update(fracao_por_dobra=None, ok=False, motivo="alvo sem positivos")
        else:
            fracoes = {d: float(contagem[d]) / total for d in range(k)}
            fora = [d for d, f in fracoes.items() if not FRACAO_MIN <= f <= FRACAO_MAX]
            entrada.update(
                fracao_por_dobra=fracoes,
                ok=not fora,
                motivo=None if not fora else f"dobras fora de 10%-30%: {fora}",
            )
        resultado[nome] = entrada
    resultado["geral"] = {"ok": all(v["ok"] for v in resultado.values())}
    return resultado


def escolher_dobras(
    upas: pd.DataFrame,
    alvos: Mapping[str, tuple[pd.DataFrame, str]],
    semente: int = SEMENTE,
    tentativas: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primeira atribuição externa que `conferir_dobras` aprova.

    Tenta as sementes `semente + 100·i`. Devolve as dobras e o registro
    (semente usada e, por tentativa, a semente e o resultado da conferência).
    Se nenhuma aprovar, levanta `ValueError` com o registro das tentativas.
    """
    registro: list[dict[str, Any]] = []
    for i in range(tentativas):
        s = semente + 100 * i
        dobras = atribuir_dobras(upas, K_EXTERNO, s)
        resultado = conferir_dobras(alvos, dobras)
        registro.append({"semente": s, "resultado": resultado})
        if resultado["geral"]["ok"]:
            return dobras, {"semente": s, "tentativas": registro}
    raise ValueError(f"nenhuma das {tentativas} tentativas de dobras aprovou: {registro}")
