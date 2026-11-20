"""Bases sintéticas com todas as colunas de contexto, para os testes dos modelos de parte."""

import numpy as np
import pandas as pd

from occasion.features.variaveis import DOMINIOS, regiao_da_uf
from occasion.models.base import sigmoide

UFS = [uf for uf in DOMINIOS["uf"]]


def contexto_sintetico(n: int, semente: int = 0, excluir_uf: int | None = None) -> pd.DataFrame:
    """`n` linhas de contexto com valores válidos; `excluir_uf` tira uma UF do sorteio."""
    rng = np.random.default_rng(semente)
    ufs = [u for u in UFS if u != excluir_uf]
    uf = rng.choice(ufs, size=n)
    return pd.DataFrame(
        {
            "dia_semana": rng.choice(DOMINIOS["dia_semana"], size=n),
            "faixa_horario": rng.choice(DOMINIOS["faixa_horario"], size=n),
            "local": rng.choice(DOMINIOS["local"], size=n),
            "uf": uf,
            "regiao": [regiao_da_uf(int(u)) for u in uf],
            "situacao": rng.choice(DOMINIOS["situacao"], size=n),
            "idade": rng.integers(10, 90, size=n),
            "sexo": rng.choice(DOMINIOS["sexo"], size=n),
            "renda_pc_sm": rng.lognormal(0.0, 1.0, size=n),
            "n_moradores_uc": rng.integers(1, 8, size=n),
        }
    )


def base_sintetica(
    n: int = 3000, semente: int = 0, excluir_uf: int | None = None, com_pizza: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """X, y e pesos com sinal real: noite e fora de casa aumentam a chance, a idade a diminui."""
    X = contexto_sintetico(n, semente, excluir_uf)
    rng = np.random.default_rng(semente + 1)
    if com_pizza:
        X["tem_pizza"] = rng.integers(0, 2, size=n)
    z = (
        -1.0
        + 1.2 * (X["faixa_horario"] == "noite")
        + 0.6 * (X["local"] == "fora_de_casa")
        - 0.02 * (X["idade"] - 40)
        + (0.8 * X["tem_pizza"] if com_pizza else 0.0)
    )
    y = pd.Series((rng.random(n) < sigmoide(z.to_numpy())).astype(int), name="y")
    w = pd.Series(rng.uniform(0.5, 3.0, size=n), name="peso_final")
    return X, y, w
