"""O que o pré-registro fixa antes de qualquer modelo: variáveis, espaços de busca e interações.

Tudo aqui é escrito antes da escada rodar (`docs/pre-registro.md`). As listas vêm do design, do
enunciado do TAPI e das perguntas do projeto, não do tamanho dos efeitos vistos na EDA: escolher
interações pelo alvo de 2008-2009 inteiro e depois medir o degrau C contra o B na mesma base
deixaria H2 otimista. O degrau A e o conjunto de interações do C são hiperparâmetros, escolhidos
dentro da validação aninhada (decisão do autor em 20/11/2026).
"""

import math
from itertools import product
from typing import Any, Literal

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.model_selection import ParameterSampler

from occasion.config import SEMENTE
from occasion.features.variaveis import COLUNAS_CONTEXTO, DOMINIOS, NUMERICAS

ParteModelo = Literal["ocasiao", "preferencia", "direto"]
_PARTES: tuple[ParteModelo, ...] = ("ocasiao", "preferencia", "direto")


def variaveis_da_parte(parte: ParteModelo) -> list[str]:
    """Variáveis de entrada da parte: o contexto, mais `tem_pizza` na preferência."""
    if parte not in _PARTES:
        raise ValueError(f"parte desconhecida: {parte!r}")
    if parte == "preferencia":
        return [*COLUNAS_CONTEXTO, "tem_pizza"]
    return list(COLUNAS_CONTEXTO)


# ---------------------------------------------------------------- degraus B e C (logística)

# Inverso da força de regularização L2, do mais regularizado (o mais simples) ao menos.
GRADE_C_LOGISTICA: list[float] = [float(c) for c in np.logspace(-3, 2, 8)]

# Curva do degrau C, aplicada a `idade` e a `log1p(renda_pc_sm)`: 5 colunas por curva.
SPLINE: dict[str, Any] = {"n_knots": 4, "degree": 3, "knots": "quantile", "include_bias": False}
CURVAS: list[str] = ["idade", "renda_pc_sm"]

_INTERACOES_OCASIAO: dict[str, list[tuple[str, str]]] = {
    # "Sexta a domingo, à noite e em casa" (TAPI) e a diferença urbano/rural fora de casa.
    "pre_declarado": [
        ("dia_semana", "faixa_horario"),
        ("faixa_horario", "local"),
        ("local", "situacao"),
    ],
    # Mesma pergunta com o dia agrupado como no TAPI: a EDA mostrou que dia x faixa só tem
    # positivos à noite, e 24 graus de liberdade gastos nisso seriam quase todos ruído.
    "fim_de_semana": [
        ("fim_de_semana", "faixa_horario"),
        ("faixa_horario", "local"),
        ("local", "situacao"),
    ],
}

# Conjuntos nomeados; o conjunto é hiperparâmetro do degrau C, escolhido junto com `C`.
INTERACOES: dict[ParteModelo, dict[str, list[tuple[str, str]]]] = {
    "ocasiao": {nome: list(pares) for nome, pares in _INTERACOES_OCASIAO.items()},
    # A pizza muda a preferência dentro e fora de casa, e por região? A região muda com o local?
    "preferencia": {
        "pre_declarado": [
            ("tem_pizza", "local"),
            ("tem_pizza", "regiao"),
            ("regiao", "local"),
            ("faixa_horario", "local"),
        ]
    },
    "direto": {nome: list(pares) for nome, pares in _INTERACOES_OCASIAO.items()},
}


def _graus_categorica(variavel: str) -> int:
    return len(DOMINIOS[variavel]) - 1


def _graus_curva() -> int:
    return SPLINE["n_knots"] + SPLINE["degree"] - 2


def _conjunto(parte: ParteModelo, conjunto: str) -> list[tuple[str, str]]:
    if conjunto not in INTERACOES[parte]:
        raise ValueError(
            f"conjunto de interações {conjunto!r} não existe na parte {parte!r}: "
            f"{list(INTERACOES[parte])}"
        )
    return INTERACOES[parte][conjunto]


def graus_de_liberdade(
    parte: ParteModelo, degrau: Literal["B", "C"], conjunto: str = "pre_declarado"
) -> int:
    """Parâmetros do modelo logístico, sem o intercepto.

    Categórica com `L` níveis conta `L - 1`; numérica conta 1; no degrau C, cada curva conta
    `n_knots + degree - 2` no lugar do termo linear; interação `a x b` conta `(La - 1)(Lb - 1)`.
    """
    if degrau not in ("B", "C"):
        raise ValueError(f"graus de liberdade só valem para os degraus B e C: {degrau!r}")
    total = 0
    for v in variaveis_da_parte(parte):
        if v in NUMERICAS:
            total += _graus_curva() if degrau == "C" and v in CURVAS else 1
        else:
            total += _graus_categorica(v)
    if degrau == "C":
        total += sum(
            _graus_categorica(a) * _graus_categorica(b) for a, b in _conjunto(parte, conjunto)
        )
    return total


def parametros_de_pizza(parte: ParteModelo = "preferencia", conjunto: str = "pre_declarado") -> int:
    """Parâmetros que dependem de `tem_pizza`: ela mesma e as interações que a envolvem.

    É o que o recorte de pizza (101 positivos em 2008-2009) precisa sustentar no degrau C.
    """
    if "tem_pizza" not in variaveis_da_parte(parte):
        return 0
    interacoes = sum(
        _graus_categorica(a) * _graus_categorica(b)
        for a, b in _conjunto(parte, conjunto)
        if "tem_pizza" in (a, b)
    )
    return _graus_categorica("tem_pizza") + interacoes


# ---------------------------------------------------------------- degrau D (LightGBM)

N_CONFIGS_LIGHTGBM = 30

_FIXOS_LIGHTGBM: dict[str, Any] = {
    "objective": "binary",
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 4,
    "random_state": SEMENTE,
    "verbose": -1,
}


def espaco_lightgbm(semente: int = SEMENTE) -> list[dict[str, Any]]:
    """As 30 configurações sorteadas do LightGBM, da mais simples à mais complexa.

    A ordem, por `(max_depth, n_estimators)`, é a do desempate: na mesma métrica, fica a
    primeira. Os valores saem como tipos do Python, para irem direto ao JSON do experimento.
    """
    distribuicoes = {
        "learning_rate": loguniform(0.01, 0.2),
        "n_estimators": randint(100, 801),  # [100, 800]: o limite de cima do randint é aberto
        "max_depth": [2, 3, 4],
        "min_child_samples": [200, 500, 1000, 2000],
        "reg_lambda": loguniform(0.01, 10),
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }
    sorteio = ParameterSampler(distribuicoes, n_iter=N_CONFIGS_LIGHTGBM, random_state=semente)
    configuracoes = []
    for p in sorteio:
        profundidade = int(p["max_depth"])
        configuracoes.append(
            {
                "learning_rate": float(p["learning_rate"]),
                "n_estimators": int(p["n_estimators"]),
                "max_depth": profundidade,
                "num_leaves": 2**profundidade,
                "min_child_samples": int(p["min_child_samples"]),
                "reg_lambda": float(p["reg_lambda"]),
                "subsample": float(p["subsample"]),
                "subsample_freq": 1,
                "colsample_bytree": float(p["colsample_bytree"]),
                **_FIXOS_LIGHTGBM,
            }
        )
    return sorted(configuracoes, key=lambda c: (c["max_depth"], c["n_estimators"]))


# ---------------------------------------------------------------- degrau A (tabela de taxas)

SUAVIZACOES_A: tuple[int, ...] = (10, 100, 1000)

_DIMENSOES_A_OCASIAO: list[list[str]] = [
    ["dia_semana", "faixa_horario", "local"],  # a tabela do Plano 1
    ["fim_de_semana", "faixa_horario", "local"],
    ["faixa_horario", "local"],
    ["dia_semana", "faixa_horario"],  # o exemplo literal do TAPI
]
_DIMENSOES_A_PREFERENCIA: list[list[str]] = [
    ["regiao", "local"],  # a tabela do Plano 1
    ["regiao"],
    ["uf"],
    ["uf", "local"],
]


def _combinacoes(dimensoes: list[str]) -> int:
    return math.prod(len(DOMINIOS[d]) for d in dimensoes)


def _espaco_a(opcoes: list[list[str]]) -> list[dict[str, Any]]:
    """Dimensões x suavização, da mais simples (menos combinações, mais suavização) à mais complexa."""
    configs = [{"dimensoes": list(d), "suavizacao": m} for d, m in product(opcoes, SUAVIZACOES_A)]
    return sorted(configs, key=lambda c: (_combinacoes(c["dimensoes"]), -c["suavizacao"]))


# O degrau A também é escolhido na validação aninhada (decisão do autor em 20/11/2026): o
# baseline é o mais forte possível, para os modelos não serem comparados com uma tabela fraca
# por ruído. A `suavizacao` é o `m` da `TabelaDeTaxas`, em linhas de peso médio 1.
ESPACO_A: dict[ParteModelo, list[dict[str, Any]]] = {
    "ocasiao": _espaco_a(_DIMENSOES_A_OCASIAO),
    "preferencia": _espaco_a(_DIMENSOES_A_PREFERENCIA),
    "direto": _espaco_a(_DIMENSOES_A_OCASIAO),
}
