"""Validação aninhada por setor: escolhe a configuração dentro do treino e prevê fora dele.

Em cada dobra externa `k`, o treino são os setores de `dobras` com `dobra != k`. A configuração
é escolhida por três dobras internas, tiradas da tabela de setores (as mesmas para qualquer base),
e só então o modelo é retreinado no treino externo inteiro e prevê a dobra `k`. Um setor nunca
está, ao mesmo tempo, no treino e na previsão do mesmo modelo, nas bases extras também.

O treino usa `peso_final` (normalizado pelo próprio modelo) só com `usar_peso`; a seleção, a
métrica da escolha, é sempre ponderada. Bases extras (por exemplo, as células da ocasião) são
previstas pelo modelo que não viu o setor de cada linha; na preferência, com `tem_pizza = 1`.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.config import SEMENTE
from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.models.base import ModeloParte
from occasion.models.escada import Degrau, construir, espaco
from occasion.validacao.dobras import dobra_das_linhas, dobras_internas
from occasion.validacao.metricas import log_loss_ponderado

Construtor = Callable[[Degrau, ParteModelo, dict[str, Any]], ModeloParte]


@dataclass
class PrevisaoInterna:
    """Previsões fora da dobra interna, da configuração escolhida em uma dobra externa."""

    indices: np.ndarray  # posições (iloc), na base, das linhas do treino externo
    p: np.ndarray  # previsão de cada uma dessas linhas, alinhada a `indices`
    extras: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


@dataclass
class ResultadoAninhado:
    oof: np.ndarray
    oof_extras: dict[str, np.ndarray]
    parametros: list[dict[str, Any]]
    interno: list[PrevisaoInterna]
    modelos: list[ModeloParte]  # um por dobra externa, treinado no treino externo inteiro


def _perda_finita(perda: float, degrau: Degrau, params: dict[str, Any]) -> float:
    """A perda da seleção precisa ser finita: NaN ou infinito é erro, nunca é escolhido em silêncio."""
    if not np.isfinite(perda):
        raise ValueError(f"perda não finita ({perda}) na seleção do degrau {degrau}: {params}")
    return perda


def _entrada(df: pd.DataFrame, parte: ParteModelo, *, extra: bool) -> pd.DataFrame:
    """Colunas do modelo mais `cod_upa`; numa base extra da preferência, `tem_pizza = 1`."""
    colunas = list(dict.fromkeys([*variaveis_da_parte(parte), "cod_upa"]))
    if extra and parte == "preferencia":
        return cast(pd.DataFrame, df.assign(tem_pizza=1)[colunas])
    return cast(pd.DataFrame, df[colunas])


def _ajustar(
    construtor: Construtor,
    degrau: Degrau,
    parte: ParteModelo,
    params: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    usar_peso: bool,
) -> ModeloParte:
    return construtor(degrau, parte, params).fit(X, y, w if usar_peso else None)


def _preparar(
    base: pd.DataFrame,
    alvo: str,
    parte: ParteModelo,
    bases_extras: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict[str, pd.DataFrame]]:
    X = _entrada(base, parte, extra=False)
    extras = {n: _entrada(b, parte, extra=True) for n, b in bases_extras.items()}
    return X, cast(pd.Series, base[alvo]), cast(pd.Series, base["peso_final"]), extras


def _fora_da_dobra_externa(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    extras: Mapping[str, pd.DataFrame],
    por_linha: np.ndarray,
    por_linha_extras: Mapping[str, np.ndarray],
    params: dict[str, Any],
    degrau: Degrau,
    parte: ParteModelo,
    construtor: Construtor,
    usar_peso: bool,
    k: int,
) -> tuple[ModeloParte, np.ndarray, dict[str, np.ndarray]]:
    """Retreina fora da dobra `k` e prevê as linhas da dobra `k` (base e extras)."""
    dentro = por_linha != k
    modelo = _ajustar(
        construtor, degrau, parte, params,
        cast(pd.DataFrame, X[dentro]), cast(pd.Series, y[dentro]), cast(pd.Series, w[dentro]),
        usar_peso,
    )  # fmt: skip
    p = modelo.predict_proba(cast(pd.DataFrame, X[~dentro]))
    p_extras = {
        n: modelo.predict_proba(cast(pd.DataFrame, e[por_linha_extras[n] == k]))
        for n, e in extras.items()
        if bool((por_linha_extras[n] == k).any())
    }
    return modelo, p, p_extras


def validacao_aninhada(
    base: pd.DataFrame,
    alvo: str,
    degrau: Degrau,
    parte: ParteModelo,
    dobras: pd.DataFrame,
    *,
    usar_peso: bool,
    bases_extras: Mapping[str, pd.DataFrame] | None = None,
    construtor: Construtor = construir,
    semente: int = SEMENTE,
    progresso: Callable[[int], None] | None = None,
) -> ResultadoAninhado:
    """Validação aninhada por setor; ver o cabeçalho do módulo.

    `progresso`, se dado, é chamado com `k` ao começar cada dobra externa (só para acompanhar).
    """
    X, y, w, extras = _preparar(base, alvo, parte, bases_extras or {})
    por_linha = dobra_das_linhas(base, dobras)
    por_linha_extras = {n: dobra_das_linhas(b, dobras) for n, b in (bases_extras or {}).items()}
    configuracoes = espaco(degrau, parte)
    k_externo = int(np.max(dobras["dobra"].to_numpy())) + 1

    oof = np.full(len(base), np.nan)
    oof_extras = {n: np.full(len(b), np.nan) for n, b in extras.items()}
    parametros: list[dict[str, Any]] = []
    interno: list[PrevisaoInterna] = []
    modelos: list[ModeloParte] = []

    for k in range(k_externo):
        if progresso is not None:
            progresso(k)
        treino = np.flatnonzero(por_linha != k)
        internas = dobras_internas(dobras, k, semente=semente)
        j_base = dobra_das_linhas(base.iloc[treino], internas)
        k_interno = int(np.max(internas["dobra"].to_numpy())) + 1
        y_t, w_t = cast(pd.Series, y.iloc[treino]), cast(pd.Series, w.iloc[treino])

        melhor: tuple[float, dict[str, Any], np.ndarray, list[ModeloParte]] | None = None
        for params in configuracoes:
            p_int = np.full(len(treino), np.nan)
            internos: list[ModeloParte] = []
            for j in range(k_interno):
                tr, va = j_base != j, j_base == j
                m = _ajustar(
                    construtor, degrau, parte, params,
                    cast(pd.DataFrame, X.iloc[treino[tr]]),
                    cast(pd.Series, y_t[tr]), cast(pd.Series, w_t[tr]), usar_peso,
                )  # fmt: skip
                p_int[va] = m.predict_proba(cast(pd.DataFrame, X.iloc[treino[va]]))
                internos.append(m)
            perda = _perda_finita(
                log_loss_ponderado(y_t.to_numpy(), p_int, w_t.to_numpy()), degrau, params
            )
            if melhor is None or perda < melhor[0]:  # empate: fica a primeira
                melhor = (perda, params, p_int, internos)
        assert melhor is not None
        _, escolhida, p_int, internos = melhor

        # só a configuração escolhida prevê as bases extras, com os modelos que não viram o setor
        p_int_extras: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for n, e in extras.items():
            posicoes = np.flatnonzero(por_linha_extras[n] != k)
            j_extra = dobra_das_linhas(bases_extras[n].iloc[posicoes], internas)  # type: ignore[index]
            p_e = np.full(len(posicoes), np.nan)
            for j in range(k_interno):
                alvo_j = j_extra == j
                if alvo_j.any():
                    p_e[alvo_j] = internos[j].predict_proba(
                        cast(pd.DataFrame, e.iloc[posicoes[alvo_j]])
                    )
            p_int_extras[n] = (posicoes, p_e)
        interno.append(PrevisaoInterna(treino, p_int, p_int_extras))
        parametros.append(escolhida)

        modelo, p, p_extras = _fora_da_dobra_externa(
            X, y, w, extras, por_linha, por_linha_extras, escolhida, degrau, parte,
            construtor, usar_peso, k,
        )  # fmt: skip
        oof[por_linha == k] = p
        for n, p_n in p_extras.items():
            oof_extras[n][por_linha_extras[n] == k] = p_n
        modelos.append(modelo)

    return ResultadoAninhado(oof, oof_extras, parametros, interno, modelos)


def _oof_com_modelos(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    extras: Mapping[str, pd.DataFrame],
    por_linha: np.ndarray,
    por_linha_extras: Mapping[str, np.ndarray],
    params: dict[str, Any],
    degrau: Degrau,
    parte: ParteModelo,
    construtor: Construtor,
    usar_peso: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    oof = np.full(len(X), np.nan)
    oof_extras = {n: np.full(len(e), np.nan) for n, e in extras.items()}
    for k in range(int(por_linha.max()) + 1):
        _, p, p_extras = _fora_da_dobra_externa(
            X, y, w, extras, por_linha, por_linha_extras, params, degrau, parte,
            construtor, usar_peso, k,
        )  # fmt: skip
        oof[por_linha == k] = p
        for n, p_n in p_extras.items():
            oof_extras[n][por_linha_extras[n] == k] = p_n
    return oof, oof_extras


def previsao_fora_da_dobra(
    base: pd.DataFrame,
    alvo: str,
    degrau: Degrau,
    parte: ParteModelo,
    params: dict[str, Any],
    dobras: pd.DataFrame,
    *,
    usar_peso: bool,
    bases_extras: Mapping[str, pd.DataFrame] | None = None,
    construtor: Construtor = construir,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Validação simples nas dobras externas, com os parâmetros fixos `params`."""
    X, y, w, extras = _preparar(base, alvo, parte, bases_extras or {})
    return _oof_com_modelos(
        X, y, w, extras, dobra_das_linhas(base, dobras),
        {n: dobra_das_linhas(b, dobras) for n, b in (bases_extras or {}).items()},
        params, degrau, parte, construtor, usar_peso,
    )  # fmt: skip


def selecao_final(
    base: pd.DataFrame,
    alvo: str,
    degrau: Degrau,
    parte: ParteModelo,
    dobras: pd.DataFrame,
    *,
    usar_peso: bool,
    bases_extras: Mapping[str, pd.DataFrame] | None = None,
    construtor: Construtor = construir,
) -> tuple[dict[str, Any], np.ndarray, dict[str, np.ndarray]]:
    """Escolhe a configuração pelo log loss ponderado fora da dobra nas 5 dobras externas.

    Devolve os parâmetros (que fixam o modelo final, treinado depois em 2008-2009 inteira) e as
    previsões fora da dobra deles, na base e nas extras, para a calibração do artefato.
    Empate: a primeira configuração do espaço.
    """
    X, y, w, extras = _preparar(base, alvo, parte, bases_extras or {})
    por_linha = dobra_das_linhas(base, dobras)
    por_linha_extras = {n: dobra_das_linhas(b, dobras) for n, b in (bases_extras or {}).items()}
    melhor: tuple[float, dict[str, Any], np.ndarray, dict[str, np.ndarray]] | None = None
    for params in espaco(degrau, parte):
        oof, oof_e = _oof_com_modelos(
            X, y, w, extras, por_linha, por_linha_extras, params, degrau, parte,
            construtor, usar_peso,
        )  # fmt: skip
        perda = _perda_finita(log_loss_ponderado(y.to_numpy(), oof, w.to_numpy()), degrau, params)
        if melhor is None or perda < melhor[0]:
            melhor = (perda, params, oof, oof_e)
    assert melhor is not None
    return melhor[1], melhor[2], melhor[3]
