"""Curva de aprendizado de C e D em 2008-2009 (tarefa 17a, exploratória).

Pergunta: com mais dados, o ganho do LightGBM (D) sobre a logística com curvas (C) cresce? A regra
está declarada na seção 27 do registro de decisões: o resultado não troca o modelo servido.

Desenho, por parte (ocasião e preferência):
- em cada dobra externa `k`, os setores de treino (`dobra != k`) são permutados com a semente
  `SEMENTE + 1000 + k` e as frações de 25%, 50%, 75% e 100% são os primeiros setores da
  permutação (cada fração contém a anterior); as linhas avaliadas são sempre todas as da dobra
  `k`;
- C e D usam, na dobra `k`, os hiperparâmetros que a validação aninhada escolheu naquela dobra
  com 100% dos setores (`parametros_por_dobra` de `escada_<parte>.json`), reusados nas frações
  menores: nada é reescolhido por fração;
- o treino usa `peso_final` conforme H5 (`usar_peso`); a métrica é sempre ponderada.

O ganho D - C (log loss, menor é melhor) tem intervalo por bootstrap de setores dentro da UF. A
projeção para o dobro dos eventos de treino é extrapolação, não medida: ajusta-se uma reta em
`log(n)` ao ganho de cada réplica do bootstrap, com o mesmo multiplicador em todas as frações.
"""

import math
from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.config import SEMENTE
from occasion.experimentos.aninhada import _entrada
from occasion.experimentos.especificacao import ParteModelo
from occasion.experimentos.etapas import (
    PARTES,
    Parte,
    _base_da_parte,
    _celulas,
    _conferir_dobras_de_h5,
    _dados,
    _dobras,
    _info_dobras,
    _ocasioes,
    _pasta_relatorios,
    _relatorio,
    _sem_nan,
    _usar_peso_decidido,
    experimento,
)
from occasion.experimentos.registro import salvar_resultado
from occasion.models.escada import Degrau, construir
from occasion.validacao.bootstrap import B_PADRAO, Intervalo, intervalo_ganho, multiplicadores
from occasion.validacao.dobras import dobra_das_linhas
from occasion.validacao.metricas import log_loss_ponderado, pr_auc_ponderada

FRACOES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
DEGRAUS: tuple[Degrau, Degrau] = ("C", "D")
DESLOCAMENTO_DA_SEMENTE = 1000
FATOR_DA_PROJECAO = 2.0
_AVISO_PROJECAO = (
    "a projeção é extrapolação: vai além dos dados medidos e supõe que a tendência em log(n) "
    "continua; não é uma medida, e o modelo servido não depende dela"
)
_FORMA = (
    "ganho(n) = a + b * log(n), reta em log do número de eventos de treino; escolhida no lugar "
    "de a + b * n^(-c) porque, com 4 pontos e ruído, a potência tem 3 parâmetros e o ajuste é "
    "instável (c e a não se identificam); a reta tem 2 parâmetros e sempre tem solução"
)


def fracoes_aninhadas(
    setores: np.ndarray, k: int, semente: int = SEMENTE
) -> dict[float, np.ndarray]:
    """Setores de cada fração, aninhados: permuta com `semente + 1000 + k` e pega os primeiros."""
    unicos = np.unique(np.asarray(setores))
    ordem = np.random.default_rng(semente + DESLOCAMENTO_DA_SEMENTE + k).permutation(len(unicos))
    permutados = unicos[ordem]
    return {f: permutados[: max(1, min(len(unicos), math.ceil(f * len(unicos))))] for f in FRACOES}


def projetar(n: np.ndarray, ganho: np.ndarray, n_alvo: float) -> float | None:
    """Reta do ganho em `log(n)` projetada em `n_alvo`; `None` se o ajuste não for possível."""
    n, ganho = np.asarray(n, dtype=float), np.asarray(ganho, dtype=float)
    if not (np.isfinite(n).all() and np.isfinite(ganho).all()) or (n <= 0).any() or n_alvo <= 0:
        return None
    try:
        b, a = np.polyfit(np.log(n), ganho, 1)
    except np.linalg.LinAlgError:
        return None
    valor = float(a + b * np.log(n_alvo))
    return valor if math.isfinite(valor) else None


def _ajustar_e_prever(
    X: pd.DataFrame,
    y: pd.Series,
    w: pd.Series,
    treino: np.ndarray,
    teste: np.ndarray,
    degrau: Degrau,
    parte: ParteModelo,
    params: dict[str, Any],
    usar_peso: bool,
) -> np.ndarray:
    """Ajusta o degrau nas linhas `treino` (posições) e prevê as linhas `teste`."""
    modelo = construir(degrau, parte, params)
    modelo.fit(
        cast(pd.DataFrame, X.iloc[treino]),
        cast(pd.Series, y.iloc[treino]),
        cast(pd.Series, w.iloc[treino]) if usar_peso else None,
    )
    return modelo.predict_proba(cast(pd.DataFrame, X.iloc[teste]))


def _intervalo_em_json(i: Intervalo) -> dict[str, Any]:
    return {"estimativa": i.estimativa, "ic_inf": i.ic_inf, "ic_sup": i.ic_sup, "b": i.b}


def _projecao(
    y: np.ndarray,
    w: np.ndarray,
    p: dict[Degrau, dict[float, np.ndarray]],
    cod_upa: np.ndarray,
    uf: np.ndarray,
    eventos: np.ndarray,
    b: int,
) -> dict[str, Any]:
    """Projeção do ganho D - C para o dobro dos eventos de 100%, com intervalo do bootstrap."""
    alvo = FATOR_DA_PROJECAO * float(eventos[-1])

    def ganhos(peso: np.ndarray) -> np.ndarray:
        return np.array(
            [
                log_loss_ponderado(y, p["C"][f], peso) - log_loss_ponderado(y, p["D"][f], peso)
                for f in FRACOES
            ]
        )

    estimativa = projetar(eventos, ganhos(w), alvo)
    replicas: list[float] = []
    falhas = 0
    for m in multiplicadores(cod_upa, uf, b, SEMENTE):
        valor = projetar(eventos, ganhos(w * m), alvo)
        if valor is None:
            falhas += 1
        else:
            replicas.append(valor)
    saida: dict[str, Any] = {
        "forma": _FORMA,
        "eventos_projetados": alvo,
        "replicas": b,
        "replicas_que_falharam": falhas,
        "aviso": _AVISO_PROJECAO,
    }
    if estimativa is None or not replicas:
        return {**saida, "ganho_projetado": None, "intervalo": None}
    baixo, alto = np.percentile(replicas, [2.5, 97.5])
    ic = Intervalo(estimativa, float(baixo), float(alto), len(replicas))
    return {**saida, "ganho_projetado": estimativa, "intervalo": _intervalo_em_json(ic)}


def etapa_curva_aprendizado(b: int = B_PADRAO) -> None:
    """Curva de aprendizado de C e D por fração de setores; grava o relatório e as previsões."""
    config: dict[str, Any] = {
        "o_que": "ganho do D (LightGBM) sobre o C (logística com curvas) em frações do treino",
        "regra": "exploratório; o resultado não troca o modelo servido (seção 27 do registro)",
        "fracoes": list(FRACOES),
        "fracoes_unidade": "setores (cod_upa) de treino de cada dobra externa, aninhados",
        "semente_das_fracoes": f"SEMENTE + {DESLOCAMENTO_DA_SEMENTE} + k, k = dobra externa",
        "parametros": "os escolhidos pela validação aninhada em cada dobra externa com 100% dos "
        "setores (escada_<parte>.json), reusados nas frações menores; nada é reescolhido",
        "metrica": "log loss e PR-AUC ponderados fora da dobra, nas 5 dobras concatenadas",
        "ganho": "D - C no log loss ponderado (positivo favorece o D), bootstrap de setores por UF",
        "bootstrap": {"b": b, "estrato": "uf", "unidade": "cod_upa"},
        "projecao": {
            "forma": _FORMA,
            "alvo": f"{FATOR_DA_PROJECAO:g}x os eventos de treino de 100%",
            "intervalo": "ajuste em cada réplica, mesmo multiplicador em todas as frações",
            "aviso": _AVISO_PROJECAO,
        },
        "edicao": "somente 2008-2009",
    }
    with experimento("curva_aprendizado", "exploratório", config, _pasta_relatorios()) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        usar_peso = _usar_peso_decidido()
        r.configuracao["usar_peso"] = usar_peso
        previsoes: dict[str, Any] = {}
        parametros_usados: dict[str, Any] = {}
        for parte in PARTES:
            escada = _relatorio(f"escada_{parte}", dobras)["metricas"]["degraus"]
            params: dict[Degrau, list[dict[str, Any]]] = {
                d: escada[d]["parametros_por_dobra"] for d in DEGRAUS
            }
            parametros_usados[parte] = params
            r.metricas[parte] = _curva_da_parte(
                parte, celulas, ocasioes, dobras, params, usar_peso[parte], b, previsoes
            )
            base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
            r.dados[parte] = _dados(base, alvo)
        r.configuracao["parametros_por_dobra"] = parametros_usados
        salvar_resultado(
            "curva_aprendizado",
            {
                "previsoes": previsoes,
                "fracoes": list(FRACOES),
                "hash_dobras": r.dados["dobras"]["hash"],
            },
        )


def _curva_da_parte(
    parte: Parte,
    celulas: pd.DataFrame,
    ocasioes: pd.DataFrame,
    dobras: pd.DataFrame,
    params: dict[Degrau, list[dict[str, Any]]],
    usar_peso: bool,
    b: int,
    previsoes: dict[str, Any],
) -> dict[str, Any]:
    base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
    X = _entrada(base, cast(ParteModelo, parte), extra=False)
    y_serie, w_serie = cast(pd.Series, base[alvo]), cast(pd.Series, base["peso_final"])
    y, w = y_serie.to_numpy(), w_serie.to_numpy()
    cod_upa, uf = base["cod_upa"].to_numpy(), base["uf"].to_numpy()
    por_linha = dobra_das_linhas(base, dobras)
    k_externo = int(por_linha.max()) + 1
    for d in DEGRAUS:
        if len(params[d]) != k_externo:
            raise ValueError(f"escada_{parte}.json não tem um conjunto de parâmetros por dobra")

    p: dict[Degrau, dict[float, np.ndarray]] = {
        d: {f: np.full(len(base), np.nan) for f in FRACOES} for d in DEGRAUS
    }
    eventos = {f: [] for f in FRACOES}
    for k in range(k_externo):
        teste = np.flatnonzero(por_linha == k)
        setores = fracoes_aninhadas(np.unique(cod_upa[por_linha != k]), k)
        for f in FRACOES:
            treino = np.flatnonzero((por_linha != k) & np.isin(cod_upa, setores[f]))
            eventos[f].append(int(np.sum(y[treino] == 1)))
            for d in DEGRAUS:
                print(
                    f"curva {parte}: dobra externa {k + 1} de {k_externo}, fração {f:.0%}, "
                    f"modelo {d} ({len(treino)} linhas)",
                    flush=True,
                )
                p[d][f][teste] = _ajustar_e_prever(
                    X, y_serie, w_serie, treino, teste, d, cast(ParteModelo, parte),
                    params[d][k], usar_peso,
                )  # fmt: skip
    for d in DEGRAUS:
        for f in FRACOES:
            _sem_nan(p[d][f], f"{parte} {d} fração {f}")
    previsoes[parte] = {f"{f:g}": {d: p[d][f] for d in DEGRAUS} for f in FRACOES}

    media_eventos = np.array([float(np.mean(eventos[f])) for f in FRACOES])
    fracoes_json = []
    for f, n_eventos in zip(FRACOES, media_eventos, strict=True):
        print(f"curva {parte}: intervalo do ganho, fração {f:.0%}", flush=True)
        ic = intervalo_ganho(log_loss_ponderado, False, y, p["D"][f], p["C"][f], w, cod_upa, uf, b)
        fracoes_json.append(
            {
                "fracao": f,
                "eventos_de_treino": float(n_eventos),
                "eventos_de_treino_por_dobra": eventos[f],
                "log_loss_C": log_loss_ponderado(y, p["C"][f], w),
                "log_loss_D": log_loss_ponderado(y, p["D"][f], w),
                "pr_auc_C": pr_auc_ponderada(y, p["C"][f], w),
                "pr_auc_D": pr_auc_ponderada(y, p["D"][f], w),
                "ganho_D_menos_C": _intervalo_em_json(ic),
            }
        )
    print(f"curva {parte}: projeção", flush=True)
    return {
        "fracoes": fracoes_json,
        "projecao": _projecao(y, w, p, cod_upa, uf, media_eventos, b),
    }
