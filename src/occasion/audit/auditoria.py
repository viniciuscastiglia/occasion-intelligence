"""Auditoria dos dados, antes de qualquer modelo (spec, seção 4.5).

É só relato: nada aqui altera bases ou variáveis. Nenhuma coluna `y_` nem `tem_pizza` de
2017-2018 é lida; só as variáveis de contexto dessa edição e as contagens de exclusão já gravadas
em `reports/bases.json`.
"""

import json
import math
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from occasion.config import (
    DADOS_PROCESSADOS,
    EDICAO_TESTE,
    EDICAO_TREINO,
    RELATORIOS,
    SEMENTE,
)
from occasion.features.variaveis import COLUNAS_CONTEXTO

# Fonte das fórmulas de tamanho de amostra: Riley RD, Snell KIE, Ensor J et al. "Minimum sample
# size for developing a multivariable prediction model: Part II - binary and time-to-event
# outcomes", Statistics in Medicine 2019;38:1276-1296 (critério i, encolhimento global), e Riley
# et al., BMJ 2020;368:m441. As equações foram conferidas na reapresentação de Pate et al.
# (arXiv:2207.12892, eqs. 3, 4, 24 e 25), que reproduz os critérios de Riley et al.:
#   critério (i):  n = p / ((S - 1) * ln(1 - R2_cs_adj / S)), com S >= 0,9;
#   R2_cs máximo:  1 - exp(2 * lnL_nulo / n), com lnL_nulo = E ln(E/n) + (n-E) ln(1 - E/n);
#   sem estudo anterior, R2_cs_adj = 0,15 * R2_cs máximo (R2 de Nagelkerke de 0,15).
FRACAO_DO_MAXIMO: float = 0.15
ENCOLHIMENTO: float = 0.9

DOBRAS: int = 5
DIAS_AMOSTRADOS_POR_EDICAO: int = 15_000
_PERMUTACOES: int = 3
_LINHAS_PERMUTACAO: int = 20_000
_NUMERICAS_CATEGORICAS: frozenset[str] = frozenset({"uf", "dia_semana"})

# Item 5. Dicionários de variáveis conferidos nas duas edições (todos os registros de consumo,
# morador e domicílio). Não há trimestre nem data de coleta em nenhuma delas.
PERIODO_COLETA: dict[str, Any] = {
    "pergunta": "algum registro informa o período da coleta?",
    "2008-2009": {
        "informa": True,
        "campo": "PERIODO_REAL",
        "registro": "T_DOMICILIO_S (domicílio; posição 41, tamanho 4)",
        "descricao": (
            "semana em que o domicílio foi efetivamente pesquisado, definida pelo primeiro dia "
            "de preenchimento da caderneta de despesa coletiva. Está no registro do domicílio, "
            "não no de consumo; a junção com o consumo não faz parte das bases atuais."
        ),
    },
    "2017-2018": {
        "informa": False,
        "campo": None,
        "registro": None,
        "descricao": (
            "não há: os registros de consumo, morador e domicílio não trazem trimestre, mês nem "
            "data de coleta. O consumo só informa o dia da semana (DIA_SEMANA)."
        ),
    },
    "uso": "se existir, serve só para checar sazonalidade na EDA; nenhuma variável é criada aqui.",
}

# Item 6. O comentário equivalente está em `ingest/layouts.py` (tarefa 5).
DEFINICOES: dict[str, Any] = {
    "local": {
        "2008-2009": (
            "V9018 é a fonte do alimento: 1 dentro e 2 fora do domicílio. Não é o local da "
            "refeição."
        ),
        "2017-2018": (
            "V9018 é o local da refeição: 1 em casa e 2 a 7 fora de casa; 9 é não determinado, "
            "contado como exclusão."
        ),
        "consequencia": (
            "as duas edições são mapeadas para em_casa e fora_de_casa, mas a medida não é a "
            "mesma; parte da diferença entre as edições pode vir daí."
        ),
    },
    "ocasiao": "dia + hora nas duas edições (decisão R12 do autor).",
    "refrigerante": (
        "definido pelo produto, genérico ou de marca, com a mesma regra nas duas edições "
        "(decisão R13 do autor, documentada em config.CODIGOS)."
    ),
}


def r2_cs_maximo(n: int, eventos: int) -> float:
    """Maior R² de Cox-Snell possível para a prevalência observada (Riley et al. 2019)."""
    if n <= 0 or not 0 < eventos < n:
        raise ValueError(f"é preciso 0 < eventos < n, veio eventos={eventos}, n={n}")
    ln_l0 = eventos * math.log(eventos / n) + (n - eventos) * math.log(1 - eventos / n)
    return 1 - math.exp(2 * ln_l0 / n)


def parametros_sustentados(n: int, r2_cs: float, encolhimento: float = ENCOLHIMENTO) -> int:
    """Parâmetros que `n` sustenta com encolhimento global `encolhimento` (critério i de Riley).

    Resolve `n = p / ((S - 1) * ln(1 - R2_cs / S))` para `p`, arredondando para baixo.
    """
    if n <= 0:
        raise ValueError(f"n deve ser positivo, veio {n}")
    if not 0 < encolhimento < 1:
        raise ValueError(f"o encolhimento deve estar entre 0 e 1, veio {encolhimento}")
    if not 0 < r2_cs < encolhimento:
        raise ValueError(f"R² de Cox-Snell fora de (0, {encolhimento}): {r2_cs}")
    return math.floor(n * (encolhimento - 1) * math.log(1 - r2_cs / encolhimento))


def tamanho_de_amostra(n: int, eventos: int) -> dict[str, float | int]:
    """Contagens simples, R² máximo, R² usado (15% do máximo) e parâmetros sustentados."""
    maximo = r2_cs_maximo(n, eventos)
    usado = FRACAO_DO_MAXIMO * maximo
    return {
        "n": n,
        "eventos": eventos,
        "r2_cs_maximo": maximo,
        "r2_cs_usado": usado,
        "parametros_sustentados": parametros_sustentados(n, usado),
    }


def _preparar(
    treino: pd.DataFrame, teste: pd.DataFrame, colunas: list[str]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Empilha as duas bases, com categorias comuns, e devolve X, rótulo da edição e setores."""
    juntas = pd.concat([treino, teste], ignore_index=True)
    x = cast(pd.DataFrame, juntas[colunas]).copy()
    for coluna in colunas:
        serie = cast(pd.Series, x[coluna])
        if coluna in _NUMERICAS_CATEGORICAS or not pd.api.types.is_numeric_dtype(serie):
            x[coluna] = serie.astype(str).astype("category")
    rotulo = np.r_[np.zeros(len(treino), dtype=int), np.ones(len(teste), dtype=int)]
    # Os setores de edições diferentes são distintos mesmo que o código coincida.
    setores = np.r_[
        ("a|" + treino["cod_upa"].astype(str)).to_numpy(),
        ("b|" + teste["cod_upa"].astype(str)).to_numpy(),
    ]
    return x, rotulo, setores


def validacao_adversarial(
    treino: pd.DataFrame, teste: pd.DataFrame, colunas: list[str], semente: int
) -> dict[str, Any]:
    """AUC de um classificador que distingue as linhas de `treino` das de `teste`.

    Usa só `colunas` (variáveis de entrada; colunas `y_*` e `tem_pizza` são recusadas). Validação
    cruzada de 5 dobras estratificada e agrupada por `cod_upa`: uma pessoa-dia aparece em várias
    linhas com as mesmas variáveis, e dobras aleatórias inflariam a AUC por memorização.
    Importância por permutação (queda da AUC) média nas dobras, em ordem decrescente.
    """
    proibidas = [c for c in colunas if c.startswith("y_") or c == "tem_pizza"]
    if proibidas:
        raise ValueError(f"a validação adversarial não usa target nem tem_pizza: {proibidas}")
    x, rotulo, setores = _preparar(treino, teste, colunas)
    dobras = StratifiedGroupKFold(n_splits=DOBRAS, shuffle=True, random_state=semente)
    previsto = np.zeros(len(x))
    queda = np.zeros(len(colunas))
    aucs: list[float] = []
    rng = np.random.default_rng(semente)
    for ordem, (i_treino, i_teste) in enumerate(dobras.split(x, rotulo, groups=setores)):
        modelo = HistGradientBoostingClassifier(
            categorical_features="from_dtype",
            max_iter=100,
            early_stopping=cast(Any, False),
            random_state=semente,
        )
        modelo.fit(x.iloc[i_treino], rotulo[i_treino])
        previsto[i_teste] = modelo.predict_proba(x.iloc[i_teste])[:, 1]
        aucs.append(float(roc_auc_score(rotulo[i_teste], previsto[i_teste])))
        sorteio = rng.choice(i_teste, size=min(_LINHAS_PERMUTACAO, len(i_teste)), replace=False)
        perm = permutation_importance(
            modelo,
            x.iloc[sorteio],
            rotulo[sorteio],
            scoring="roc_auc",
            n_repeats=_PERMUTACOES,
            random_state=semente + ordem,
        )
        queda += np.asarray(perm["importances_mean"]) / DOBRAS
    importancias = sorted(
        ({"variavel": c, "valor": float(v)} for c, v in zip(colunas, queda, strict=True)),
        key=lambda item: item["valor"],
        reverse=True,
    )
    return {
        "auc": float(np.mean(aucs)),
        "auc_por_dobra": aucs,
        "importancias": importancias,
        "dobras": {
            "n_dobras": DOBRAS,
            "esquema": "StratifiedGroupKFold(shuffle=True)",
            "agrupamento": "cod_upa",
            "semente": semente,
        },
        "importancia_metodo": f"permutação, queda média da AUC, {_PERMUTACOES} repetições",
        "linhas": {EDICAO_TREINO: len(treino), EDICAO_TESTE: len(teste)},
    }


def _disponibilidade(base: str) -> dict[str, dict[str, dict[str, Any]]]:
    """Presença e fração de nulos de cada coluna de contexto, por edição (só contexto)."""
    saida: dict[str, dict[str, dict[str, Any]]] = {c: {} for c in COLUNAS_CONTEXTO}
    for edicao in (EDICAO_TREINO, EDICAO_TESTE):
        caminho = DADOS_PROCESSADOS / f"{base}_{edicao}.parquet"
        quadro = pd.read_parquet(caminho, columns=COLUNAS_CONTEXTO)
        for coluna in COLUNAS_CONTEXTO:
            serie = cast(pd.Series, quadro[coluna])
            saida[coluna][edicao] = {"presente": True, "fracao_nulos": float(serie.isna().mean())}
    return saida


def _amostra_de_dias(quadro: pd.DataFrame, dias: int, semente: int) -> pd.DataFrame:
    """Sorteia `dias` dias com a semente e leva todas as células de cada dia sorteado."""
    ids = np.sort(np.asarray(cast(pd.Series, quadro["id_dia"]).unique()))
    if len(ids) <= dias:
        return quadro
    sorteados = np.random.default_rng(semente).choice(ids, size=dias, replace=False)
    return cast(pd.DataFrame, quadro[cast(pd.Series, quadro["id_dia"]).isin(list(sorteados))])


def _ler_contexto(base: str, edicao: str, extras: list[str]) -> pd.DataFrame:
    """Lê só contexto, setor e identificação; nunca `y_*` nem `tem_pizza`."""
    return pd.read_parquet(
        DADOS_PROCESSADOS / f"{base}_{edicao}.parquet",
        columns=[*extras, "cod_upa", *COLUNAS_CONTEXTO],
    )


def rodar_auditoria() -> dict[str, Any]:
    """Roda a auditoria e grava `reports/auditoria.json`."""
    bases = json.loads((RELATORIOS / "bases.json").read_text(encoding="utf-8"))

    celulas_08 = pd.read_parquet(
        DADOS_PROCESSADOS / f"celulas_{EDICAO_TREINO}.parquet", columns=["y_pizza_refri"]
    )
    refri_08 = pd.read_parquet(
        DADOS_PROCESSADOS / f"ocasioes_refri_{EDICAO_TREINO}.parquet",
        columns=["tem_pizza", "y_guarana"],
    )
    pizza_08 = cast(pd.DataFrame, refri_08[refri_08["tem_pizza"] == 1])
    amostra = {
        "parte1": tamanho_de_amostra(
            len(celulas_08), int(np.sum(celulas_08["y_pizza_refri"].to_numpy()))
        ),
        "parte2": tamanho_de_amostra(len(refri_08), int(np.sum(refri_08["y_guarana"].to_numpy()))),
        "parte2_recorte_pizza": tamanho_de_amostra(
            len(pizza_08), int(np.sum(pizza_08["y_guarana"].to_numpy()))
        ),
    }

    c_08 = _amostra_de_dias(
        _ler_contexto("celulas", EDICAO_TREINO, ["id_dia"]), DIAS_AMOSTRADOS_POR_EDICAO, SEMENTE
    )
    c_17 = _amostra_de_dias(
        _ler_contexto("celulas", EDICAO_TESTE, ["id_dia"]), DIAS_AMOSTRADOS_POR_EDICAO, SEMENTE
    )
    o_08 = _ler_contexto("ocasioes_refri", EDICAO_TREINO, [])
    o_17 = _ler_contexto("ocasioes_refri", EDICAO_TESTE, [])
    adversarial = {
        "parte1": {
            **validacao_adversarial(c_08, c_17, COLUNAS_CONTEXTO, SEMENTE),
            "amostragem": (
                f"{DIAS_AMOSTRADOS_POR_EDICAO} dias por edição (semente {SEMENTE}), com as 10 "
                "células de cada dia"
            ),
        },
        "parte2": {
            **validacao_adversarial(o_08, o_17, COLUNAS_CONTEXTO, SEMENTE),
            "amostragem": "todas as ocasiões com refrigerante, sem amostragem",
        },
    }

    relatorio: dict[str, Any] = {
        "disponibilidade": {
            "celulas": _disponibilidade("celulas"),
            "ocasioes_refri": _disponibilidade("ocasioes_refri"),
        },
        "exclusoes": bases,
        "tamanho_de_amostra": {
            "edicao": EDICAO_TREINO,
            "criterio": "Riley et al. 2019, critério (i), encolhimento 0,9",
            "r2_cs_usado_como": f"{FRACAO_DO_MAXIMO} do R² de Cox-Snell máximo",
            "contagens": "simples, sem peso amostral",
            **amostra,
        },
        "validacao_adversarial": adversarial,
        "periodo_coleta": PERIODO_COLETA,
        "definicoes": DEFINICOES,
    }
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    (RELATORIOS / "auditoria.json").write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return relatorio
