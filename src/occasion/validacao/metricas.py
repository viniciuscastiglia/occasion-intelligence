"""Métricas ponderadas por peso amostral: erro de probabilidade, ranking, calibração e topo.

Todas recebem `y`, `p` (ou score) e `w` como arrays 1-D do mesmo tamanho. As saídas de `resumo`
são tipos nativos do Python, prontas para JSON.
"""

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score, roc_auc_score

from occasion.models.base import EPS, logit, sigmoide
from occasion.score.score import CORTE_ALTA

__all__ = [
    "EPS",
    "ajustar_sigmoide",
    "brier_ponderado",
    "calibracao",
    "concentracao_topo",
    "log_loss_ponderado",
    "metricas_no_corte",
    "posicoes_ponderadas",
    "pr_auc_ponderada",
    "resumo",
    "roc_auc_ponderada",
]


def _arrays(
    y: np.ndarray, p: np.ndarray, w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=float), np.asarray(p, dtype=float), np.asarray(w, dtype=float)


def log_loss_ponderado(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Log loss médio ponderado, com `p` cortado em `[EPS, 1 - EPS]`."""
    y, p, w = _arrays(y, p, w)
    p = np.clip(p, EPS, 1.0 - EPS)
    perda = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    return float(np.sum(w * perda) / np.sum(w))


def brier_ponderado(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float:
    """Erro quadrático médio ponderado entre `p` e `y`."""
    y, p, w = _arrays(y, p, w)
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))


def _duas_classes(y: np.ndarray, w: np.ndarray) -> bool:
    """Há as duas classes entre as linhas de peso positivo."""
    return np.unique(np.asarray(y)[np.asarray(w) > 0]).size == 2


def pr_auc_ponderada(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    """Precisão média ponderada; `None` quando só há uma classe."""
    if not _duas_classes(y, w):
        return None
    return float(average_precision_score(y, p, sample_weight=w))


def roc_auc_ponderada(y: np.ndarray, p: np.ndarray, w: np.ndarray) -> float | None:
    """AUC ROC ponderada; `None` quando só há uma classe."""
    if not _duas_classes(y, w):
        return None
    return float(roc_auc_score(y, p, sample_weight=w))


def _nll_e_gradiente(
    parametros: np.ndarray, l: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, np.ndarray]:
    """Log-verossimilhança negativa média ponderada de `sigmoide(a·l + b)` e seu gradiente."""
    a, b = parametros
    z = a * l + b
    soma = np.sum(w)
    nll = np.sum(w * (np.logaddexp(0.0, z) - y * z)) / soma
    r = w * (sigmoide(z) - y) / soma
    return float(nll), np.array([np.sum(r * l), np.sum(r)])


def ajustar_sigmoide(p: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """`(inclinacao, intercepto)` de `logit(q) = inclinacao · logit(p) + intercepto`.

    Máxima verossimilhança ponderada, partindo de `(1, 0)`.
    """
    y, p, w = _arrays(y, p, w)
    resultado = minimize(
        _nll_e_gradiente, x0=np.array([1.0, 0.0]), args=(logit(p), y, w), jac=True, method="BFGS"
    )
    return float(resultado.x[0]), float(resultado.x[1])


def _intercepto_com_inclinacao_um(l: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    def objetivo(b: np.ndarray) -> tuple[float, np.ndarray]:
        nll, grad = _nll_e_gradiente(np.array([1.0, b[0]]), l, y, w)
        return nll, grad[1:]

    return float(minimize(objetivo, x0=np.array([0.0]), jac=True, method="BFGS").x[0])


def calibracao(y: np.ndarray, p: np.ndarray, w: np.ndarray, n_faixas: int = 10) -> dict[str, Any]:
    """Calibração em três níveis: global (`o_e`, `intercepto_cil`), inclinação e curva por faixas.

    As faixas são quantis ponderados de `p`; valores iguais de `p` ficam sempre na mesma faixa
    (uma faixa pode ficar maior que as outras) e faixas vazias são omitidas.
    """
    y, p, w = _arrays(y, p, w)
    inclinacao, _ = ajustar_sigmoide(p, y, w)
    intercepto_cil = _intercepto_com_inclinacao_um(logit(p), y, w)
    esperado = float(np.sum(w * p))
    o_e = float(np.sum(w * y) / esperado) if esperado > 0 else float("nan")

    _, inverso = np.unique(p, return_inverse=True)
    peso_por_valor = np.bincount(inverso, weights=w)
    acumulado = np.cumsum(peso_por_valor)
    meio = (acumulado - peso_por_valor / 2.0) / acumulado[-1]
    faixa_do_valor = np.minimum((meio * n_faixas).astype(int), n_faixas - 1)
    faixa = faixa_do_valor[inverso]

    curva: list[dict[str, Any]] = []
    for k in range(n_faixas):
        dentro = faixa == k
        if not dentro.any():
            continue
        wk = w[dentro]
        if np.sum(wk) <= 0:  # faixa só com linhas de peso zero (réplica de bootstrap)
            continue
        curva.append(
            {
                "p_media": float(np.sum(wk * p[dentro]) / np.sum(wk)),
                "taxa_observada": float(np.sum(wk * y[dentro]) / np.sum(wk)),
                "n": int(dentro.sum()),
                "positivos": int(np.sum(y[dentro] == 1)),
            }
        )
    return {
        "o_e": o_e,
        "intercepto_cil": intercepto_cil,
        "inclinacao": inclinacao,
        "curva": curva,
    }


def posicoes_ponderadas(valores: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Posição de 0 a 100 de cada valor dentro do próprio conjunto.

    Mesma regra de `ReferenciaScore.posicao`: fração ponderada dos valores menores ou iguais.
    """
    valores = np.asarray(valores, dtype=float)
    w = np.asarray(w, dtype=float)
    _, inverso = np.unique(valores, return_inverse=True)
    acumulado = np.cumsum(np.bincount(inverso, weights=w))
    return acumulado[inverso] / acumulado[-1] * 100.0


def metricas_no_corte(y: np.ndarray, alta: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    """Precision, recall e F1 ponderados para a previsão booleana `alta`, mais a matriz em pesos."""
    y = np.asarray(y) == 1
    alta = np.asarray(alta, dtype=bool)
    w = np.asarray(w, dtype=float)
    vp = float(np.sum(w[alta & y]))
    fp = float(np.sum(w[alta & ~y]))
    fn = float(np.sum(w[~alta & y]))
    vn = float(np.sum(w[~alta & ~y]))

    def razao(num: float, den: float) -> float | None:
        return num / den if den > 0 else None

    return {
        "precision": razao(vp, vp + fp),
        "recall": razao(vp, vp + fn),
        "f1": razao(2 * vp, 2 * vp + fp + fn),
        "matriz": {"vp": vp, "fp": fp, "fn": fn, "vn": vn},
    }


def concentracao_topo(
    y: np.ndarray, score: np.ndarray, w: np.ndarray, fracao: float = 0.10
) -> float | None:
    """Parcela ponderada dos positivos que cai nos `fracao` (em peso) de maior score.

    Um grupo de scores empatados que atravessa a fronteira entra em proporção: a fração do peso
    dele que cabe na fatia, aplicada aos positivos dele. Sem positivos, `None`.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    _, inverso = np.unique(np.asarray(score, dtype=float), return_inverse=True)
    peso_grupo = np.bincount(inverso, weights=w)[::-1]  # score decrescente
    positivos_grupo = np.bincount(inverso, weights=w * y)[::-1]
    total_positivos = float(positivos_grupo.sum())
    if total_positivos <= 0:
        return None
    antes = np.cumsum(peso_grupo) - peso_grupo
    orcamento = fracao * float(peso_grupo.sum())
    com_peso = peso_grupo > 0  # grupo de peso zero não ocupa fatia nem tem positivos ponderados
    cabe = np.zeros_like(peso_grupo)
    cabe[com_peso] = np.clip((orcamento - antes[com_peso]) / peso_grupo[com_peso], 0.0, 1.0)
    return float(np.sum(cabe * positivos_grupo) / total_positivos)


def resumo(
    y: np.ndarray, p: np.ndarray, w: np.ndarray, posicao: np.ndarray | None = None
) -> dict[str, Any]:
    """Todas as métricas de uma vez; com `posicao`, o corte da classe alta e o topo por posição."""
    y_f, _, w_f = _arrays(y, p, w)
    saida: dict[str, Any] = {
        "log_loss": log_loss_ponderado(y, p, w),
        "brier": brier_ponderado(y, p, w),
        "pr_auc": pr_auc_ponderada(y, p, w),
        "roc_auc": roc_auc_ponderada(y, p, w),
        "calibracao": calibracao(y, p, w),
        "prevalencia": float(np.sum(w_f * y_f) / np.sum(w_f)),
        "n": len(y_f),
        "positivos": int(np.sum(y_f == 1)),
    }
    if posicao is None:
        saida["concentracao_topo"] = concentracao_topo(y, p, w)
    else:
        saida["corte_alta"] = metricas_no_corte(y, np.asarray(posicao) >= CORTE_ALTA, w)
        saida["concentracao_topo"] = concentracao_topo(y, posicao, w)
    return saida
