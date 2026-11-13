"""Testes do score (produto, posição e classificação), com dados sintéticos."""

import numpy as np
import pandas as pd
import pytest

from occasion.score.score import (
    ReferenciaScore,
    classificar,
    construir_referencia,
    probabilidades,
)


class ModeloFixo:
    def __init__(self, p: float) -> None:
        self.p = p

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.p)


class ModeloQueExigeTemPizza:
    """Devolve 0.5 só se `tem_pizza` for 1 em todas as linhas; senão, 0.1."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        ok = bool((X["tem_pizza"] == 1).all())
        return np.full(len(X), 0.5 if ok else 0.1)


class ModeloPorLocal:
    """Probabilidade depende da coluna `v`, para ordenar as linhas."""

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return X["v"].to_numpy(dtype=float)


def referencia_de(valores: list[float], pesos: list[float]) -> ReferenciaScore:
    celulas = pd.DataFrame({"v": valores, "peso_final": pesos})
    return construir_referencia(celulas, ModeloPorLocal(), ModeloFixo(1.0))


def test_posicao_e_fracao_ponderada():
    ref = referencia_de(valores=[0.1, 0.2, 0.3, 0.4], pesos=[1, 1, 1, 1])
    assert ref.posicao(0.2, "produto") == pytest.approx(50.0)
    assert ref.posicao(0.05, "produto") == 0.0
    assert ref.posicao(0.9, "produto") == 100.0


def test_posicao_respeita_o_peso():
    ref = referencia_de(valores=[0.1, 0.2], pesos=[3, 1])
    assert ref.posicao(0.1, "produto") == pytest.approx(75.0)


def test_posicao_com_empate_e_valor_maximo():
    ref = referencia_de(valores=[0.3, 0.1, 0.3, 0.2], pesos=[1, 1, 1, 1])
    assert ref.posicao(0.3, "produto") == 100.0
    assert ref.posicao(0.1, "produto") == pytest.approx(25.0)


def test_referencia_ordena_com_pesos_acompanhando():
    ref = referencia_de(valores=[0.4, 0.1], pesos=[1, 3])
    assert ref.produto.tolist() == [0.1, 0.4]
    assert ref.peso_acumulado["produto"].tolist() == pytest.approx([0.75, 1.0])
    assert ref.posicao(0.1, "ocasiao") == pytest.approx(75.0)
    assert ref.posicao(0.99, "preferencia") == 0.0
    assert ref.posicao(1.0, "preferencia") == 100.0


def test_produto_e_a_multiplicacao_e_usa_tem_pizza_igual_a_um():
    X = pd.DataFrame({"tem_pizza": [0, 0, 1]})
    p = probabilidades(X, ModeloFixo(0.02), ModeloQueExigeTemPizza())
    assert list(p.columns) == ["p_ocasiao", "p_preferencia", "p_produto"]
    assert (p.p_preferencia == 0.5).all()
    assert (p.p_produto == p.p_ocasiao * p.p_preferencia).all()
    assert p.p_produto.iloc[0] == pytest.approx(0.01)
    assert X["tem_pizza"].tolist() == [0, 0, 1]  # a entrada não é alterada


def test_probabilidades_sem_coluna_tem_pizza():
    X = pd.DataFrame({"local": ["em_casa", "fora_de_casa"]}, index=[7, 9])
    p = probabilidades(X, ModeloFixo(0.1), ModeloFixo(0.5))
    assert p.index.tolist() == [7, 9]
    assert p.p_produto.tolist() == pytest.approx([0.05, 0.05])


@pytest.mark.parametrize(
    "score,esperado", [(80.0, "alta"), (79.9, "média"), (50.0, "média"), (49.9, "baixa")]
)
def test_classificar(score, esperado):
    assert classificar(score) == esperado
