"""Testes da auditoria: fórmulas de tamanho de amostra e validação adversarial."""

import numpy as np
import pandas as pd
import pytest

from occasion.audit.auditoria import (
    parametros_sustentados,
    r2_cs_maximo,
    tamanho_de_amostra,
    validacao_adversarial,
)
from occasion.config import SEMENTE


def _quadro(idades: np.ndarray, setores: np.ndarray, sexos: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"cod_upa": setores, "idade": idades, "sexo": sexos})


def _amostra(semente: int, n: int, media: float, prefixo: str) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    return _quadro(
        rng.normal(media, 8, n).round().astype(int),
        np.array([f"{prefixo}{i % 60:03d}" for i in range(n)]),
        rng.choice(["f", "m"], n),
    )


def test_parametros_sustentados_valor_calculado_a_mao():
    # 10000 * (0.9 - 1) * ln(1 - 0.02 / 0.9) = 22,47
    assert parametros_sustentados(10_000, 0.02) == 22


def test_parametros_sustentados_cresce_com_n():
    assert parametros_sustentados(20_000, 0.02) > parametros_sustentados(10_000, 0.02)


def test_r2_cs_maximo_com_prevalencia_de_meio():
    # lnL0 = n * ln(0.5); 1 - exp(2 * ln(0.5)) = 0.75
    assert r2_cs_maximo(1_000, 500) == pytest.approx(0.75)


def test_r2_cs_maximo_recusa_eventos_fora_do_intervalo():
    with pytest.raises(ValueError):
        r2_cs_maximo(100, 0)


def test_tamanho_de_amostra_usa_quinze_por_cento_do_maximo():
    resultado = tamanho_de_amostra(10_000, 500)
    assert resultado["n"] == 10_000
    assert resultado["eventos"] == 500
    assert resultado["r2_cs_usado"] == pytest.approx(0.15 * resultado["r2_cs_maximo"])
    assert resultado["parametros_sustentados"] == parametros_sustentados(
        10_000, resultado["r2_cs_usado"]
    )


def test_validacao_adversarial_distingue_bases_diferentes_e_nao_bases_iguais():
    a = _amostra(1, 4_000, 30, "a")
    outra_metade = _amostra(2, 4_000, 30, "b")
    jovens = _amostra(3, 4_000, 20, "j")
    idosos = _amostra(4, 4_000, 60, "i")
    iguais = validacao_adversarial(a, outra_metade, ["idade", "sexo"], SEMENTE)
    diferentes = validacao_adversarial(jovens, idosos, ["idade", "sexo"], SEMENTE)
    assert diferentes["auc"] > 0.95
    assert abs(iguais["auc"] - 0.5) < 0.1
    assert diferentes["importancias"][0]["variavel"] == "idade"
    assert set(diferentes["importancias"][0]) == {"variavel", "valor"}


def test_validacao_adversarial_recusa_coluna_de_target():
    a = _amostra(1, 200, 30, "a").assign(y_pizza_refri=0)
    b = _amostra(2, 200, 30, "b").assign(y_pizza_refri=0)
    with pytest.raises(ValueError):
        validacao_adversarial(a, b, ["idade", "y_pizza_refri"], SEMENTE)
    with pytest.raises(ValueError):
        validacao_adversarial(a.assign(tem_pizza=0), b.assign(tem_pizza=0), ["tem_pizza"], SEMENTE)


def test_validacao_adversarial_dobras_nao_dividem_setor():
    # Setor com variável que o identifica: sem agrupamento a AUC seria alta por memorização.
    # Com dobras agrupadas por setor, a variável de setor não ajuda a prever fora do setor.
    rng = np.random.default_rng(5)
    n = 4_000
    setores_a = np.array([f"a{i % 80:03d}" for i in range(n)])
    setores_b = np.array([f"b{i % 80:03d}" for i in range(n)])
    ruido_a = pd.Series(setores_a).map(dict(zip(sorted(set(setores_a)), rng.normal(size=80))))
    ruido_b = pd.Series(setores_b).map(dict(zip(sorted(set(setores_b)), rng.normal(size=80))))
    a = pd.DataFrame({"cod_upa": setores_a, "ruido": ruido_a})
    b = pd.DataFrame({"cod_upa": setores_b, "ruido": ruido_b})
    resultado = validacao_adversarial(a, b, ["ruido"], SEMENTE)
    assert abs(resultado["auc"] - 0.5) < 0.15
    assert resultado["dobras"]["agrupamento"] == "cod_upa"
