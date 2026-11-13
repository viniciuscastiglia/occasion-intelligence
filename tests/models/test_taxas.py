"""Testes do degrau A (tabela de taxas), com dados sintéticos."""

import numpy as np
import pandas as pd
import pytest

from occasion.models.taxas import DIMENSOES_PARTE_1, DIMENSOES_PARTE_2, TabelaDeTaxas


def _treino_so_em_casa() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    X = pd.DataFrame({"local": ["em_casa"] * 4})
    return X, pd.Series([1, 0, 0, 0]), pd.Series([1.0] * 4)


def test_dimensoes_das_partes():
    assert DIMENSOES_PARTE_1 == ["dia_semana", "faixa_horario", "local"]
    assert DIMENSOES_PARTE_2 == ["regiao", "local"]


def test_taxa_ponderada_sem_suavizacao():
    X = pd.DataFrame({"local": ["em_casa"] * 4 + ["fora_de_casa"] * 4})
    y = pd.Series([1, 0, 0, 0, 1, 1, 0, 0])
    w = pd.Series([1.0] * 8)
    t = TabelaDeTaxas(["local"], forca_suavizacao=0.0).fit(X, y, w)
    p = t.predict_proba(pd.DataFrame({"local": ["em_casa", "fora_de_casa"]}))
    assert p.shape == (2,)
    assert p.tolist() == pytest.approx([0.25, 0.5])


def test_peso_amostral_muda_a_taxa():
    X = pd.DataFrame({"local": ["em_casa", "em_casa"]})
    y = pd.Series([1, 0])
    t = TabelaDeTaxas(["local"], 0.0).fit(X, y, pd.Series([3.0, 1.0]))
    assert t.predict_proba(X)[0] == pytest.approx(0.75)


def test_combinacao_nao_vista_recebe_a_taxa_geral():
    X, y, w = _treino_so_em_casa()
    t = TabelaDeTaxas(["local"], 50.0).fit(X, y, w)
    p = t.predict_proba(pd.DataFrame({"local": ["fora_de_casa"]}))
    assert p[0] == pytest.approx(t.taxa_geral_) and np.isfinite(p).all()


def test_suavizacao_puxa_celula_pequena_para_a_taxa_geral():
    # 10 linhas em_casa (1 positivo) e 2 fora_de_casa (2 positivos); taxa geral = 3/12 = 0.25.
    X = pd.DataFrame({"local": ["em_casa"] * 10 + ["fora_de_casa"] * 2})
    y = pd.Series([1] + [0] * 9 + [1, 1])
    t = TabelaDeTaxas(["local"], forca_suavizacao=8.0).fit(X, y, pd.Series([1.0] * 12))
    assert t.taxa_geral_ == pytest.approx(0.25)
    p = t.predict_proba(pd.DataFrame({"local": ["em_casa", "fora_de_casa"]}))
    # Valor à mão: fora = (2 + 8 * 0.25) / (2 + 8) = 0.4 (a taxa crua seria 1.0).
    assert p[1] == pytest.approx(0.4)
    # em_casa = (1 + 8 * 0.25) / (10 + 8) = 3 / 18.
    assert p[0] == pytest.approx(3 / 18)
    assert 0.25 < p[1] < 1.0


def test_pesos_sao_normalizados_pela_media():
    X, y, w = _treino_so_em_casa()
    a = TabelaDeTaxas(["local"], 4.0).fit(X, y, w)
    b = TabelaDeTaxas(["local"], 4.0).fit(X, y, w * 1000)
    assert a.predict_proba(X) == pytest.approx(b.predict_proba(X))


def test_contribuicoes_tem_uma_coluna_por_dimensao_e_sinal_coerente():
    X = pd.DataFrame(
        {
            "local": ["em_casa"] * 6 + ["fora_de_casa"] * 6,
            "regiao": ["sul", "norte"] * 6,
        }
    )
    y = pd.Series([1, 1, 1, 0, 0, 0] + [0, 0, 0, 0, 0, 1])
    t = TabelaDeTaxas(["local", "regiao"], 2.0).fit(X, y, pd.Series([1.0] * 12))
    novo = pd.DataFrame({"local": ["em_casa", "fora_de_casa"], "regiao": ["sul", "norte"]})
    c = t.contribuicoes(novo)
    assert list(c.columns) == ["local", "regiao"]
    assert len(c) == 2
    assert c.loc[0, "local"] > 0  # em_casa tem taxa acima da geral
    assert c.loc[1, "local"] < 0
    g = t.taxa_geral_
    m = (3 + 2 * g) / (6 + 2)
    esperado = np.log(m / (1 - m)) - np.log(g / (1 - g))
    assert c.loc[0, "local"] == pytest.approx(esperado)


def test_valor_nao_visto_tem_contribuicao_zero():
    X, y, w = _treino_so_em_casa()
    t = TabelaDeTaxas(["local"], 5.0).fit(X, y, w)
    c = t.contribuicoes(pd.DataFrame({"local": ["fora_de_casa"]}))
    assert c.loc[0, "local"] == 0.0


def test_coluna_ausente_levanta_erro_e_coluna_a_mais_e_ignorada():
    X, y, w = _treino_so_em_casa()
    with pytest.raises(ValueError, match="regiao"):
        TabelaDeTaxas(["local", "regiao"]).fit(X, y, w)
    t = TabelaDeTaxas(["local"], 1.0).fit(X.assign(tem_pizza=1), y, w)
    com_extra = pd.DataFrame({"local": ["em_casa"], "tem_pizza": [1]})
    assert t.predict_proba(com_extra).shape == (1,)
    assert list(t.contribuicoes(com_extra).columns) == ["local"]
    with pytest.raises(ValueError, match="local"):
        t.predict_proba(pd.DataFrame({"outra": [1]}))
    with pytest.raises(ValueError, match="local"):
        t.contribuicoes(pd.DataFrame({"outra": [1]}))


@pytest.mark.parametrize("pesos", [[1.0, -1.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0] * 4])
def test_pesos_invalidos_levantam_erro(pesos: list[float]):
    X, y, _ = _treino_so_em_casa()
    with pytest.raises(ValueError, match="peso"):
        TabelaDeTaxas(["local"]).fit(X, y, pd.Series(pesos))


def test_nan_nas_dimensoes_e_recusado_mas_valor_nao_visto_vai_para_a_taxa_geral():
    X = pd.DataFrame({"a": ["x", "y", "x", "y"], "b": [1, 1, 2, 2]})
    y = pd.Series([1, 0, 0, 0])
    w = pd.Series([1.0, 1.0, 1.0, 1.0])
    modelo = TabelaDeTaxas(["a", "b"]).fit(X, y, w)
    com_nan = pd.DataFrame({"a": ["x"], "b": [float("nan")]})
    with pytest.raises(ValueError, match="NaN"):
        modelo.predict_proba(com_nan)
    with pytest.raises(ValueError, match="NaN"):
        modelo.contribuicoes(com_nan)
    nao_visto = pd.DataFrame({"a": ["z"], "b": [9]})
    assert modelo.predict_proba(nao_visto)[0] == pytest.approx(modelo.taxa_geral_)
