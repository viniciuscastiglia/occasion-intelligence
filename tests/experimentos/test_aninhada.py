"""Testes da validação aninhada: sem setor nos dois lados, dobras internas e semente."""

from typing import Any, Self

import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.config import SEMENTE
from occasion.experimentos.aninhada import (
    previsao_fora_da_dobra,
    selecao_final,
    validacao_aninhada,
)
from occasion.experimentos.especificacao import ParteModelo
from occasion.experimentos.regras import superioridade
from occasion.models.base import ModeloParte, sigmoide
from occasion.models.escada import Degrau, construir, espaco
from occasion.validacao.bootstrap import intervalo_ganho
from occasion.validacao.dobras import atribuir_dobras, dobra_das_linhas, tabela_de_upas
from occasion.validacao.metricas import log_loss_ponderado


def _base(
    n: int = 1500, semente: int = 0, com_pizza: bool = False, sinal: bool = True
) -> pd.DataFrame:
    """Base sintética com setores (`cod_upa`, vários por UF), `peso_final` e o alvo `y`."""
    X = contexto_sintetico(n, semente)
    rng = np.random.default_rng(semente + 5)
    # o setor pertence a uma só UF: 3 setores por UF
    X["cod_upa"] = [f"{uf}-{rng.integers(0, 3)}" for uf in X["uf"]]
    X["estrato"] = X["cod_upa"]
    if com_pizza:
        X["tem_pizza"] = rng.integers(0, 2, size=n)
    z = -1.0 + (1.2 * (X["faixa_horario"] == "noite") - 0.02 * (X["idade"] - 40) if sinal else 0.0)
    X["y"] = (rng.random(n) < sigmoide(np.asarray(z, dtype=float))).astype(int)
    X["peso_final"] = rng.uniform(0.5, 3.0, size=n)
    return X


@pytest.fixture
def base() -> pd.DataFrame:
    return _base()


@pytest.fixture
def dobras(base: pd.DataFrame) -> pd.DataFrame:
    return atribuir_dobras(tabela_de_upas(base), 5, SEMENTE)


class _Espiao:
    """Construtor que guarda os setores de cada `fit` e de cada `predict_proba`."""

    def __init__(self, interno: Any = None) -> None:
        self.modelos: list[_ModeloEspiao] = []
        self._interno = interno or construir

    def construir(self, degrau: Degrau, parte: ParteModelo, params: dict[str, Any]) -> ModeloParte:
        m = _ModeloEspiao(self._interno(degrau, parte, params))
        self.modelos.append(m)
        return m

    def pares(self) -> list[tuple[set[Any], set[Any]]]:
        return [(m.treino, p) for m in self.modelos for p in m.previstos]


class _ModeloEspiao:
    def __init__(self, real: ModeloParte) -> None:
        self.real = real
        self.treino: set[Any] = set()
        self.previstos: list[set[Any]] = []

    @property
    def variaveis(self) -> list[str]:
        return self.real.variaveis

    @property
    def base_(self) -> float:
        return self.real.base_

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> Self:
        self.treino = set(X["cod_upa"])
        self.real.fit(X, y, sample_weight)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.previstos.append(set(X["cod_upa"]))
        return self.real.predict_proba(X)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.real.contribuicoes(X)


def test_setor_nunca_dos_dois_lados(base: pd.DataFrame, dobras: pd.DataFrame) -> None:
    extra = _base(600, semente=3)
    espiao = _Espiao()
    validacao_aninhada(
        base,
        "y",
        "B",
        "ocasiao",
        dobras,
        usar_peso=True,
        bases_extras={"celulas": extra},
        construtor=espiao.construir,
    )
    pares = espiao.pares()
    assert len(pares) > 0
    for treino, previsto in pares:
        assert not (treino & previsto)


def test_dobras_internas_iguais_na_base_e_nas_extras(
    base: pd.DataFrame, dobras: pd.DataFrame
) -> None:
    extra = _base(600, semente=3)
    espiao = _Espiao()
    validacao_aninhada(
        base,
        "y",
        "B",
        "ocasiao",
        dobras,
        usar_peso=False,
        bases_extras={"celulas": extra},
        construtor=espiao.construir,
    )
    setores_base = set(base["cod_upa"])
    internos = [m for m in espiao.modelos if len(m.previstos) == 2]
    assert len(internos) > 0
    for m in internos:
        p_base, p_extra = m.previstos
        # um setor que está nas duas bases cai na mesma dobra interna
        assert (p_extra & setores_base) <= p_base
        assert p_base & p_extra


def test_oof_cobre_cada_linha_uma_vez_sem_nan(base: pd.DataFrame, dobras: pd.DataFrame) -> None:
    extra = _base(600, semente=3)
    r = validacao_aninhada(
        base, "y", "B", "ocasiao", dobras, usar_peso=True, bases_extras={"celulas": extra}
    )
    assert r.oof.shape == (len(base),)
    assert not np.isnan(r.oof).any()
    assert r.oof_extras["celulas"].shape == (len(extra),)
    assert not np.isnan(r.oof_extras["celulas"]).any()
    assert len(r.modelos) == len(r.parametros) == len(r.interno) == 5
    por_linha = dobra_das_linhas(base, dobras)
    for k, interno in enumerate(r.interno):
        assert set(interno.indices) == set(np.flatnonzero(por_linha != k))
        assert interno.p.shape == interno.indices.shape
        assert not np.isnan(interno.p).any()
        posicoes, p = interno.extras["celulas"]
        assert posicoes.shape == p.shape and not np.isnan(p).any()


def test_previsao_de_cada_linha_vem_de_modelo_que_nao_viu_o_setor(
    base: pd.DataFrame, dobras: pd.DataFrame
) -> None:
    r = validacao_aninhada(base, "y", "B", "ocasiao", dobras, usar_peso=True)
    por_linha = dobra_das_linhas(base, dobras)
    for k, modelo in enumerate(r.modelos):
        fora = por_linha == k
        linhas = base.loc[fora]
        esperado = modelo.predict_proba(linhas)
        np.testing.assert_allclose(r.oof[fora], esperado)


def test_preferencia_preve_extras_com_tem_pizza_um(dobras: pd.DataFrame) -> None:
    base = _base(1500, com_pizza=True)
    extra = _base(600, semente=3)  # sem a coluna tem_pizza
    r = validacao_aninhada(
        base, "y", "B", "preferencia", dobras, usar_peso=True, bases_extras={"c": extra}
    )
    por_linha = dobra_das_linhas(extra, dobras)
    for k, modelo in enumerate(r.modelos):
        fora = por_linha == k
        esperado = modelo.predict_proba(extra.loc[fora].assign(tem_pizza=1))
        np.testing.assert_allclose(r.oof_extras["c"][fora], esperado)
    assert "tem_pizza" not in extra.columns


def _lgbm_leve(degrau: Degrau, parte: ParteModelo, params: dict[str, Any]) -> ModeloParte:
    leve = {**params, "n_estimators": 20, "min_child_samples": 20}
    return construir(degrau, parte, leve)


@pytest.mark.parametrize("degrau", ["B", "D"])
def test_mesma_semente_mesmo_resultado_logistica_e_lightgbm(
    degrau: Degrau, base: pd.DataFrame, dobras: pd.DataFrame
) -> None:
    pequeno = base.iloc[:500]
    d = atribuir_dobras(tabela_de_upas(pequeno), 5, SEMENTE)

    def rodar() -> Any:
        return validacao_aninhada(
            pequeno, "y", degrau, "ocasiao", d, usar_peso=True, construtor=_lgbm_leve
        )

    a, b = rodar(), rodar()
    np.testing.assert_array_equal(a.oof, b.oof)
    assert a.parametros == b.parametros


def test_previsao_fora_da_dobra_e_selecao_final(base: pd.DataFrame, dobras: pd.DataFrame) -> None:
    configs = espaco("B", "ocasiao")
    oof, extras = previsao_fora_da_dobra(
        base, "y", "B", "ocasiao", configs[0], dobras, usar_peso=True
    )
    assert oof.shape == (len(base),) and not np.isnan(oof).any() and extras == {}

    extra = _base(600, semente=3)
    params, p, p_extras = selecao_final(
        base, "y", "B", "ocasiao", dobras, usar_peso=True, bases_extras={"celulas": extra}
    )
    perdas = [
        log_loss_ponderado(
            base["y"].to_numpy(),
            previsao_fora_da_dobra(base, "y", "B", "ocasiao", c, dobras, usar_peso=True)[0],
            base["peso_final"].to_numpy(),
        )
        for c in configs
    ]
    assert params == configs[int(np.argmin(perdas))]
    esperado, esperado_extras = previsao_fora_da_dobra(
        base, "y", "B", "ocasiao", params, dobras, usar_peso=True, bases_extras={"celulas": extra}
    )
    np.testing.assert_allclose(p, esperado)
    np.testing.assert_allclose(p_extras["celulas"], esperado_extras["celulas"])


def test_rotulos_embaralhados_nao_vencem_a_constante(dobras: pd.DataFrame) -> None:
    base = _base(1500, sinal=True)
    d = atribuir_dobras(tabela_de_upas(base), 5, SEMENTE)
    embaralhada = base.assign(y=np.random.default_rng(SEMENTE).permutation(base["y"].to_numpy()))
    r = validacao_aninhada(embaralhada, "y", "B", "ocasiao", d, usar_peso=True)
    y, w = embaralhada["y"].to_numpy(), embaralhada["peso_final"].to_numpy()
    constante = np.full(len(base), np.average(y, weights=w))
    ic = intervalo_ganho(
        log_loss_ponderado,
        False,
        y,
        r.oof,
        constante,
        w,
        embaralhada["cod_upa"].to_numpy(),
        embaralhada["uf"].to_numpy(),
        b=200,  # B reduzido só para o teste ser rápido
    )
    assert superioridade(ic) == "refutada"


class _ModeloNaN:
    """Modelo cuja previsão é NaN: a seleção não pode escolher uma configuração assim."""

    base_: float = float("nan")

    @property
    def variaveis(self) -> list[str]:
        return []

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> Self:
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), np.nan)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(index=X.index)


def _construtor_nan(degrau: Degrau, parte: ParteModelo, params: dict[str, Any]) -> ModeloParte:
    return _ModeloNaN()


def test_perda_nan_na_selecao_vira_erro(base: pd.DataFrame, dobras: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="perda"):
        validacao_aninhada(
            base, "y", "B", "ocasiao", dobras, usar_peso=True, construtor=_construtor_nan
        )
    with pytest.raises(ValueError, match="perda"):
        selecao_final(base, "y", "B", "ocasiao", dobras, usar_peso=True, construtor=_construtor_nan)
