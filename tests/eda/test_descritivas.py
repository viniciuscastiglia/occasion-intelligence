from typing import Any, get_args

import numpy as np
import pandas as pd
import pytest

from occasion.config import SEMENTE
from occasion.eda.descritivas import FAIXAS_ETARIAS, faixa_de_renda, faixa_etaria, taxa_por_grupo
from occasion.inference.contract import FaixaRenda


def test_faixa_etaria_nos_limites() -> None:
    idades = pd.Series([10, 17, 18, 29, 30, 60, 99])
    obtido = faixa_etaria(idades).tolist()
    assert obtido == ["10-17", "10-17", "18-29", "18-29", "30-44", "60+", "60+"]


def test_faixa_etaria_cobre_as_faixas_declaradas_e_preserva_o_indice() -> None:
    idades = pd.Series([12, 20, 44, 45, 59, 130], index=[5, 6, 7, 8, 9, 10])
    obtido = faixa_etaria(idades)
    assert list(obtido.index) == [5, 6, 7, 8, 9, 10]
    assert obtido.tolist() == ["10-17", "18-29", "30-44", "45-59", "45-59", "60+"]
    assert set(obtido) == set(FAIXAS_ETARIAS)


def test_faixa_etaria_recusa_idade_fora_das_faixas() -> None:
    with pytest.raises(ValueError, match="idade"):
        faixa_etaria(pd.Series([9, 20]))


def test_faixa_de_renda_fechada_a_esquerda() -> None:
    renda = pd.Series([0, 0.49, 0.5, 1, 1.99, 2, 5, 12])
    obtido = faixa_de_renda(renda).tolist()
    assert obtido == [
        "ate_meio_sm",
        "ate_meio_sm",
        "meio_a_1_sm",
        "de_1_a_2_sm",
        "de_1_a_2_sm",
        "de_2_a_5_sm",
        "mais_de_5_sm",
        "mais_de_5_sm",
    ]


def test_faixa_de_renda_usa_os_codigos_do_contrato() -> None:
    renda = pd.Series([0.1, 0.7, 1.5, 3.0, 8.0])
    assert faixa_de_renda(renda).tolist() == list(get_args(FaixaRenda))


def test_faixa_de_renda_recusa_renda_negativa_ou_ausente() -> None:
    with pytest.raises(ValueError, match="renda"):
        faixa_de_renda(pd.Series([1.0, -0.1]))
    with pytest.raises(ValueError, match="renda"):
        faixa_de_renda(pd.Series([1.0, np.nan]))


def _base(n_setores: int = 60, por_setor: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(SEMENTE)
    setor = np.repeat(np.arange(n_setores), por_setor)
    n = len(setor)
    return pd.DataFrame(
        {
            "cod_upa": [f"s{k}" for k in setor],
            "uf": setor % 3 + 11,
            "peso_final": rng.uniform(0.5, 3.0, n),
            "grupo": np.where(rng.random(n) < 0.5, "a", "b"),
            "y": rng.binomial(1, 0.2, n).astype(np.int8),
        }
    )


def test_taxa_por_grupo_colunas_contagens_e_taxa_ponderada() -> None:
    base = _base()
    obtido = taxa_por_grupo(base, "y", ["grupo"], b=50)
    assert list(obtido.columns) == ["grupo", "n", "positivos", "taxa", "ic_inf", "ic_sup"]
    assert obtido["grupo"].tolist() == ["a", "b"]
    linha: dict[str, Any]
    for linha in obtido.to_dict(orient="records"):  # type: ignore[assignment]
        em_grupo = (base["grupo"] == linha["grupo"]).to_numpy()
        y = base["y"].to_numpy()[em_grupo]
        w = base["peso_final"].to_numpy()[em_grupo]
        assert linha["n"] == int(em_grupo.sum())
        assert linha["positivos"] == int(y.sum())
        esperado = float((w * y).sum() / w.sum())
        assert linha["taxa"] == pytest.approx(esperado)
        assert linha["ic_inf"] <= linha["taxa"] <= linha["ic_sup"]
        assert linha["ic_inf"] < linha["ic_sup"]


def test_taxa_por_grupo_varias_colunas_e_reprodutivel() -> None:
    base = _base().assign(outro=lambda d: np.where(d["uf"] == 11, "x", "y"))
    a = taxa_por_grupo(base, "y", ["grupo", "outro"], b=30, semente=SEMENTE)
    b = taxa_por_grupo(base, "y", ["grupo", "outro"], b=30, semente=SEMENTE)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 4
    assert int(a["n"].to_numpy().sum()) == len(base)


def test_taxa_por_grupo_intervalo_e_o_bootstrap_de_setores_da_base_inteira() -> None:
    """O intervalo do grupo usa os setores de toda a base (estimação de domínio).

    Um setor do grupo `b` tem todas as suas linhas em `b`; mesmo assim a réplica sorteia setores
    de toda a UF, e o intervalo do grupo `a` precisa coincidir com o cálculo direto.
    """
    from occasion.validacao.bootstrap import intervalo_estatistica

    base = _base()
    w = base["peso_final"].to_numpy()
    wy = w * base["y"].to_numpy()
    em_a = (base["grupo"] == "a").to_numpy()

    def taxa_a(m: np.ndarray) -> float | None:
        den = float((m * w * em_a).sum())
        return float((m * wy * em_a).sum()) / den if den > 0 else None

    direto = intervalo_estatistica(
        taxa_a, base["cod_upa"].to_numpy(), base["uf"].to_numpy(), b=40, semente=SEMENTE
    )
    obtido: dict[str, Any] = taxa_por_grupo(base, "y", ["grupo"], b=40, semente=SEMENTE).to_dict(
        orient="records"
    )[0]  # type: ignore[assignment]
    assert obtido["taxa"] == pytest.approx(direto.estimativa)
    assert obtido["ic_inf"] == pytest.approx(direto.ic_inf)
    assert obtido["ic_sup"] == pytest.approx(direto.ic_sup)
