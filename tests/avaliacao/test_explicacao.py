import numpy as np
import pandas as pd
import pytest

from occasion.avaliacao.explicacao import estabilidade, importancia_global


class _Falso:
    """Modelo falso: devolve contribuições fixas, uma coluna por variável."""

    def __init__(self, tabela: dict[str, np.ndarray | float]) -> None:
        self._tabela = tabela
        self.base_ = 0.0

    @property
    def variaveis(self) -> list[str]:
        return list(self._tabela)

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> "_Falso":
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), 0.5)

    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                v: np.broadcast_to(np.asarray(c, dtype=float), len(X))
                for v, c in self._tabela.items()
            },
            index=X.index,
        )


def _X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "local": ["em_casa", "em_casa", "fora_de_casa", "fora_de_casa"],
            "idade": [12, 20, 35, 70],
            "renda_pc_sm": [0.3, 1.5, 3.0, 8.0],
        }
    )


def test_importancia_global_ordena_pela_contribuicao():
    X = _X()
    w = np.array([1.0, 3.0, 1.0, 1.0])
    modelo = _Falso(
        {"local": np.array([0.1, -0.1, 0.1, -0.1]), "idade": np.array([1.0, -1.0, 0.0, 0.0]),
         "renda_pc_sm": 0.0}
    )  # fmt: skip
    imp = importancia_global(modelo, X, w)
    assert list(imp) == ["idade", "local", "renda_pc_sm"]
    assert imp["idade"] == pytest.approx((1 * 1 + 3 * 1) / 6)
    assert imp["local"] == pytest.approx(0.1)
    assert imp["renda_pc_sm"] == 0.0


def _por(lista: list[dict], variavel: str, nivel: str) -> dict:
    return next(l for l in lista if l["variavel"] == variavel and l["nivel"] == nivel)


def test_estabilidade_marca_sinal_trocado():
    X, w = _X(), np.ones(4)
    m1 = _Falso({"local": np.array([0.2, 0.2, -0.2, -0.2]), "idade": 0.0, "renda_pc_sm": 0.0})
    m2 = _Falso({"local": np.array([-0.2, -0.2, 0.2, 0.2]), "idade": 0.0, "renda_pc_sm": 0.0})
    linhas = estabilidade([m1, m2], X, w)
    em_casa = _por(linhas, "local", "em_casa")
    assert em_casa["contribuicao_por_modelo"] == pytest.approx([0.2, -0.2])
    assert em_casa["instavel"] is True
    assert _por(linhas, "idade", "10-17")["instavel"] is False


def test_sinal_trocado_abaixo_do_limite_nao_e_instavel_e_mesmo_sinal_tambem_nao():
    X, w = _X(), np.ones(4)
    pequeno = [
        _Falso({"local": 0.03, "idade": 0.0, "renda_pc_sm": 0.0}),
        _Falso({"local": -0.03, "idade": 0.0, "renda_pc_sm": 0.0}),
    ]
    assert not _por(estabilidade(pequeno, X, w), "local", "em_casa")["instavel"]
    grande_igual = [
        _Falso({"local": 0.4, "idade": 0.0, "renda_pc_sm": 0.0}),
        _Falso({"local": 0.1, "idade": 0.0, "renda_pc_sm": 0.0}),
    ]
    assert not _por(estabilidade(grande_igual, X, w), "local", "em_casa")["instavel"]


def test_estabilidade_usa_faixas_e_media_ponderada_por_nivel():
    X = _X()
    w = np.array([1.0, 3.0, 1.0, 1.0])
    contrib = np.array([0.0, 0.4, 0.0, 0.0])
    modelo = _Falso({"local": 0.0, "idade": contrib, "renda_pc_sm": contrib})
    linhas = estabilidade([modelo], X, w)
    # idades 12 e 20 caem em faixas diferentes; a média de cada nível usa os pesos das linhas
    assert _por(linhas, "idade", "10-17")["contribuicao_por_modelo"] == [0.0]
    assert _por(linhas, "idade", "18-29")["contribuicao_por_modelo"] == pytest.approx([0.4])
    em_casa = _por(linhas, "local", "em_casa")
    assert em_casa["n"] == 2
    niveis_renda = {l["nivel"] for l in linhas if l["variavel"] == "renda_pc_sm"}
    assert len(niveis_renda) == 4 and "3.0" not in niveis_renda
    peso_misto = estabilidade([_Falso({"local": np.array([0.0, 0.4, 0.2, 0.2])})], X, w)
    assert _por(peso_misto, "local", "em_casa")["contribuicao_por_modelo"] == pytest.approx(
        [(0 * 1 + 0.4 * 3) / 4]
    )
