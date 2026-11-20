"""Testes das dobras por setor, espalhadas dentro de cada estrato."""

import numpy as np
import pandas as pd
import pytest

from occasion.config import SEMENTE
from occasion.validacao.dobras import (
    atribuir_dobras,
    conferir_dobras,
    dobra_das_linhas,
    dobras_internas,
    escolher_dobras,
    tabela_de_upas,
)


@pytest.fixture
def upas_com_estratos() -> pd.DataFrame:
    """60 setores em 6 estratos de tamanhos diferentes (inclui um estrato de 1 setor)."""
    tamanhos = [1, 4, 7, 11, 17, 20]
    linhas = []
    for e, n in enumerate(tamanhos):
        for i in range(n):
            linhas.append({"cod_upa": f"{e}-{i:03d}", "estrato": f"E{e}"})
    return pd.DataFrame(linhas)


@pytest.fixture
def base_sintetica(upas_com_estratos: pd.DataFrame) -> pd.DataFrame:
    """Três linhas por setor, com o mesmo setor repetido."""
    base = upas_com_estratos.loc[upas_com_estratos.index.repeat(3)].reset_index(drop=True)
    base["y_pizza_refri"] = (np.arange(len(base)) % 2).astype(int)
    return base


def test_tabela_de_upas_uma_linha_por_setor(base_sintetica, upas_com_estratos):
    upas = tabela_de_upas(base_sintetica)
    assert list(upas.columns) == ["cod_upa", "estrato"]
    assert upas["cod_upa"].is_unique
    assert list(upas.index) == list(range(len(upas)))
    assert set(upas["cod_upa"]) == set(upas_com_estratos["cod_upa"])


def test_setor_inteiro_numa_dobra_so(base_sintetica):
    dobras = atribuir_dobras(tabela_de_upas(base_sintetica), 5, SEMENTE)
    assert dobras["cod_upa"].is_unique
    por_linha = dobra_das_linhas(base_sintetica, dobras)
    assert bool(
        (pd.Series(por_linha).groupby(base_sintetica["cod_upa"].to_numpy()).nunique() == 1).all()
    )


def test_estratos_espalhados(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    por_estrato = d.groupby(["estrato", "dobra"]).size().unstack(fill_value=0)
    assert (por_estrato.max(axis=1) - por_estrato.min(axis=1)).max() <= 1


def test_dobras_do_mesmo_tamanho_no_total(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    tamanhos = d.groupby("dobra").size()
    assert set(tamanhos.index) == {0, 1, 2, 3, 4}
    assert np.ptp(tamanhos.to_numpy()) <= 1
    assert pd.api.types.is_integer_dtype(d["dobra"])


def test_mesma_semente_mesmas_dobras(upas_com_estratos):
    assert atribuir_dobras(upas_com_estratos, 5, 7).equals(atribuir_dobras(upas_com_estratos, 5, 7))


def test_semente_diferente_muda_dobras(upas_com_estratos):
    assert not atribuir_dobras(upas_com_estratos, 5, 7).equals(
        atribuir_dobras(upas_com_estratos, 5, 8)
    )


def test_dobras_internas_so_com_setores_de_treino(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    internas = dobras_internas(d, externa=2)
    assert set(internas["cod_upa"]) == set(d.loc[d["dobra"] != 2, "cod_upa"])
    assert set(internas["dobra"]) == {0, 1, 2}


def test_dobras_internas_vem_da_tabela_de_setores(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    esperado = atribuir_dobras(d.loc[d["dobra"] != 3, ["cod_upa", "estrato"]], 3, SEMENTE + 3)
    assert dobras_internas(d, externa=3).equals(esperado)


def test_setor_desconhecido_levanta_erro(base_sintetica, upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    base = pd.concat(
        [base_sintetica, pd.DataFrame({"cod_upa": ["X"], "estrato": ["E0"], "y_pizza_refri": [1]})]
    )
    with pytest.raises(ValueError, match="X"):
        dobra_das_linhas(base, dobras)


def test_dobra_das_linhas_devolve_a_dobra_do_setor(base_sintetica, upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    mapa = dobras.set_index("cod_upa")["dobra"]
    por_linha = dobra_das_linhas(base_sintetica, dobras)
    assert isinstance(por_linha, np.ndarray)
    assert (por_linha == base_sintetica["cod_upa"].map(mapa).to_numpy()).all()


@pytest.fixture
def base_com_positivos_num_setor(upas_com_estratos) -> pd.DataFrame:
    """Todos os positivos concentrados em um único setor."""
    base = upas_com_estratos.copy()
    base["y"] = 0
    base.loc[base.index[10], "y"] = 1
    return base


def test_conferencia_reprova_dobra_sem_positivos(base_com_positivos_num_setor, upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    r = conferir_dobras({"alvo": (base_com_positivos_num_setor, "y")}, dobras)
    assert r["alvo"]["ok"] is False
    assert r["geral"]["ok"] is False
    assert sum(r["alvo"]["positivos_por_dobra"].values()) == 1


def test_conferencia_aprova_positivos_espalhados(upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    base = upas_com_estratos.copy()
    base["y"] = dobra_das_linhas(base, dobras) % 2 * 0 + 1  # um positivo por setor
    r = conferir_dobras({"alvo": (base, "y")}, dobras)
    assert r["alvo"]["ok"] is True
    assert r["geral"]["ok"] is True
    assert sum(r["alvo"]["fracao_por_dobra"].values()) == pytest.approx(1.0)
    assert all(0.10 <= f <= 0.30 for f in r["alvo"]["fracao_por_dobra"].values())


def test_conferencia_alvo_sem_positivos_reprova_com_motivo(upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    base = upas_com_estratos.copy()
    base["y"] = 0
    r = conferir_dobras({"alvo": (base, "y")}, dobras)
    assert r["alvo"]["ok"] is False
    assert "positivo" in r["alvo"]["motivo"]


def test_conferencia_geral_reprova_se_um_alvo_reprova(upas_com_estratos):
    dobras = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    bom = upas_com_estratos.copy()
    bom["y"] = 1
    ruim = upas_com_estratos.copy()
    ruim["y"] = 0
    r = conferir_dobras({"bom": (bom, "y"), "ruim": (ruim, "y")}, dobras)
    assert r["bom"]["ok"] is True
    assert r["ruim"]["ok"] is False
    assert r["geral"]["ok"] is False


def test_escolher_dobras_usa_a_primeira_semente_que_aprova(upas_com_estratos):
    base = upas_com_estratos.copy()
    base["y"] = 1
    dobras, registro = escolher_dobras(upas_com_estratos, {"alvo": (base, "y")})
    assert registro["semente"] == SEMENTE
    assert len(registro["tentativas"]) == 1
    assert registro["tentativas"][0]["semente"] == SEMENTE
    assert dobras.equals(atribuir_dobras(upas_com_estratos, 5, SEMENTE))


def test_escolher_dobras_tenta_sementes_seguintes():
    """Com um único positivo, nenhuma dobra passa de 100% nem fica entre 10% e 30%:
    todas as tentativas reprovam e o erro traz as sementes tentadas."""
    upas = pd.DataFrame({"cod_upa": list("abcde"), "estrato": "E"})
    base = upas.copy()
    base["y"] = [1, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="tentativas") as exc:
        escolher_dobras(upas, {"alvo": (base, "y")}, tentativas=3)
    msg = str(exc.value)
    for s in (SEMENTE, SEMENTE + 100, SEMENTE + 200):
        assert str(s) in msg


def test_escolher_dobras_passa_para_a_semente_seguinte():
    """Com 10 setores e 5 positivos (setores 0 a 4), a primeira semente que
    espalha um positivo por dobra aprova; as anteriores reprovam."""
    upas = pd.DataFrame({"cod_upa": [f"u{i}" for i in range(10)], "estrato": "E"})
    base = upas.copy()
    base["y"] = [1] * 5 + [0] * 5
    dobras, registro = escolher_dobras(upas, {"alvo": (base, "y")}, tentativas=500)
    tentativas = registro["tentativas"]
    assert registro["semente"] == tentativas[-1]["semente"]
    assert [t["semente"] for t in tentativas] == [SEMENTE + 100 * i for i in range(len(tentativas))]
    assert tentativas[-1]["resultado"]["geral"]["ok"] is True
    assert all(not t["resultado"]["geral"]["ok"] for t in tentativas[:-1])
    assert dobras.equals(atribuir_dobras(upas, 5, registro["semente"]))
