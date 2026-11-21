import numpy as np
import pandas as pd
import pytest

from occasion.avaliacao.erros import contextos_com_mais_erro


def _base(n_por_contexto: int = 600) -> pd.DataFrame:
    linhas = []
    for regiao in ("sul", "norte"):
        for local in ("em_casa", "fora_de_casa"):
            linhas += [{"regiao": regiao, "local": local}] * n_por_contexto
    base = pd.DataFrame(linhas)
    base["peso_final"] = np.tile([0.5, 1.5], len(base) // 2)
    return base


def test_erros_apontam_o_contexto_subestimado():
    base = _base()
    rng = np.random.default_rng(0)
    p = np.full(len(base), 0.1)
    y = (rng.random(len(base)) < 0.1).astype(float)
    sub = ((base["regiao"] == "sul") & (base["local"] == "fora_de_casa")).to_numpy()
    sup = ((base["regiao"] == "norte") & (base["local"] == "em_casa")).to_numpy()
    y[sub] = (rng.random(sub.sum()) < 0.4).astype(float)  # observado muito acima do previsto
    p[sup] = 0.5  # previsto muito acima do observado
    saida = contextos_com_mais_erro(base, y, p, ["regiao", "local"], n=1)
    primeiro = saida["subestimados"][0]
    assert (primeiro["regiao"], primeiro["local"]) == ("sul", "fora_de_casa")
    assert primeiro["taxa_observada"] > primeiro["taxa_prevista"] and primeiro["z"] > 0
    ultimo = saida["superestimados"][0]
    assert (ultimo["regiao"], ultimo["local"]) == ("norte", "em_casa")
    assert ultimo["z"] < 0


def test_z_confere_com_a_formula_e_o_peso_e_normalizado():
    base = _base(500).assign(peso_final=lambda d: d["peso_final"] * 1000.0)
    rng = np.random.default_rng(1)
    p = rng.uniform(0.05, 0.3, len(base))
    y = (rng.random(len(base)) < 0.2).astype(float)
    saida = contextos_com_mais_erro(base, y, p, ["regiao"], n=2, minimo_linhas=10)
    w = base["peso_final"].to_numpy() / base["peso_final"].mean()
    for linha in saida["subestimados"] + saida["superestimados"]:
        m = (base["regiao"] == linha["regiao"]).to_numpy()
        o, e = np.sum(w[m] * y[m]), np.sum(w[m] * p[m])
        assert linha["z"] == pytest.approx((o - e) / np.sqrt(np.sum(w[m] ** 2 * p[m] * (1 - p[m]))))
        assert linha["taxa_observada"] == pytest.approx(o / np.sum(w[m]))
        assert linha["taxa_prevista"] == pytest.approx(e / np.sum(w[m]))


def test_grupo_pequeno_fica_de_fora_e_listas_nao_se_repetem():
    base = _base(600)
    pequeno = pd.DataFrame({"regiao": ["centro_oeste"] * 30, "local": ["em_casa"] * 30})
    pequeno["peso_final"] = 1.0
    base = pd.concat([base, pequeno], ignore_index=True)
    y = np.zeros(len(base))
    y[-30:] = 1  # o grupo pequeno seria o mais subestimado
    p = np.full(len(base), 0.05)
    saida = contextos_com_mais_erro(base, y, p, ["regiao"], n=5)
    ocorrem = {l["regiao"] for l in saida["subestimados"] + saida["superestimados"]}
    assert "centro_oeste" not in ocorrem
    assert ocorrem == {"sul", "norte"}  # só dois grupos: as listas não repetem o mesmo grupo
    assert len(saida["subestimados"]) + len(saida["superestimados"]) == 2
