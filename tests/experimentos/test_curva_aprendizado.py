"""Curva de aprendizado de C e D (tarefa 17a), em base sintética pequena."""

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from tests.experimentos.test_etapas import _json, pastas  # noqa: F401  (fixture)

from occasion.experimentos import curva_aprendizado, etapas
from occasion.experimentos.curva_aprendizado import FRACOES, fracoes_aninhadas
from occasion.validacao.dobras import dobra_das_linhas


def _setores(n: int) -> np.ndarray:
    return np.array([f"S{i:03d}" for i in range(n)])


def test_fracoes_sao_aninhadas_e_cobrem_o_treino() -> None:
    setores = _setores(40)
    f = fracoes_aninhadas(setores, k=2, semente=123)
    assert tuple(f) == FRACOES
    assert [len(f[x]) for x in FRACOES] == [10, 20, 30, 40]
    for menor, maior in pairwise(FRACOES):
        assert set(f[menor]) <= set(f[maior])
    assert set(f[1.0]) == set(setores)
    # a mesma semente repete; outra dobra externa permuta diferente
    assert list(fracoes_aninhadas(setores, 2, 123)[0.25]) == list(f[0.25])
    assert list(fracoes_aninhadas(setores, 3, 123)[0.25]) != list(f[0.25])


def test_fracoes_dao_pelo_menos_um_setor() -> None:
    f = fracoes_aninhadas(_setores(3), k=0, semente=1)
    assert all(len(v) >= 1 for v in f.values())


def test_curva_em_base_sintetica(pastas: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_escada("preferencia")
    etapas.etapa_escada("ocasiao")

    # nenhum setor das linhas avaliadas pode estar no treino de nenhuma fração
    chamadas: list[tuple[set, set, int]] = []
    original = curva_aprendizado._ajustar_e_prever

    def espiao(X, y, w, treino, teste, *args, **kwargs):  # type: ignore[no-untyped-def]
        chamadas.append(
            (set(X["cod_upa"].to_numpy()[treino]), set(X["cod_upa"].to_numpy()[teste]), len(treino))
        )
        return original(X, y, w, treino, teste, *args, **kwargs)

    monkeypatch.setattr(curva_aprendizado, "_ajustar_e_prever", espiao)
    curva_aprendizado.etapa_curva_aprendizado(b=20)
    assert len(chamadas) == 2 * 5 * 4 * 2  # partes x dobras x frações x modelos
    assert all(not (tr & te) for tr, te, _ in chamadas)

    rel = _json(pastas / "reports" / "experimentos" / "curva_aprendizado.json")
    assert rel["natureza"] == "exploratório"
    m = rel["metricas"]
    assert m["status"] == "ok"
    cfg = rel["configuracao"]
    for chave in ("fracoes", "semente_das_fracoes", "parametros", "projecao", "regra"):
        assert chave in cfg
    assert "extrapolação" in cfg["projecao"]["aviso"]
    assert rel["dados"]["dobras"]["hash"]
    for parte in ("ocasiao", "preferencia"):
        p = m[parte]
        assert [f["fracao"] for f in p["fracoes"]] == list(FRACOES)
        eventos = [f["eventos_de_treino"] for f in p["fracoes"]]
        assert eventos == sorted(eventos) and eventos[0] > 0
        for f in p["fracoes"]:
            for chave in ("log_loss_C", "log_loss_D", "pr_auc_C", "pr_auc_D", "ganho_D_menos_C"):
                assert chave in f
            assert {"estimativa", "ic_inf", "ic_sup"} <= set(f["ganho_D_menos_C"])
        proj = p["projecao"]
        assert proj["eventos_projetados"] == pytest.approx(2 * eventos[-1])
        assert proj["replicas_que_falharam"] >= 0
        assert proj["forma"]
    assert (pastas / "experimentos" / "curva_aprendizado.joblib").exists()


def test_setores_da_dobra_externa_ficam_fora_do_treino(pastas: Path) -> None:  # noqa: F811
    etapas.etapa_dobras()
    dobras = etapas._dobras()
    base = etapas._celulas()
    por_linha = dobra_das_linhas(base, dobras)
    for k in range(5):
        treino = np.unique(base["cod_upa"].to_numpy()[por_linha != k])
        f = fracoes_aninhadas(treino, k, 7)
        fora = set(base["cod_upa"].to_numpy()[por_linha == k])
        assert not (set(f[1.0]) & fora)


def test_projecao_reta_em_log() -> None:
    n = np.array([100.0, 200.0, 400.0, 800.0])
    ganho = 0.01 + 0.002 * np.log(n)
    p = curva_aprendizado.projetar(n, ganho, 1600.0)
    assert p == pytest.approx(0.01 + 0.002 * np.log(1600.0))
    assert curva_aprendizado.projetar(n, np.array([np.nan, 0, 0, 0]), 1600.0) is None


def test_recusa_escada_de_outras_dobras(pastas: Path) -> None:  # noqa: F811
    etapas.etapa_dobras()
    etapas.etapa_peso()
    with pytest.raises((FileNotFoundError, ValueError)):
        curva_aprendizado.etapa_curva_aprendizado(b=5)
    rel = json.loads(
        (pastas / "reports" / "experimentos" / "curva_aprendizado.json").read_text(encoding="utf-8")
    )
    assert rel["metricas"]["status"] == "falhou"
