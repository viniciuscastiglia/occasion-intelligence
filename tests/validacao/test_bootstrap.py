import itertools

import numpy as np
import pytest

from occasion.config import SEMENTE
from occasion.validacao.bootstrap import (
    intervalo_estatistica,
    intervalo_ganho,
    intervalo_por_unidades,
    multiplicadores,
)
from occasion.validacao.metricas import log_loss_ponderado, pr_auc_ponderada


@pytest.fixture
def base() -> dict[str, np.ndarray]:
    """50 mil linhas em 500 setores de 5 UFs; `p` é a probabilidade verdadeira."""
    rng = np.random.default_rng(SEMENTE)
    n_setores, n = 500, 50_000
    setor = rng.integers(0, n_setores, n)
    uf = (setor % 5).astype(int)  # cada setor pertence a uma só UF
    p = rng.uniform(0.02, 0.5, n)
    return {
        "y": rng.binomial(1, p),
        "p": p,
        "w": rng.uniform(0.5, 3.0, n),
        "upa": np.array([f"s{k}" for k in setor]),
        "uf": uf,
    }


def test_setor_sai_inteiro():
    upa = np.array(["a", "a", "b", "c"])
    uf = np.array([1, 1, 1, 2])
    for m in itertools.islice(multiplicadores(upa, uf, 50, SEMENTE), 50):
        assert m[0] == m[1]  # linhas do mesmo setor saem juntas
        assert m[3] == 1  # UF com um setor só: ele sai uma vez
        assert m[0] + m[2] == 2  # a UF 1 tem dois setores: dois sorteios


def test_multiplicadores_somam_o_numero_de_setores_por_grupo(base):
    upa, uf = base["upa"], base["uf"]
    m = next(multiplicadores(upa, uf, 1, SEMENTE))
    assert m.shape == upa.shape and m.dtype.kind == "i"
    for g in np.unique(uf):
        setores = np.unique(upa[uf == g])
        primeira_linha = [np.flatnonzero(upa == s)[0] for s in setores]
        assert m[primeira_linha].sum() == len(setores)


def test_multiplicadores_sao_deterministicos_e_geram_b_replicas():
    upa = np.array(["a", "b", "c", "d"])
    uf = np.array([1, 1, 1, 1])
    r1 = list(multiplicadores(upa, uf, 7, SEMENTE))
    r2 = list(multiplicadores(upa, uf, 7, SEMENTE))
    assert len(r1) == 7
    assert all(np.array_equal(a, b) for a, b in zip(r1, r2, strict=True))


def test_setor_em_dois_grupos_e_recusado():
    with pytest.raises(ValueError):
        next(multiplicadores(np.array(["a", "a"]), np.array([1, 2]), 1, SEMENTE))


def test_estatistica_estimativa_e_intervalo_coerentes(base):
    w = base["w"]
    ic = intervalo_estatistica(
        lambda m: float(np.sum(w * m) / np.sum(m)), base["upa"], base["uf"], b=200
    )
    assert ic.estimativa == pytest.approx(np.sum(w) / len(w))
    assert ic.ic_inf < ic.estimativa < ic.ic_sup
    assert ic.b == 200 and ic.nivel == 0.95
    assert ic.largura == pytest.approx(ic.ic_sup - ic.ic_inf)


def test_estatistica_descarta_replicas_none_e_conta_as_validas():
    upa = np.array(["a", "b", "c", "d"])
    uf = np.array([1, 1, 1, 1])
    chamadas = {"n": 0}

    def funcao(m: np.ndarray) -> float | None:
        chamadas["n"] += 1
        return None if chamadas["n"] % 2 == 0 else 1.0

    ic = intervalo_estatistica(funcao, upa, uf, b=10)
    assert ic.b == 5  # a chamada da estimativa é a primeira (válida); 5 das 10 réplicas passam
    assert ic.ic_inf == ic.ic_sup == 1.0


def test_estatistica_sem_replica_valida_levanta_erro():
    upa = np.array(["a", "b"])
    uf = np.array([1, 1])
    with pytest.raises(ValueError):
        intervalo_estatistica(lambda m: None, upa, uf, b=5)  # type: ignore[arg-type, return-value]


def test_ganho_nulo_quando_previsoes_iguais(base):
    ic = intervalo_ganho(
        log_loss_ponderado,
        False,
        base["y"],
        base["p"],
        base["p"],
        base["w"],
        base["upa"],
        base["uf"],
        b=200,
    )
    assert ic.estimativa == 0 and ic.ic_inf == 0 and ic.ic_sup == 0


def test_ganho_positivo_para_previsao_melhor(base):
    constante = np.full_like(base["p"], np.average(base["y"], weights=base["w"]))
    args = (base["y"], base["p"], constante, base["w"], base["upa"], base["uf"])
    perda = intervalo_ganho(log_loss_ponderado, False, *args, b=200)
    assert perda.estimativa > 0 and perda.ic_inf > 0
    ganho_pr = intervalo_ganho(pr_auc_ponderada, True, *args, b=200)  # type: ignore[arg-type]
    assert ganho_pr.estimativa > 0 and ganho_pr.ic_inf > 0
    # o sentido troca quando candidato e referência trocam de lugar
    invertido = intervalo_ganho(
        log_loss_ponderado,
        False,
        base["y"],
        constante,
        base["p"],
        base["w"],
        base["upa"],
        base["uf"],
        b=200,
    )
    assert invertido.ic_sup < 0


def test_intervalo_por_unidades_cobre_a_media():
    valores = np.arange(27, dtype=float)
    ic = intervalo_por_unidades(lambda idx: float(valores[idx].mean()), 27, b=500)
    assert ic.estimativa == pytest.approx(13.0)
    assert ic.ic_inf < 13.0 < ic.ic_sup and ic.b == 500


@pytest.mark.parametrize(
    ("upa", "uf", "o_que"),
    [
        (np.array(["a", None, "b"], dtype=object), np.array([1, 1, 2]), "setor"),
        (np.array([1.0, np.nan, 2.0]), np.array([1, 1, 2]), "setor"),
        (np.array(["a", "b", "c"]), np.array([1.0, np.nan, 2.0]), "grupo"),
        (np.array(["a", "b", "c"]), np.array([11, None, 35], dtype=object), "grupo"),
    ],
)
def test_ausente_no_setor_ou_no_grupo_e_recusado(upa, uf, o_que):
    with pytest.raises(ValueError, match=o_que):
        next(multiplicadores(upa, uf, 1, SEMENTE))
    with pytest.raises(ValueError, match=o_que):
        intervalo_estatistica(lambda m: float(m.sum()), upa, uf, b=2)
