"""Artefato avaliado montado da receita, com bases sintéticas pequenas (degraus B e D)."""

import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.config import FAIXAS_HORARIO, LOCAIS
from occasion.decision.engine import CORTE_OCASIAO, CORTE_PREFERENCIA
from occasion.experimentos.especificacao import variaveis_da_parte
from occasion.features.variaveis import DOMINIOS
from occasion.inference.contract import CODIGO_UF, Contexto, Resultado
from occasion.inference.predict import (
    ArtefatoModelo,
    carregar_artefato,
    contexto_para_linha,
    pontuar,
    prever,
    salvar_artefato,
    treinar_baseline,
)
from occasion.inference.treino import ajustar_parte, treinar_avaliado
from occasion.models.calibracao import ModeloCalibrado
from occasion.receita import Receita, ReceitaParte
from occasion.score.score import CORTE_ALTA, CORTE_MEDIA, classificar, probabilidades

_D = {
    "objective": "binary",
    "n_estimators": 15,
    "learning_rate": 0.2,
    "max_depth": 2,
    "num_leaves": 4,
    "min_child_samples": 20,
    "n_jobs": 1,
    "random_state": 1,
    "verbose": -1,
    "deterministic": True,
    "force_row_wise": True,
}
CONTEXTO = Contexto(
    uf="SP",
    situacao="urbano",
    dia_semana="sabado",
    faixa_horario="noite",
    local="fora_de_casa",
    idade=28,
    sexo="feminino",
    faixa_renda="de_2_a_5_sm",
    numero_moradores=3,
)


def contextos_validos() -> st.SearchStrategy[Contexto]:
    return st.builds(
        Contexto,
        uf=st.sampled_from(sorted(CODIGO_UF)),
        situacao=st.sampled_from(["urbano", "rural"]),
        dia_semana=st.sampled_from(
            ["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
        ),
        faixa_horario=st.sampled_from(list(FAIXAS_HORARIO)),
        local=st.sampled_from(list(LOCAIS)),
        idade=st.integers(18, 110),
        sexo=st.sampled_from(["masculino", "feminino"]),
        faixa_renda=st.sampled_from(
            ["ate_meio_sm", "meio_a_1_sm", "de_1_a_2_sm", "de_2_a_5_sm", "mais_de_5_sm"]
        ),
        numero_moradores=st.integers(1, 20),
    )


def _celulas(n: int = 500, edicao: str = "2008-2009") -> pd.DataFrame:
    rng = np.random.default_rng(3)
    base = contexto_sintetico(n, semente=3)
    base["faixa_horario"] = rng.choice(list(DOMINIOS["faixa_horario"]), size=n)
    z = -2.0 + 1.5 * (base["faixa_horario"] == "noite") + 0.8 * (base["local"] == "em_casa")
    p1 = 1 / (1 + np.exp(-z.to_numpy()))
    base["edicao"] = edicao
    base["peso_final"] = rng.uniform(500, 3000, size=n)
    base["y_pizza_refri"] = (rng.random(n) < p1).astype(int)
    base["y_pizza_guarana"] = (rng.random(n) < p1 * 0.3).astype(int)
    return base


def _ocasioes(n: int = 400, edicao: str = "2008-2009") -> pd.DataFrame:
    rng = np.random.default_rng(4)
    base = contexto_sintetico(n, semente=4)
    base["tem_pizza"] = rng.integers(0, 2, size=n)
    z = -1.0 + 0.9 * base["tem_pizza"] + 0.5 * (base["local"] == "fora_de_casa")
    base["edicao"] = edicao
    base["peso_final"] = rng.uniform(500, 3000, size=n)
    base["y_guarana"] = (rng.random(n) < 1 / (1 + np.exp(-z.to_numpy()))).astype(int)
    return base


def _parte(degrau: str, **mudancas: Any) -> ReceitaParte:
    parametros: dict[str, Any] = {"B": {"C": 1.0}, "D": _D}[degrau]
    campos: dict[str, Any] = {
        "degrau": degrau,
        "parametros": parametros,
        "usar_peso": True,
        "calibrar": True,
    }
    campos.update(mudancas)
    return ReceitaParte(**campos)


def _receita(**mudancas: Any) -> Receita:
    campos: dict[str, Any] = {
        "faixas_horario": {nome: list(faixa) for nome, faixa in FAIXAS_HORARIO.items()},
        "variaveis": {p: variaveis_da_parte(p) for p in ("ocasiao", "preferencia", "direto")},
        "interacoes": {"ocasiao": [], "preferencia": [], "direto": []},
        "ocasiao": _parte("D"),
        "preferencia": _parte("B"),
        "score": "produto",
        "direto": None,
        "calibrar_score": True,
        "referencia_a": {
            "ocasiao": {"dimensoes": ["faixa_horario", "local"], "suavizacao": 100},
            "preferencia": {"dimensoes": ["regiao"], "suavizacao": 1000},
        },
        "cortes": {
            "alta": CORTE_ALTA,
            "media": CORTE_MEDIA,
            "ocasiao": CORTE_OCASIAO,
            "preferencia": CORTE_PREFERENCIA,
        },
        "experimentos": [],
    }
    campos.update(mudancas)
    return Receita(**campos)


def _selecoes(n_celulas: int, n_ocasioes: int, com_direto: bool = False) -> dict[str, Any]:
    """Previsões fora da dobra fictícias, mas com ruído, para a calibração ter o que ajustar."""
    rng = np.random.default_rng(5)

    def oof(n: int, centro: float) -> np.ndarray:
        return np.clip(centro + rng.normal(0, 0.05, size=n), 0.01, 0.99)

    selecoes: dict[str, Any] = {
        "ocasiao": {
            "degrau": "D",
            "parametros": _D,
            "oof": oof(n_celulas, 0.3),
            "oof_extras": {},
            "hash_dobras": "h",
        },
        "preferencia": {
            "degrau": "B",
            "parametros": {"C": 1.0},
            "oof": oof(n_ocasioes, 0.4),
            "oof_extras": {"celulas": oof(n_celulas, 0.4)},
            "hash_dobras": "h",
        },
    }
    if com_direto:
        selecoes["direto"] = {
            "degrau": "B",
            "parametros": {"C": 1.0},
            "oof": oof(n_celulas, 0.1),
            "oof_extras": {},
            "hash_dobras": "h",
        }
    return selecoes


@cache
def _artefato_d_b() -> ArtefatoModelo:
    celulas, ocasioes = _celulas(), _ocasioes()
    return treinar_avaliado(
        _receita(), celulas, ocasioes, _selecoes(len(celulas), len(ocasioes)), "abc"
    )


@cache
def _artefato_direto() -> ArtefatoModelo:
    celulas, ocasioes = _celulas(), _ocasioes()
    a_ocasiao = {"dimensoes": ["faixa_horario", "local"], "suavizacao": 100}
    receita = _receita(
        score="direto",
        direto=_parte("B"),
        referencia_a={
            "ocasiao": a_ocasiao,
            "preferencia": {"dimensoes": ["regiao"], "suavizacao": 1000},
            "direto": a_ocasiao,
        },
    )
    return treinar_avaliado(
        receita, celulas, ocasioes, _selecoes(len(celulas), len(ocasioes), True), "abc"
    )


def test_treinar_avaliado_monta_as_partes_da_receita() -> None:
    a = _artefato_d_b()
    assert type(a.modelo_p1).__name__ == "ModeloCalibrado"
    assert isinstance(a.modelo_p2, ModeloCalibrado)
    assert a.calibracao_score is not None
    assert a.modelo_direto is None
    assert a.assinatura_receita is None
    assert a.metadados.modelo == "avaliado"
    assert a.metadados.treinado_em == ["2008-2009"]
    assert a.metadados.degrau.startswith("ocasião: D (LightGBM); preferência: B (logística)")


def test_degrau_descrito_em_portugues_com_curvas_e_interacoes() -> None:
    celulas, ocasioes = _celulas(), _ocasioes()
    receita = _receita(
        ocasiao=_parte("B", calibrar=False),
        preferencia=_parte("B", calibrar=False),
        calibrar_score=False,
    )
    selecoes = _selecoes(len(celulas), len(ocasioes))
    selecoes["ocasiao"]["degrau"] = "B"
    a = treinar_avaliado(receita, celulas, ocasioes, selecoes, "abc", "assinatura")
    assert a.metadados.degrau == "ocasião: B (logística); preferência: B (logística)"
    assert a.calibracao_score is None
    assert a.assinatura_receita == "assinatura"


def test_paridade_lote_e_prever_com_logistica_e_lightgbm() -> None:
    for artefato in (_artefato_d_b(), _artefato_direto()):
        ctx = CONTEXTO
        lote = pontuar(pd.concat([contexto_para_linha(ctx)] * 3, ignore_index=True), artefato)
        assert len(set(lote["score"].to_numpy().tolist())) == 1
        assert prever(ctx, artefato).occasion_score == pytest.approx(float(lote["score"].iloc[0]))


@settings(max_examples=25, deadline=None)
@given(contexto=contextos_validos())
def test_saida_respeita_o_contrato(contexto: Contexto) -> None:
    for artefato in (_artefato_d_b(), _artefato_direto()):
        r = prever(contexto, artefato)
        assert Resultado.model_validate(r.model_dump()) == r


@settings(max_examples=25, deadline=None)
@given(contexto=contextos_validos())
def test_score_entre_0_e_100_e_classificacao_coerente(contexto: Contexto) -> None:
    r = prever(contexto, _artefato_d_b())
    assert 0 <= r.occasion_score <= 100
    assert r.classification == classificar(r.occasion_score)


def test_coluna_que_nao_e_entrada_nao_muda_o_score() -> None:
    artefato = _artefato_d_b()
    linha = contexto_para_linha(CONTEXTO)
    a = pontuar(linha, artefato)
    extra = linha.assign(id_celula="x", peso_final=1.0, edicao="2017-2018")
    b = pontuar(extra, artefato)
    pd.testing.assert_frame_equal(a, b)


def test_ramo_direto_usa_fatores_do_direto() -> None:
    artefato = _artefato_direto()
    assert artefato.modelo_direto is not None
    r = prever(CONTEXTO, artefato)
    assert {f.parte for f in r.top_factors} == {"direto"}
    assert "tem_pizza" not in {f.variavel for f in r.top_factors}
    assert "fim_de_semana" not in {f.variavel for f in r.top_factors}
    p = probabilidades(
        contexto_para_linha(CONTEXTO),
        artefato.modelo_p1,
        artefato.modelo_p2,
        artefato.calibracao_score,
        artefato.modelo_direto,
    )
    assert r.occasion_probability == pytest.approx(float(p["p_produto"].iloc[0]))


def test_fatores_do_ramo_produto_nao_mostram_tem_pizza() -> None:
    r = prever(CONTEXTO, _artefato_d_b())
    assert {f.parte for f in r.top_factors} <= {"ocasiao", "preferencia"}
    assert "tem_pizza" not in {f.variavel for f in r.top_factors}


def test_artefato_do_plano_1_continua_carregando(tmp_path: Path) -> None:
    antigo = treinar_baseline(_celulas_baseline(), _ocasioes_baseline(), "abc")
    # um artefato do Plano 1 não tem os campos novos no `__dict__`
    for campo in ("calibracao_score", "modelo_direto", "assinatura_receita"):
        antigo.__dict__.pop(campo, None)
        assert campo not in antigo.__dict__
    salvar_artefato(antigo, tmp_path / "baseline")
    carregado = carregar_artefato(tmp_path)
    assert carregado.calibracao_score is None
    assert carregado.modelo_direto is None
    assert carregado.assinatura_receita is None
    ctx = CONTEXTO
    r = prever(ctx, carregado)
    assert 0 <= r.occasion_score <= 100
    assert "tem_pizza" not in {f.variavel for f in r.top_factors}
    assert len(pickle.dumps(carregado)) > 0


def _celulas_baseline() -> pd.DataFrame:
    from tests.inference.test_predict import _celulas_sinteticas

    return _celulas_sinteticas()


def _ocasioes_baseline() -> pd.DataFrame:
    from tests.inference.test_predict import _ocasioes_refri_sinteticas

    return _ocasioes_refri_sinteticas()


def test_calibracao_do_score_nao_muda_as_posicoes() -> None:
    celulas, ocasioes = _celulas(), _ocasioes()
    selecoes = _selecoes(len(celulas), len(ocasioes))
    com = _artefato_d_b()
    sem = treinar_avaliado(_receita(calibrar_score=False), celulas, ocasioes, selecoes, "abc")
    assert com.calibracao_score is not None
    assert com.calibracao_score[0] > 0
    X = celulas.sample(100, random_state=0)
    a, b = pontuar(X, com), pontuar(X, sem)
    assert not np.allclose(a["p_produto"], b["p_produto"])  # a probabilidade muda
    assert np.array_equal(a["score"].to_numpy(), b["score"].to_numpy())  # a posição, não
    assert np.array_equal(a["posicao_produto"].to_numpy(), b["posicao_produto"].to_numpy())


def test_treinar_avaliado_recusa_edicao_de_teste() -> None:
    celulas, ocasioes = _celulas(), _ocasioes()
    selecoes = _selecoes(len(celulas), len(ocasioes))
    with pytest.raises(ValueError, match="2008-2009"):
        treinar_avaliado(_receita(), _celulas(edicao="2017-2018"), ocasioes, selecoes, "abc")
    with pytest.raises(ValueError, match="2008-2009"):
        treinar_avaliado(_receita(), celulas, _ocasioes(edicao="2017-2018"), selecoes, "abc")


def test_treinar_avaliado_recusa_selecao_de_outro_degrau_ou_outras_dobras() -> None:
    celulas, ocasioes = _celulas(), _ocasioes()
    selecoes = _selecoes(len(celulas), len(ocasioes))
    selecoes["ocasiao"]["degrau"] = "B"
    with pytest.raises(ValueError, match="degrau"):
        treinar_avaliado(_receita(), celulas, ocasioes, selecoes, "abc")
    selecoes = _selecoes(len(celulas), len(ocasioes))
    selecoes["preferencia"]["hash_dobras"] = "outro"
    with pytest.raises(ValueError, match="dobras"):
        treinar_avaliado(_receita(), celulas, ocasioes, selecoes, "abc")


def test_ajustar_parte_calibra_com_a_previsao_fora_da_dobra_e_peso_final() -> None:
    celulas = _celulas()
    oof = np.clip(np.random.default_rng(6).uniform(0.05, 0.6, size=len(celulas)), 0.01, 0.99)
    calibrado = ajustar_parte(celulas, "y_pizza_refri", "ocasiao", _parte("B"), oof)
    assert isinstance(calibrado, ModeloCalibrado)
    puro = ajustar_parte(celulas, "y_pizza_refri", "ocasiao", _parte("B", calibrar=False), oof)
    assert not isinstance(puro, ModeloCalibrado)
    with pytest.raises(ValueError, match="NaN"):
        ajustar_parte(
            celulas, "y_pizza_refri", "ocasiao", _parte("B"), np.full(len(celulas), np.nan)
        )


def test_usar_peso_muda_o_ajuste() -> None:
    celulas = _celulas()
    oof = np.full(len(celulas), 0.3)
    com = ajustar_parte(celulas, "y_pizza_refri", "ocasiao", _parte("B", calibrar=False), oof)
    sem = ajustar_parte(
        celulas, "y_pizza_refri", "ocasiao", _parte("B", calibrar=False, usar_peso=False), oof
    )
    X = celulas.head(50)
    assert not np.allclose(com.predict_proba(X), sem.predict_proba(X))
