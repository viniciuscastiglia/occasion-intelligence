"""Testes do artefato e da função de inferência, com bases sintéticas pequenas."""

from functools import cache
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from occasion.config import EDICAO_TREINO, FAIXAS_HORARIO, LOCAIS
from occasion.features.variaveis import regiao_da_uf
from occasion.inference.contract import CODIGO_UF, UF, Contexto, Resultado
from occasion.inference.predict import (
    N_UFS_MIDIA_REGIONAL,
    ArtefatoAusente,
    ArtefatoModelo,
    carregar_artefato,
    contexto_para_linha,
    pontuar,
    prever,
    salvar_artefato,
    treinar_baseline,
)
from occasion.score.score import classificar

CONTEXTO_VALIDO = {
    "uf": "SP",
    "situacao": "urbano",
    "dia_semana": "sabado",
    "faixa_horario": "noite",
    "local": "fora_de_casa",
    "idade": 28,
    "sexo": "feminino",
    "faixa_renda": "de_2_a_5_sm",
    "numero_moradores": 3,
}

_DIAS = list(range(1, 8))
_UFS = sorted(CODIGO_UF.values())


def _celulas_sinteticas(edicao: str = EDICAO_TREINO, n_pessoas: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    linhas = []
    for i in range(n_pessoas):
        uf = _UFS[i % len(_UFS)]
        dia = _DIAS[i % 7]
        for faixa in FAIXAS_HORARIO:
            for local in LOCAIS:
                p = 0.25 if (faixa == "noite" and local == "fora_de_casa") else 0.03
                linhas.append(
                    {
                        "edicao": edicao,
                        "peso_final": float(rng.uniform(500, 3000)),
                        "dia_semana": dia,
                        "faixa_horario": faixa,
                        "local": local,
                        "uf": uf,
                        "regiao": regiao_da_uf(uf),
                        "y_pizza_refri": int(rng.random() < p),
                    }
                )
    return pd.DataFrame(linhas)


def _ocasioes_refri_sinteticas(edicao: str = EDICAO_TREINO, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    uf = rng.choice(_UFS, size=n)
    return pd.DataFrame(
        {
            "edicao": edicao,
            "peso_final": rng.uniform(500, 3000, size=n),
            "uf": uf,
            "regiao": [regiao_da_uf(int(u)) for u in uf],
            "local": rng.choice(list(LOCAIS), size=n),
            "tem_pizza": rng.integers(0, 2, size=n),
            "y_guarana": (rng.random(n) < 0.3).astype(int),
        }
    )


@cache
def _artefato_unico() -> ArtefatoModelo:
    return treinar_baseline(_celulas_sinteticas(), _ocasioes_refri_sinteticas(), "abc")


@pytest.fixture
def celulas_2008_sintetica() -> pd.DataFrame:
    return _celulas_sinteticas()


@pytest.fixture
def ocasioes_refri_2008() -> pd.DataFrame:
    return _ocasioes_refri_sinteticas()


@pytest.fixture
def celulas_com_2017() -> pd.DataFrame:
    return pd.concat([_celulas_sinteticas(), _celulas_sinteticas("2017-2018", 5)])


@pytest.fixture
def artefato_sintetico(
    celulas_2008_sintetica: pd.DataFrame, ocasioes_refri_2008: pd.DataFrame
) -> ArtefatoModelo:
    return treinar_baseline(celulas_2008_sintetica, ocasioes_refri_2008, "abc")


def test_prever_devolve_resultado_valido(artefato_sintetico: ArtefatoModelo) -> None:
    r = prever(Contexto(**CONTEXTO_VALIDO), artefato_sintetico)  # type: ignore[arg-type]
    assert 0 <= r.occasion_score <= 100
    assert r.classification == classificar(r.occasion_score)
    assert r.occasion_probability == pytest.approx(
        r.score_components["ocasiao"].probabilidade
        * r.score_components["preferencia"].probabilidade
    )


def test_inferencia_e_lote_dao_o_mesmo_score(artefato_sintetico: ArtefatoModelo) -> None:
    ctx = Contexto(**CONTEXTO_VALIDO)  # type: ignore[arg-type]
    lote = pontuar(pd.concat([contexto_para_linha(ctx)] * 3, ignore_index=True), artefato_sintetico)
    assert len(set(lote["score"].to_numpy().tolist())) == 1
    assert prever(ctx, artefato_sintetico).occasion_score == pytest.approx(
        float(lote["score"].iloc[0])
    )


def test_campo_que_nao_e_entrada_do_modelo_nao_altera_o_score(
    artefato_sintetico: ArtefatoModelo,
) -> None:
    a = prever(Contexto(**CONTEXTO_VALIDO), artefato_sintetico)  # type: ignore[arg-type]
    b = prever(Contexto(**{**CONTEXTO_VALIDO, "versao_contrato": "9.9"}), artefato_sintetico)  # type: ignore[arg-type]
    assert a.occasion_score == b.occasion_score


def test_treinar_baseline_recusa_edicao_de_teste(
    celulas_com_2017: pd.DataFrame, ocasioes_refri_2008: pd.DataFrame
) -> None:
    with pytest.raises(ValueError):
        treinar_baseline(celulas_com_2017, ocasioes_refri_2008, "abc")


def test_treinar_baseline_recusa_edicao_de_teste_nas_ocasioes(
    celulas_2008_sintetica: pd.DataFrame,
) -> None:
    mistura = pd.concat([_ocasioes_refri_sinteticas(), _ocasioes_refri_sinteticas("2017-2018", 5)])
    with pytest.raises(ValueError):
        treinar_baseline(celulas_2008_sintetica, mistura, "abc")


def test_artefato_ausente_tem_mensagem_clara(tmp_path: Path) -> None:
    with pytest.raises(ArtefatoAusente, match="make baseline"):
        carregar_artefato(tmp_path / "nao_existe")


def test_treinar_duas_vezes_da_o_mesmo_artefato(
    celulas_2008_sintetica: pd.DataFrame, ocasioes_refri_2008: pd.DataFrame
) -> None:
    a = treinar_baseline(celulas_2008_sintetica, ocasioes_refri_2008, "abc")
    b = treinar_baseline(celulas_2008_sintetica, ocasioes_refri_2008, "abc")
    assert np.array_equal(a.referencia.produto, b.referencia.produto)
    assert a.modelo_p1.taxa_geral_ == b.modelo_p1.taxa_geral_
    assert a.modelo_p2.taxa_geral_ == b.modelo_p2.taxa_geral_
    assert a.ufs_midia_regional == b.ufs_midia_regional
    linha = contexto_para_linha(Contexto(**CONTEXTO_VALIDO))  # type: ignore[arg-type]
    assert np.array_equal(pontuar(linha, a), pontuar(linha, b))


def test_ufs_midia_regional_inclui_empatadas_com_a_setima(
    celulas_2008_sintetica: pd.DataFrame, ocasioes_refri_2008: pd.DataFrame
) -> None:
    # Pesos iguais: a mistura de local é a mesma em toda UF, então a parte 2 (região x local)
    # dá a mesma média às UFs da mesma região e o corte em 7 seria arbitrário.
    celulas = celulas_2008_sintetica.assign(peso_final=1000.0)
    ufs = treinar_baseline(celulas, ocasioes_refri_2008, "abc").ufs_midia_regional
    assert len(ufs) >= N_UFS_MIDIA_REGIONAL
    regioes = {regiao_da_uf(CODIGO_UF[cast(UF, u)]) for u in ufs}
    for uf, codigo in CODIGO_UF.items():
        assert (uf in ufs) == (regiao_da_uf(codigo) in regioes)


def test_salvar_e_carregar_preferem_servido(
    artefato_sintetico: ArtefatoModelo, tmp_path: Path
) -> None:
    salvar_artefato(artefato_sintetico, tmp_path / "avaliado")
    assert carregar_artefato(tmp_path).metadados.modelo == "avaliado"
    servido = ArtefatoModelo(
        artefato_sintetico.modelo_p1,
        artefato_sintetico.modelo_p2,
        artefato_sintetico.referencia,
        artefato_sintetico.ufs_midia_regional,
        artefato_sintetico.metadados.model_copy(update={"modelo": "servido"}),
    )
    salvar_artefato(servido, tmp_path / "servido")
    assert carregar_artefato(tmp_path).metadados.modelo == "servido"


def test_top_factors_sao_os_cinco_de_maior_valor_absoluto(
    artefato_sintetico: ArtefatoModelo,
) -> None:
    r = prever(Contexto(**CONTEXTO_VALIDO), artefato_sintetico)  # type: ignore[arg-type]
    assert len(r.top_factors) == 5
    absolutos = [abs(f.contribuicao) for f in r.top_factors]
    assert absolutos == sorted(absolutos, reverse=True)
    assert {f.parte for f in r.top_factors} <= {"ocasiao", "preferencia"}
    for f in r.top_factors:
        assert f.direcao == ("aumenta" if f.contribuicao > 0 else "diminui")


def test_nan_e_recusado(artefato_sintetico: ArtefatoModelo) -> None:
    linha = contexto_para_linha(Contexto(**CONTEXTO_VALIDO))  # type: ignore[arg-type]
    artefato_sintetico.modelo_p1.taxa_geral_ = float("nan")
    artefato_sintetico.modelo_p1._conjunta = artefato_sintetico.modelo_p1._conjunta * float("nan")
    with pytest.raises(ValueError, match="NaN"):
        pontuar(linha, artefato_sintetico)


def test_contexto_para_linha_converte_campos() -> None:
    linha = contexto_para_linha(Contexto(**CONTEXTO_VALIDO))  # type: ignore[arg-type]
    assert linha.iloc[0]["uf"] == 35
    assert linha.iloc[0]["regiao"] == "sudeste"
    assert linha.iloc[0]["dia_semana"] == 6
    assert linha.iloc[0]["renda_pc_sm"] == 3.5
    assert linha.iloc[0]["n_moradores_uc"] == 3


@settings(max_examples=40, deadline=None)
@given(
    ctx=st.builds(
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
)
def test_qualquer_contexto_valido_gera_resultado_valido(ctx: Contexto) -> None:
    r = prever(ctx, _artefato_unico())
    assert Resultado.model_validate(r.model_dump()) == r


def test_carregar_artefato_acha_o_baseline_sem_servido_nem_avaliado(
    artefato_sintetico: ArtefatoModelo, tmp_path: Path
) -> None:
    salvar_artefato(artefato_sintetico, tmp_path / "baseline")
    assert carregar_artefato(tmp_path).metadados.modelo == "avaliado"


def _artefato_logistico(artefato_base: ArtefatoModelo) -> ArtefatoModelo:
    from tests.models.bases_sinteticas import base_sintetica

    from occasion.experimentos.especificacao import INTERACOES
    from occasion.models.logistica import ModeloLogistico
    from occasion.score.score import construir_referencia

    X1, y1, w1 = base_sintetica(2000, semente=0)
    X2, y2, w2 = base_sintetica(2000, semente=10, com_pizza=True)
    p1 = ModeloLogistico("ocasiao", C=1.0, interacoes=INTERACOES["ocasiao"]["fim_de_semana"]).fit(
        X1, y1, w1
    )
    p2 = ModeloLogistico("preferencia", C=1.0).fit(X2, y2, w2)
    celulas = X1.assign(peso_final=w1)
    return ArtefatoModelo(
        modelo_p1=p1,
        modelo_p2=p2,
        referencia=construir_referencia(celulas, p1, p2),
        ufs_midia_regional=frozenset({"SP"}),
        metadados=artefato_base.metadados,
    )


def test_top_fatores_nao_mostra_tem_pizza(artefato_sintetico: ArtefatoModelo) -> None:
    artefato = _artefato_logistico(artefato_sintetico)
    r = prever(Contexto(**CONTEXTO_VALIDO), artefato)  # type: ignore[arg-type]
    assert "tem_pizza" not in {f.variavel for f in r.top_factors}
    assert len(r.top_factors) == 5
    assert "fim_de_semana" not in {f.variavel for f in r.top_factors}


def test_fator_fim_de_semana_aparece_como_dia_da_semana(
    artefato_sintetico: ArtefatoModelo,
) -> None:
    from occasion.models.taxas import TabelaDeTaxas

    X = contexto_para_linha(Contexto(**CONTEXTO_VALIDO))  # type: ignore[arg-type]
    base = pd.concat([X] * 4, ignore_index=True)
    base["dia_semana"] = [1, 5, 6, 7]
    tabela = TabelaDeTaxas(["fim_de_semana", "local"], 0.0).fit(
        base, pd.Series([0, 1, 1, 0]), pd.Series([1.0] * 4)
    )
    artefato = artefato_sintetico
    artefato.modelo_p1 = tabela
    r = prever(Contexto(**CONTEXTO_VALIDO), artefato)  # type: ignore[arg-type]
    fatores = [f for f in r.top_factors if f.parte == "ocasiao" and f.variavel == "dia_semana"]
    assert len(fatores) == 1
    assert fatores[0].valor == "sabado"
