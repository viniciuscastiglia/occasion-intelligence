"""Testes da tela Streamlit com a ferramenta de teste de aplicações do próprio Streamlit."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from tests.inference.test_predict import _artefato_unico

from occasion.inference.contract import Fator
from occasion.inference.predict import ArtefatoModelo, salvar_artefato

CAMINHO_APP = str(Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py")


@pytest.fixture
def artefato_em_disco(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ArtefatoModelo:
    artefato = _artefato_unico()
    salvar_artefato(artefato, tmp_path / "avaliado")
    monkeypatch.setenv("OCCASION_MODELOS", str(tmp_path))
    return artefato


@pytest.fixture
def sem_artefato(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OCCASION_MODELOS", str(tmp_path / "vazio"))


def test_tela_mostra_score_e_recomendacao(artefato_em_disco: ArtefatoModelo) -> None:
    at = AppTest.from_file(CAMINHO_APP).run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    titulos = " ".join(s.value for s in at.subheader)
    assert "Recomendação" in titulos and "Sobre o modelo" in titulos
    assert "Occasion Score" in titulos and "Fatores" in titulos
    assert len(at.metric) == 3 and len(at.expander) == 1


def test_tela_sem_artefato_mostra_instrucao_e_nao_quebra(sem_artefato: None) -> None:
    at = AppTest.from_file(CAMINHO_APP).run()
    assert not at.exception
    assert any("make baseline" in e.value for e in at.error)


def _fatores_exemplo() -> list[Fator]:
    itens = [
        ("ocasiao", "faixa_horario", "noite"),
        ("ocasiao", "dia_semana", "sabado"),
        ("ocasiao", "local", "em_casa"),
        ("preferencia", "regiao", "centro_oeste"),
        ("preferencia", "local", "fora_de_casa"),
    ]
    return [
        Fator(
            parte=parte,  # type: ignore[arg-type]
            variavel=variavel,
            valor=valor,
            direcao="aumenta" if i % 2 == 0 else "diminui",
            contribuicao=0.5 - i * 0.2,
        )
        for i, (parte, variavel, valor) in enumerate(itens)
    ]


def test_grafico_tem_um_rotulo_legivel_por_fator() -> None:
    from app.streamlit_app import ALTURA_POR_FATOR, grafico_fatores, tabela_fatores

    fatores = _fatores_exemplo()
    tabela = tabela_fatores(fatores)
    assert len(tabela) == len(fatores)
    assert tabela["fator"].is_unique
    assert not any("_" in rotulo for rotulo in tabela["fator"])
    assert "centro-oeste" in " ".join(tabela["fator"]).lower()
    assert grafico_fatores(fatores).to_dict()["height"] == {"step": ALTURA_POR_FATOR}


def test_tela_mostra_probabilidade_com_virgula_e_limites_completos(
    artefato_em_disco: ArtefatoModelo,
) -> None:
    at = AppTest.from_file(CAMINHO_APP).run()
    at.sidebar.button[0].click().run()
    probabilidade = next(m for m in at.metric if m.label == "Probabilidade conjunta")
    assert "," in probabilidade.value and "." not in probabilidade.value
    texto = " ".join(m.value for m in at.markdown)
    assert "expansão do delivery" in texto
    assert "registro alimentar" in texto and "recordatório de 24 horas" in texto
    assert "fonte do alimento" in texto
    justificativa = next(m.value for m in at.markdown if m.value.startswith("**Categoria"))
    assert justificativa
    assert not at.warning


def test_tela_avisa_quando_o_contexto_mudou_depois_do_calculo(
    artefato_em_disco: ArtefatoModelo,
) -> None:
    at = AppTest.from_file(CAMINHO_APP).run()
    at.sidebar.button[0].click().run()
    assert not at.warning
    at.sidebar.number_input[0].set_value(45).run()
    assert any("Calcular" in w.value for w in at.warning)
    at.sidebar.button[0].click().run()
    assert not at.warning


def test_linha_de_metricas_tem_rotulo_e_virgula() -> None:
    from app.streamlit_app import _linha_metricas

    linha = _linha_metricas({"pr_auc": 0.0018773, "pr_auc_degrau_a": 0.0019113})
    assert "Métricas do teste temporal da receita (2017-2018)" in linha
    assert "PR-AUC 0,001877" in linha and "." not in linha.split(":**")[1]
    assert _linha_metricas(None) == ""


def test_limitacao_das_edicoes_depende_do_modelo() -> None:
    from app.streamlit_app import limitacao_das_edicoes

    assert "(treino) e 2017-2018 (teste)" in limitacao_das_edicoes("avaliado")
    servido = limitacao_das_edicoes("servido")
    assert "reajustado com as duas edições" in servido
    assert "não tem teste temporal próprio" in servido
