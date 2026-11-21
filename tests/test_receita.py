"""Receita congelada: escolhas do Plano 2 em JSON, com ida e volta sem perda."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from occasion.config import FAIXAS_HORARIO
from occasion.decision.engine import CORTE_OCASIAO, CORTE_PREFERENCIA
from occasion.experimentos.especificacao import INTERACOES, variaveis_da_parte
from occasion.receita import Receita, ReceitaParte, carregar_receita, salvar_receita
from occasion.score.score import CORTE_ALTA, CORTE_MEDIA


def _campos(**mudancas: Any) -> dict[str, Any]:
    campos: dict[str, Any] = {
        "faixas_horario": {nome: list(faixa) for nome, faixa in FAIXAS_HORARIO.items()},
        "variaveis": {p: variaveis_da_parte(p) for p in ("ocasiao", "preferencia", "direto")},
        "interacoes": {
            "ocasiao": INTERACOES["ocasiao"]["fim_de_semana"],
            "preferencia": [],
            "direto": [],
        },
        "ocasiao": ReceitaParte(
            degrau="C",
            parametros={"C": 0.1, "interacoes": "fim_de_semana"},
            usar_peso=True,
            calibrar=False,
        ),
        "preferencia": ReceitaParte(
            degrau="D",
            parametros={"learning_rate": 0.05, "n_estimators": 300, "objective": "binary"},
            usar_peso=False,
            calibrar=True,
        ),
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
        "experimentos": ["reports/experimentos/h5_peso.json", "reports/experimentos/cortes.json"],
    }
    campos.update(mudancas)
    return campos


def test_ida_e_volta_por_json_preserva_tudo(tmp_path: Path) -> None:
    receita = Receita(**_campos())
    caminho = salvar_receita(receita, tmp_path / "receita.json")
    assert caminho == tmp_path / "receita.json"
    lida = carregar_receita(caminho)
    assert lida == receita
    # os pares de interação voltam como tuplas, não como listas
    assert lida.interacoes["ocasiao"][0] == ("fim_de_semana", "faixa_horario")
    assert lida.ocasiao.parametros == {"C": 0.1, "interacoes": "fim_de_semana"}
    assert lida.faixas_horario["almoco"] == [11, 12, 13, 14]
    # o arquivo é JSON legível, com as escolhas e sem coeficientes ajustados
    bruto = json.loads(caminho.read_text(encoding="utf-8"))
    assert bruto["score"] == "produto"
    assert bruto["preferencia"]["degrau"] == "D"


def test_ida_e_volta_com_o_score_direto(tmp_path: Path) -> None:
    direto = ReceitaParte(degrau="B", parametros={"C": 1.0}, usar_peso=True, calibrar=False)
    referencia = {**_campos()["referencia_a"], "direto": {"dimensoes": ["faixa_horario"]}}
    receita = Receita(**_campos(score="direto", direto=direto, referencia_a=referencia))
    assert carregar_receita(salvar_receita(receita, tmp_path / "r.json")) == receita


def test_score_direto_sem_o_modelo_direto_e_recusado() -> None:
    with pytest.raises(ValidationError, match="direto"):
        Receita(**_campos(score="direto", direto=None))


def test_score_produto_com_modelo_direto_e_recusado() -> None:
    direto = ReceitaParte(degrau="B", parametros={"C": 1.0}, usar_peso=True, calibrar=False)
    with pytest.raises(ValidationError, match="direto"):
        Receita(**_campos(direto=direto))


def test_score_fora_de_produto_ou_direto_e_recusado() -> None:
    with pytest.raises(ValidationError):
        Receita(**_campos(score="media"))


def test_degrau_fora_de_a_a_d_e_recusado() -> None:
    with pytest.raises(ValidationError):
        ReceitaParte(degrau="E", parametros={}, usar_peso=True, calibrar=False)  # type: ignore[arg-type]


def test_cortes_incompletos_sao_recusados() -> None:
    with pytest.raises(ValidationError, match="cortes"):
        Receita(**_campos(cortes={"alta": 80.0, "media": 50.0}))


def test_campo_desconhecido_e_recusado() -> None:
    with pytest.raises(ValidationError):
        Receita(**_campos(coeficientes_da_sigmoide=[1.0, 0.0]))
