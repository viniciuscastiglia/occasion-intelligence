"""Testes do contrato de entrada e saída da aplicação."""

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from occasion.inference.contract import CODIGO_UF, Contexto, Resultado

EXEMPLOS = Path(__file__).resolve().parents[1] / "exemplos"

CONTEXTO_VALIDO: dict[str, Any] = {
    "uf": "SP",
    "situacao": "urbano",
    "dia_semana": "sabado",
    "faixa_horario": "noite",
    "local": "fora_de_casa",
    "idade": 28,
    "sexo": "masculino",
    "faixa_renda": "de_2_a_5_sm",
    "numero_moradores": 3,
}


def test_idade_abaixo_de_18_e_recusada():
    with pytest.raises(ValidationError):
        Contexto(**{**CONTEXTO_VALIDO, "idade": 17})


def test_resultado_exemplo_respeita_o_contrato():
    Resultado.model_validate_json((EXEMPLOS / "resultado_exemplo.json").read_text(encoding="utf-8"))


def test_saida_tem_os_seis_campos_do_tapi():
    campos = set(Resultado.model_fields)
    assert {
        "occasion_score",
        "classification",
        "top_factors",
        "recommendation_category",
        "recommendation_reason",
        "model_metadata",
    } <= campos


def test_codigo_uf_cobre_as_27():
    assert len(CODIGO_UF) == 27 and CODIGO_UF["SP"] == 35 and CODIGO_UF["DF"] == 53


def test_exemplo_e_coerente_com_os_cortes_e_com_as_duas_partes():
    exemplo = json.loads((EXEMPLOS / "resultado_exemplo.json").read_text(encoding="utf-8"))
    score = exemplo["occasion_score"]
    esperada = "alta" if score >= 80 else "média" if score >= 50 else "baixa"
    assert exemplo["classification"] == esperada

    partes = exemplo["score_components"]
    produto = partes["ocasiao"]["probabilidade"] * partes["preferencia"]["probabilidade"]
    assert exemplo["occasion_probability"] == pytest.approx(produto)
