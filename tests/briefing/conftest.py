"""Contexto e resultado fixos usados pelos testes do briefing."""

from collections.abc import Callable
from typing import Any

import pytest

from occasion.inference.contract import Contexto, Resultado

CONTEXTO: dict[str, Any] = {
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


def _montar_resultado(categoria: str, razao: str = "Razão de teste.") -> Resultado:
    return Resultado.model_validate(
        {
            "occasion_score": 87.3,
            "classification": "alta",
            "occasion_probability": 0.0031,
            "score_components": {
                "ocasiao": {"posicao": 92.4, "probabilidade": 0.012},
                "preferencia": {"posicao": 31.0, "probabilidade": 0.26},
            },
            "top_factors": [
                {
                    "parte": "ocasiao",
                    "variavel": "faixa_horario",
                    "valor": "noite",
                    "direcao": "aumenta",
                    "contribuicao": 1.4,
                },
                {
                    "parte": "preferencia",
                    "variavel": "idade",
                    "valor": "28",
                    "direcao": "diminui",
                    "contribuicao": 0.6,
                },
            ],
            "recommendation_category": categoria,
            "recommendation_reason": razao,
            "model_metadata": {
                "modelo": "servido",
                "degrau": "C (logística com curvas e interações)",
                "treinado_em": ["2008-2009", "2017-2018"],
                "receita_commit": "abc1234",
                "metricas_da_receita": {"pr_auc": 0.0123},
                "descricao_metricas": "métricas internas",
                "instante_de_previsao": "planejamento, antes da ocasião",
                "aviso": "guaraná é o sabor registrado na POF, não a marca",
                "versao_contrato": "1.0",
            },
        }
    )


@pytest.fixture
def contexto() -> Contexto:
    return Contexto.model_validate(CONTEXTO)


@pytest.fixture
def resultado() -> Resultado:
    return _montar_resultado("Converter, via combo em app de delivery")


@pytest.fixture
def montar_resultado() -> Callable[[str], Resultado]:
    """Fábrica de resultados fixos, com a categoria de recomendação à escolha."""
    return _montar_resultado
