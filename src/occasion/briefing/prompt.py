"""Entrada estruturada e mensagens do briefing. O LLM só recebe o contexto e o resultado."""

import json
from typing import Any

from occasion.inference.contract import Contexto, Resultado

REGRAS_DO_SISTEMA = """\
Você escreve o briefing de uma recomendação de ativação para uma pessoa de marketing.
O JSON recebido traz o contexto informado e o resultado de um modelo que mede a associação
entre pizza e refrigerante de guaraná nos dados da POF/IBGE.

Regras:
- Escreva em português, em 3 a 5 frases.
- Use só os fatores, os números e a categoria de recomendação que estão no JSON, com a direção
  recebida (aumenta ou diminui). Não invente fatores, números nem categorias.
- Não afirme nem sugira efeito sobre vendas. O resultado descreve associação observada,
  não efeito de uma ação.
- Guaraná é o sabor de refrigerante registrado na POF, não uma marca.
- Não cite marcas.
- Responda só com o texto do briefing, sem títulos, listas ou JSON.
"""


def entrada_do_briefing(contexto: Contexto, resultado: Resultado) -> dict[str, Any]:
    """Contexto e resultado como dicionário; dos metadados do modelo só entra o aviso."""
    campos = resultado.model_dump(exclude={"model_metadata"})
    campos["model_metadata"] = {"aviso": resultado.model_metadata.aviso}
    return {"contexto": contexto.model_dump(), "resultado": campos}


def montar_mensagens(contexto: Contexto, resultado: Resultado) -> list[dict[str, str]]:
    """Mensagem `system` com as regras e mensagem `user` com o JSON da entrada."""
    entrada = entrada_do_briefing(contexto, resultado)
    return [
        {"role": "system", "content": REGRAS_DO_SISTEMA},
        {"role": "user", "content": json.dumps(entrada, ensure_ascii=False)},
    ]
