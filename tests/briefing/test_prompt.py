"""Entrada e mensagens enviadas ao provedor do briefing."""

import json

from occasion.briefing.prompt import entrada_do_briefing, montar_mensagens
from occasion.inference.contract import Contexto, Resultado


def test_entrada_nao_leva_metadados_alem_do_aviso(contexto: Contexto, resultado: Resultado) -> None:
    entrada = entrada_do_briefing(contexto, resultado)
    assert entrada["contexto"] == contexto.model_dump()
    assert entrada["resultado"]["model_metadata"] == {"aviso": resultado.model_metadata.aviso}
    esperado = set(resultado.model_dump()) | {"model_metadata"}
    assert set(entrada["resultado"]) == esperado
    assert "receita_commit" not in json.dumps(entrada)


def test_mensagens_tem_system_e_user_com_json_da_entrada(
    contexto: Contexto, resultado: Resultado
) -> None:
    mensagens = montar_mensagens(contexto, resultado)
    assert [m["role"] for m in mensagens] == ["system", "user"]
    assert json.loads(mensagens[1]["content"]) == entrada_do_briefing(contexto, resultado)
    sistema = mensagens[0]["content"]
    assert "3 a 5 frases" in sistema
    assert "marca" in sistema
