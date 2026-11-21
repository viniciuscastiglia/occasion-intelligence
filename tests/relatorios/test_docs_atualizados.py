import pytest

from occasion.config import RAIZ
from occasion.relatorios.docs import (
    ARQUIVOS_COM_BLOCOS,
    aplicar_blocos,
    gerar_blocos,
    numeros_fora_dos_blocos,
)

OBRIGATORIOS = {
    "README.md": {"hipoteses", "teste_final", "metricas_2008", "segmentos_teste", "contagens"},
    "docs/DATA_CARD.md": {"contagens", "metricas_2008"},
}


@pytest.mark.parametrize("arquivo", ARQUIVOS_COM_BLOCOS)
def test_docs_atualizados(arquivo: str):
    caminho = RAIZ / arquivo
    assert caminho.exists()
    texto = caminho.read_text(encoding="utf-8")
    for nome in OBRIGATORIOS[arquivo]:
        assert f"<!-- BEGIN:{nome} -->" in texto
    blocos = gerar_blocos(RAIZ / "reports")
    assert aplicar_blocos(texto, blocos) == texto, "rode `make docs`"
    assert numeros_fora_dos_blocos(texto) == []
