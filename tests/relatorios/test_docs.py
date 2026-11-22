import pytest

from occasion.config import RAIZ
from occasion.relatorios.docs import (
    aplicar_blocos,
    gerar_blocos,
    numeros_fora_dos_blocos,
    traduzir_veredito,
)


def test_aplicar_blocos_substitui_so_entre_marcadores():
    texto = "antes 0.5\n<!-- BEGIN:a -->\nvelho\n<!-- END:a -->\ndepois\n"
    novo = aplicar_blocos(texto, {"a": "novo"})
    assert novo == "antes 0.5\n<!-- BEGIN:a -->\nnovo\n<!-- END:a -->\ndepois\n"
    assert aplicar_blocos(novo, {"a": "novo"}) == novo


def test_marcador_sem_bloco_levanta_erro():
    with pytest.raises(ValueError):
        aplicar_blocos("<!-- BEGIN:x -->\n<!-- END:x -->", {"a": "1"})


def test_numero_no_texto_presente_no_bloco_passa():
    texto = "A PR-AUC foi 0.1234 e 12.5%.\n<!-- BEGIN:a -->\n| 0.1234 | 12.5% |\n<!-- END:a -->\n"
    assert numeros_fora_dos_blocos(texto) == []


def test_numero_inventado_no_texto_e_apontado():
    texto = "Ganho de 0.9999 e 7.5%.\n<!-- BEGIN:a -->\n| 0.1234 |\n<!-- END:a -->\n"
    assert numeros_fora_dos_blocos(texto) == ["0.9999", "7.5%"]


def test_versoes_entre_crases_e_urls_sao_ignoradas():
    texto = (
        "Usa `Python 3.12` e [guia](https://exemplo.org/v1.2/doc) e https://x.org/2.5 .\n"
        "```\nversao 9.99\n```\n"
        "<!-- BEGIN:a -->\n| 0.1 |\n<!-- END:a -->\n"
    )
    assert numeros_fora_dos_blocos(texto) == []


def test_veredito_desconhecido_levanta_erro():
    with pytest.raises(KeyError):
        traduzir_veredito("talvez")
    assert traduzir_veredito("refutada") == "refuted"


def test_blocos_reais_tem_h1_a_h10_e_h8_refutada():
    blocos = gerar_blocos(RAIZ / "reports")
    assert set(blocos) == {
        "contagens",
        "hipoteses",
        "teste_final",
        "metricas_2008",
        "segmentos_teste",
    }
    linhas = [ln for ln in blocos["hipoteses"].splitlines() if ln.startswith("| H")]
    ids = {ln.split("|")[1].strip() for ln in linhas}
    assert ids == {f"H{i}" for i in range(1, 11)}
    h8 = next(ln for ln in linhas if ln.split("|")[1].strip() == "H8")
    assert "refuted" in h8


def test_rodape_do_teste_final_diz_a_base_de_cada_parte():
    rodape = gerar_blocos(RAIZ / "reports")["teste_final"].splitlines()[-1]
    assert "person × day × cell rows" in rodape
    assert "preference rows on soda occasions" in rodape
    assert "Evaluated on 2017-2018 cells" not in rodape
