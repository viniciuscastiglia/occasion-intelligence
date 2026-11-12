"""Testes do download e da extração dos zips do IBGE (sem acesso à rede)."""

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from occasion.ingest.download import URLS, baixar_edicao, extrair_zip


def _criar_zip(caminho: Path, arquivos: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(caminho, "w") as z:
        for nome, conteudo in arquivos.items():
            z.writestr(nome, conteudo)
    return caminho.read_bytes()


def test_extrair_zip_preserva_nomes_com_acento(tmp_path: Path) -> None:
    zip_path = tmp_path / "doc.zip"
    _criar_zip(zip_path, {"Dicionários de váriaveis.xls": b"x"})
    arquivos = extrair_zip(zip_path, tmp_path / "saida")
    assert [a.name for a in arquivos] == ["Dicionários de váriaveis.xls"]


def test_extrair_zip_decodifica_nome_cp850_com_bit_utf8_desligado(tmp_path: Path) -> None:
    # O zipfile liga o bit de UTF-8 para nomes não ASCII; para simular o zip do IBGE, grava-se
    # um nome ASCII de mesmo tamanho e trocam-se os bytes por cp850 (bit desligado).
    zip_path = tmp_path / "antigo.zip"
    _criar_zip(zip_path, {"Classificacao.xls": b"x"})
    bruto = zip_path.read_bytes().replace(b"Classificacao.xls", "Classificação.xls".encode("cp850"))
    zip_path.write_bytes(bruto)
    with zipfile.ZipFile(zip_path) as z:
        assert z.infolist()[0].flag_bits & 0x800 == 0
        assert z.namelist() == ["Classificação.xls".encode("cp850").decode("cp437")]
    arquivos = extrair_zip(zip_path, tmp_path / "saida")
    assert [a.name for a in arquivos] == ["Classificação.xls"]
    assert arquivos[0].read_bytes() == b"x"


def test_extrair_zip_mantem_nome_com_bit_utf8_ligado(tmp_path: Path) -> None:
    zip_path = tmp_path / "utf8.zip"
    _criar_zip(zip_path, {"Classificação.xls": b"x"})
    with zipfile.ZipFile(zip_path) as z:
        assert z.infolist()[0].flag_bits & 0x800
    arquivos = extrair_zip(zip_path, tmp_path / "saida")
    assert [a.name for a in arquivos] == ["Classificação.xls"]


def test_extrair_zip_recusa_caminho_fora_do_destino(tmp_path: Path) -> None:
    zip_path = tmp_path / "mal.zip"
    _criar_zip(zip_path, {"../fora.txt": b"x"})
    with pytest.raises(ValueError):
        extrair_zip(zip_path, tmp_path / "saida")
    assert not (tmp_path / "fora.txt").exists()


def test_extrair_zip_recusa_caminho_absoluto(tmp_path: Path) -> None:
    zip_path = tmp_path / "abs.zip"
    alvo = tmp_path / "absoluto.txt"
    _criar_zip(zip_path, {str(alvo): b"x"})
    with pytest.raises(ValueError):
        extrair_zip(zip_path, tmp_path / "saida")
    assert not alvo.exists()


def test_extrair_zip_mantem_subpastas(tmp_path: Path) -> None:
    zip_path = tmp_path / "sub.zip"
    _criar_zip(zip_path, {"Dados/A.txt": b"a", "Dados/": b""})
    arquivos = extrair_zip(zip_path, tmp_path / "saida")
    assert arquivos == [tmp_path / "saida" / "Dados" / "A.txt"]


def test_urls_cobrem_as_duas_edicoes() -> None:
    assert set(URLS) == {"2008-2009", "2017-2018"}
    for urls in URLS.values():
        assert set(urls) == {"dados", "documentacao"}
        assert all(
            u.startswith("https://ftp.ibge.gov.br/") and u.endswith(".zip") for u in urls.values()
        )


class _Baixador:
    """Substitui a rede: grava um zip pequeno e conta as chamadas."""

    def __init__(self) -> None:
        self.chamadas: list[str] = []

    def __call__(self, url: str, destino: Path) -> None:
        self.chamadas.append(url)
        nome = "CONSUMO.txt" if "Dados" in url else "Dicionário.xls"
        _criar_zip(destino, {nome: b"conteudo"})


def test_baixar_edicao_baixa_extrai_e_registra_checksums(tmp_path: Path) -> None:
    baixador = _Baixador()
    pasta = baixar_edicao("2017-2018", tmp_path, baixador=baixador)
    assert pasta == tmp_path / "2017-2018"
    assert (pasta / "CONSUMO.txt").exists()
    assert (pasta / "Dicionário.xls").exists()
    assert len(baixador.chamadas) == 2
    registro = json.loads((tmp_path / "checksums.json").read_text())
    nomes = {u.rsplit("/", 1)[1] for u in URLS["2017-2018"].values()}
    assert set(registro["2017-2018"]) == nomes
    for nome, soma in registro["2017-2018"].items():
        assert soma == hashlib.sha256((pasta / nome).read_bytes()).hexdigest()


def test_baixar_edicao_pula_zip_com_checksum_registrado(tmp_path: Path) -> None:
    baixar_edicao("2017-2018", tmp_path, baixador=_Baixador())
    segundo = _Baixador()
    baixar_edicao("2017-2018", tmp_path, baixador=segundo)
    assert segundo.chamadas == []


def test_baixar_edicao_baixa_de_novo_zip_com_checksum_divergente(tmp_path: Path) -> None:
    baixar_edicao("2017-2018", tmp_path, baixador=_Baixador())
    zip_dados = next(tmp_path.glob("2017-2018/Dados_*.zip"))
    zip_dados.write_bytes(b"corrompido")
    segundo = _Baixador()
    baixar_edicao("2017-2018", tmp_path, baixador=segundo)
    assert len(segundo.chamadas) == 1
    assert "Dados_" in segundo.chamadas[0]
    assert zipfile.is_zipfile(zip_dados)


def test_baixar_edicao_preserva_checksums_de_outra_edicao(tmp_path: Path) -> None:
    baixar_edicao("2017-2018", tmp_path, baixador=_Baixador())
    baixar_edicao("2008-2009", tmp_path, baixador=_Baixador())
    registro = json.loads((tmp_path / "checksums.json").read_text())
    assert set(registro) == {"2008-2009", "2017-2018"}


def test_baixar_edicao_recusa_edicao_desconhecida(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        baixar_edicao("1999-2000", tmp_path, baixador=_Baixador())
