"""Download dos microdados da POF/IBGE, conferência de checksums e extração dos zips."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

from occasion.config import DADOS_BRUTOS, EDICOES

_BASE = "https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_{ano}/Microdados/"

URLS: dict[str, dict[str, str]] = {
    "2017-2018": {
        "dados": _BASE.format(ano="2017_2018") + "Dados_20230713.zip",
        "documentacao": _BASE.format(ano="2017_2018") + "Documentacao_20230713.zip",
    },
    "2008-2009": {
        "dados": _BASE.format(ano="2008_2009") + "Dados_20231009.zip",
        "documentacao": _BASE.format(ano="2008_2009") + "Documentacao_20231009.zip",
    },
}

ARQUIVO_CHECKSUMS = "checksums.json"
_BIT_UTF8 = 0x800

Baixador = Callable[[str, Path], None]


def _nome_do_membro(info: zipfile.ZipInfo) -> str:
    """Devolve o nome do membro; nomes sem o bit de UTF-8 vêm em cp850 marcados como cp437."""
    if info.flag_bits & _BIT_UTF8:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp850")
    except UnicodeError:
        return info.filename


def extrair_zip(caminho_zip: Path, destino: Path) -> list[Path]:
    """Extrai o zip em `destino`, mantendo subpastas, e devolve os arquivos criados.

    Recusa (ValueError) membros com caminho absoluto ou que saiam de `destino`, porque o
    conteúdo baixado não é confiável. Nenhum arquivo é gravado antes de validar todos os nomes.
    """
    destino = destino.resolve()
    with zipfile.ZipFile(caminho_zip) as z:
        planejado: list[tuple[zipfile.ZipInfo, Path]] = []
        for info in z.infolist():
            nome = _nome_do_membro(info)
            if Path(nome).is_absolute() or nome.startswith(("/", "\\")):
                raise ValueError(f"caminho absoluto no zip: {nome!r}")
            alvo = (destino / nome).resolve()
            if not alvo.is_relative_to(destino):
                raise ValueError(f"caminho fora do destino no zip: {nome!r}")
            if not info.is_dir():
                planejado.append((info, alvo))
        criados: list[Path] = []
        for info, alvo in planejado:
            alvo.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as origem, open(alvo, "wb") as saida:
                shutil.copyfileobj(origem, saida)
            criados.append(alvo)
    return criados


def _sha256(caminho: Path) -> str:
    soma = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            soma.update(bloco)
    return soma.hexdigest()


def _baixar_url(url: str, destino: Path) -> None:
    """Baixa `url` para `destino`, gravando primeiro em arquivo temporário."""
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    with urllib.request.urlopen(url, timeout=120) as resposta, open(parcial, "wb") as saida:
        shutil.copyfileobj(resposta, saida)
    parcial.replace(destino)


def _ler_checksums(destino: Path) -> dict[str, dict[str, str]]:
    arquivo = destino / ARQUIVO_CHECKSUMS
    if not arquivo.exists():
        return {}
    return json.loads(arquivo.read_text(encoding="utf-8"))


def baixar_edicao(
    edicao: str, destino: Path = DADOS_BRUTOS, baixador: Baixador = _baixar_url
) -> Path:
    """Baixa e extrai os zips de dados e documentação da edição; devolve `destino/<edicao>/`.

    O SHA-256 de cada zip fica em `destino/checksums.json`. Um zip presente com checksum igual
    ao registrado não é baixado de novo; um que diverge (ou sem registro) é baixado de novo.
    """
    if edicao not in URLS:
        raise ValueError(f"edição desconhecida: {edicao!r}; use uma de {sorted(URLS)}")
    pasta = destino / edicao
    pasta.mkdir(parents=True, exist_ok=True)
    registro = _ler_checksums(destino)
    da_edicao = registro.setdefault(edicao, {})
    for url in URLS[edicao].values():
        nome = url.rsplit("/", 1)[1]
        caminho = pasta / nome
        if caminho.exists() and da_edicao.get(nome) == _sha256(caminho):
            continue
        baixador(url, caminho)
        da_edicao[nome] = _sha256(caminho)
        extrair_zip(caminho, pasta)
        (destino / ARQUIVO_CHECKSUMS).write_text(
            json.dumps(registro, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return pasta


if __name__ == "__main__":
    for ed in EDICOES:
        print(f"{ed}: {baixar_edicao(ed)}")
