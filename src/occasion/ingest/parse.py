"""Leitura dos arquivos de largura fixa da POF para o esquema comum."""

import unicodedata
from pathlib import Path
from typing import cast

import pandas as pd

from occasion.ingest.layouts import LAYOUTS, Campos
from occasion.ingest.schema import validar_consumo, validar_morador

EDICAO_2017 = "2017-2018"
EDICAO_2008 = "2008-2009"

_DIAS_DA_SEMANA: dict[str, int] = {
    "segunda-feira": 1,
    "terca-feira": 2,
    "quarta-feira": 3,
    "quinta-feira": 4,
    "sexta-feira": 5,
    "sabado": 6,
    "domingo": 7,
}


def ler_largura_fixa(caminho: Path, campos: Campos) -> pd.DataFrame:
    """Lê só os campos pedidos de um arquivo de largura fixa em latin1, tudo como texto.

    `campos` mapeia o nome da coluna a `(posição inicial em base 1, tamanho)`. Os espaços das
    pontas são removidos e campo vazio vira `""` (nada é interpretado como nulo aqui).
    """
    nomes = list(campos)
    limites = [(inicio - 1, inicio - 1 + tamanho) for inicio, tamanho in campos.values()]
    return pd.read_fwf(
        caminho,
        colspecs=limites,
        names=nomes,
        header=None,
        dtype=str,
        encoding="latin1",
        keep_default_na=False,
        na_values=[],
    )


def _col(df: pd.DataFrame, nome: str) -> pd.Series:
    """Devolve a coluna como Series (os nomes das colunas são únicos)."""
    return cast(pd.Series, df[nome])


def _sem_acento_minusculo(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in base if not unicodedata.combining(c))


def _mapear(serie: pd.Series, mapa: dict[str, str] | dict[str, int], campo: str) -> pd.Series:
    """Aplica `mapa` à série; valor fora do mapa levanta erro em vez de virar nulo."""
    mapeada = serie.map(mapa)
    invalidos = sorted(set(serie[mapeada.isna()]))
    if invalidos:
        raise ValueError(f"valores desconhecidos em {campo}: {invalidos}")
    return mapeada


def _inteiro(serie: pd.Series, campo: str) -> pd.Series:
    if (serie == "").any():
        raise ValueError(f"{campo} vazio em {(serie == '').sum()} linhas")
    return serie.astype("int64")


def _decimal(serie: pd.Series, campo: str, decimais: int = 0) -> pd.Series:
    """Texto para float. Com ponto, converte direto; sem ponto, divide por 10**decimais."""
    if (serie == "").any():
        raise ValueError(f"{campo} vazio em {(serie == '').sum()} linhas")
    valor = serie.astype("float64")
    return valor.where(serie.str.contains(".", regex=False), valor / 10**decimais)


def _chaves(bruto: pd.DataFrame, edicao: str) -> pd.DataFrame:
    """Colunas de identificação comuns a consumo e morador."""
    chaves = pd.DataFrame(
        {
            "edicao": edicao,
            "uf": _inteiro(_col(bruto, "UF"), "UF"),
            "cod_upa": _col(bruto, "COD_UPA"),
            "num_dom": _inteiro(_col(bruto, "NUM_DOM"), "NUM_DOM"),
            "num_uc": _inteiro(_col(bruto, "NUM_UC"), "NUM_UC"),
            "cod_informante": _inteiro(_col(bruto, "COD_INFORMANTE"), "COD_INFORMANTE"),
        }
    )
    chaves["estrato"] = chaves["uf"].astype(str) + "-" + _col(bruto, "ESTRATO_POF")
    chaves["id_pessoa"] = (
        edicao
        + "|"
        + chaves["cod_upa"]
        + "|"
        + chaves["num_dom"].astype(str)
        + "|"
        + chaves["num_uc"].astype(str)
        + "|"
        + chaves["cod_informante"].astype(str)
    )
    return chaves


def _hora(serie: pd.Series) -> pd.Series:
    """`20H` vira 20; vazio vira nulo."""
    numero = cast(pd.Series, pd.to_numeric(serie.str.removesuffix("H").replace("", pd.NA)))
    return numero.astype("Int64")


def _local(serie: pd.Series) -> pd.Series:
    """V9018: 1 em casa, 2 a 7 fora de casa, 9 ou vazio não determinado."""
    mapa = {"1": "em_casa", "9": "indeterminado", "": "indeterminado"}
    mapa.update({str(codigo): "fora_de_casa" for codigo in range(2, 8)})
    return _mapear(serie, mapa, "V9018")


def parse_consumo_2017(caminho: Path) -> pd.DataFrame:
    """Lê `CONSUMO_ALIMENTAR.txt` (2017-2018) no esquema comum de consumo."""
    bruto = ler_largura_fixa(caminho, LAYOUTS[EDICAO_2017]["consumo"])
    consumo = _chaves(bruto, EDICAO_2017)
    consumo["quadro"] = _inteiro(_col(bruto, "QUADRO"), "QUADRO")
    consumo["id_dia"] = consumo["id_pessoa"] + "|" + consumo["quadro"].astype(str)
    consumo["cod_alimento"] = _inteiro(_col(bruto, "V9001"), "V9001")
    consumo["hora"] = _hora(_col(bruto, "V9015"))
    consumo["local"] = _local(_col(bruto, "V9018"))
    dias = _col(bruto, "DIA_SEMANA").map(_sem_acento_minusculo)
    consumo["dia_semana"] = _mapear(dias, _DIAS_DA_SEMANA, "DIA_SEMANA").astype("int64")
    consumo["peso_final"] = _decimal(_col(bruto, "PESO_FINAL"), "PESO_FINAL")
    consumo["renda_total_uc"] = _decimal(_col(bruto, "RENDA_TOTAL"), "RENDA_TOTAL")
    return consumo


def parse_morador_2017(caminho: Path) -> pd.DataFrame:
    """Lê `MORADOR.txt` (2017-2018) no esquema comum de morador.

    `n_moradores_uc` é a contagem de linhas do arquivo por unidade de consumo.
    """
    bruto = ler_largura_fixa(caminho, LAYOUTS[EDICAO_2017]["morador"])
    morador = _chaves(bruto, EDICAO_2017)
    situacao = {"1": "urbano", "2": "rural"}
    morador["situacao"] = _mapear(_col(bruto, "TIPO_SITUACAO_REG"), situacao, "TIPO_SITUACAO_REG")
    morador["idade"] = _inteiro(_col(bruto, "V0403"), "V0403")
    sexo = {"1": "masculino", "2": "feminino"}
    morador["sexo"] = _mapear(_col(bruto, "V0404"), sexo, "V0404")
    chave_uc = ["cod_upa", "num_dom", "num_uc"]
    morador["n_moradores_uc"] = morador.groupby(chave_uc)["cod_informante"].transform("size")
    return morador


def _gravar_interim(
    edicao: str, consumo: pd.DataFrame, morador: pd.DataFrame, pasta_saida: Path
) -> tuple[Path, Path]:
    """Valida os dois frames no esquema comum e só então grava os parquets da edição.

    Se qualquer um falhar na validação, nenhum arquivo é gravado.
    """
    consumo = validar_consumo(consumo)
    morador = validar_morador(morador)
    pasta_saida.mkdir(parents=True, exist_ok=True)
    destino_consumo = pasta_saida / f"consumo_{edicao}.parquet"
    destino_morador = pasta_saida / f"morador_{edicao}.parquet"
    consumo.to_parquet(destino_consumo, index=False)
    morador.to_parquet(destino_morador, index=False)
    return destino_consumo, destino_morador


def gravar_interim_2017(pasta_bruta: Path, pasta_saida: Path) -> tuple[Path, Path]:
    """Lê os brutos de 2017-2018, valida no esquema comum e grava os dois parquets."""
    consumo = parse_consumo_2017(pasta_bruta / "CONSUMO_ALIMENTAR.txt")
    morador = parse_morador_2017(pasta_bruta / "MORADOR.txt")
    return _gravar_interim(EDICAO_2017, consumo, morador, pasta_saida)


_DIAS_NUMERICOS: dict[str, int] = {str(n): n for n in range(1, 8)}


def _com_chaves_2008(bruto: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta `UF`, `COD_UPA` (UF + NUM_SEQ + NUM_DV, com zeros) e normaliza o estrato.

    Se o registro já traz `COD_UPA`, ele precisa ser igual ao montado (erro claro se divergir).
    """
    bruto = bruto.copy()
    montado = (
        _col(bruto, "COD_UF").str.zfill(2)
        + _col(bruto, "NUM_SEQ").str.zfill(3)
        + _col(bruto, "NUM_DV").str.zfill(1)
    )
    if "COD_UPA" in bruto:
        divergentes = _col(bruto, "COD_UPA").str.zfill(6) != montado
        if divergentes.any():
            raise ValueError(f"COD_UPA diverge de UF+NUM_SEQ+NUM_DV em {divergentes.sum()} linhas")
    bruto["UF"] = bruto["COD_UF"]
    bruto["COD_UPA"] = montado
    bruto["ESTRATO_POF"] = _col(bruto, "ESTRATO_POF").str.zfill(2)
    return bruto


def _local_2008(serie: pd.Series) -> pd.Series:
    """V9018 (fonte do alimento): 1 dentro, 2 fora do domicílio; outro valor ou vazio é indeterminado."""
    mapa = {"1": "em_casa", "2": "fora_de_casa"}
    return serie.map(mapa).fillna("indeterminado")


def parse_consumo_2008(caminho: Path) -> pd.DataFrame:
    """Lê `T_CONSUMO_S.txt` (2008-2009) no esquema comum de consumo."""
    bruto = _com_chaves_2008(ler_largura_fixa(caminho, LAYOUTS[EDICAO_2008]["consumo"]))
    consumo = _chaves(bruto, EDICAO_2008)
    consumo["quadro"] = _inteiro(_col(bruto, "NUM_QUADRO"), "NUM_QUADRO")
    consumo["id_dia"] = consumo["id_pessoa"] + "|" + consumo["quadro"].astype(str)
    consumo["cod_alimento"] = _inteiro(_col(bruto, "V9001"), "V9001")
    consumo["hora"] = _hora(_col(bruto, "V9015"))
    consumo["local"] = _local_2008(_col(bruto, "V9018"))
    consumo["dia_semana"] = _mapear(
        _col(bruto, "DIA_SEMANA"), _DIAS_NUMERICOS, "DIA_SEMANA"
    ).astype("int64")
    consumo["peso_final"] = _decimal(_col(bruto, "PESO_FINAL"), "PESO_FINAL", decimais=8)
    consumo["renda_total_uc"] = _decimal(_col(bruto, "RENDA_TOTAL"), "RENDA_TOTAL", decimais=2)
    return consumo


def parse_morador_2008(caminho: Path) -> pd.DataFrame:
    """Lê `T_MORADOR_S.txt` (2008-2009) no esquema comum de morador."""
    bruto = _com_chaves_2008(ler_largura_fixa(caminho, LAYOUTS[EDICAO_2008]["morador"]))
    morador = _chaves(bruto, EDICAO_2008)
    situacao = {"1": "urbano", "2": "rural"}
    morador["situacao"] = _mapear(_col(bruto, "TIPO_SITUACAO_REG"), situacao, "TIPO_SITUACAO_REG")
    morador["idade"] = _inteiro(_col(bruto, "IDADE_ANOS"), "IDADE_ANOS")
    sexo = {"1": "masculino", "2": "feminino"}
    morador["sexo"] = _mapear(_col(bruto, "V0405"), sexo, "V0405")
    chave_uc = ["cod_upa", "num_dom", "num_uc"]
    morador["n_moradores_uc"] = morador.groupby(chave_uc)["cod_informante"].transform("size")
    return morador


def gravar_interim_2008(pasta_bruta: Path, pasta_saida: Path) -> tuple[Path, Path]:
    """Lê os brutos de 2008-2009, valida no esquema comum e grava os dois parquets."""
    consumo = parse_consumo_2008(pasta_bruta / "T_CONSUMO_S.txt")
    morador = parse_morador_2008(pasta_bruta / "T_MORADOR_S.txt")
    return _gravar_interim(EDICAO_2008, consumo, morador, pasta_saida)
