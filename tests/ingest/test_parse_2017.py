"""Testes do parser da POF 2017-2018: unitários com linhas montadas e conferência nos dados reais."""

from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from occasion.config import DADOS_INTERIM
from occasion.ingest.parse import ler_largura_fixa, parse_consumo_2017, parse_morador_2017
from occasion.ingest.schema import validar_consumo, validar_morador

# Posições (base 1, tamanho) escritas aqui de propósito, independentes de layouts.py.
_CONSUMO = {
    "UF": (1, 2),
    "ESTRATO_POF": (3, 4),
    "TIPO_SITUACAO_REG": (7, 1),
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "COD_INFORMANTE": (20, 2),
    "QUADRO": (22, 2),
    "V9001": (32, 7),
    "V9015": (39, 3),
    "V9017": (44, 1),
    "V9018": (45, 1),
    "PESO_FINAL": (504, 15),
    "RENDA_TOTAL": (519, 10),
    "DIA_SEMANA": (529, 15),
}
_MORADOR = {
    "UF": (1, 2),
    "ESTRATO_POF": (3, 4),
    "TIPO_SITUACAO_REG": (7, 1),
    "COD_UPA": (8, 9),
    "NUM_DOM": (17, 2),
    "NUM_UC": (19, 1),
    "COD_INFORMANTE": (20, 2),
    "V0403": (33, 3),
    "V0404": (36, 1),
}


def _linha(posicoes: dict[str, tuple[int, int]], largura: int, **valores: str) -> str:
    """Monta uma linha de largura fixa; cada valor é alinhado à esquerda no seu campo."""
    texto = [" "] * largura
    for nome, valor in valores.items():
        inicio, tamanho = posicoes[nome]
        assert len(valor) <= tamanho, nome
        texto[inicio - 1 : inicio - 1 + tamanho] = valor.ljust(tamanho)
    return "".join(texto)


def _linha_consumo(**sobrescritas: str) -> str:
    base = {
        "UF": "35",
        "ESTRATO_POF": "3501",
        "TIPO_SITUACAO_REG": "1",
        "COD_UPA": "350000001",
        "NUM_DOM": " 3",
        "NUM_UC": "1",
        "COD_INFORMANTE": " 2",
        "QUADRO": "72",
        "V9001": "8500903",
        "V9015": "20H",
        "V9017": "4",
        "V9018": "1",
        "PESO_FINAL": "  2390.48652303",
        "RENDA_TOTAL": "   3855.34",
        "DIA_SEMANA": "Sábado",
    }
    return _linha(_CONSUMO, 543, **{**base, **sobrescritas})


def _linha_morador(**sobrescritas: str) -> str:
    base = {
        "UF": "35",
        "ESTRATO_POF": "3501",
        "TIPO_SITUACAO_REG": "1",
        "COD_UPA": "350000001",
        "NUM_DOM": " 3",
        "NUM_UC": "1",
        "COD_INFORMANTE": " 2",
        "V0403": " 64",
        "V0404": "1",
    }
    return _linha(_MORADOR, 120, **{**base, **sobrescritas})


def _numero(valor: object) -> float:
    """Converte o escalar devolvido por uma agregação do pandas (o pyright o vê como Series)."""
    return float(cast(float, valor))


def _escrever(tmp_path: Path, linhas: list[str], nome: str = "arquivo.txt") -> Path:
    caminho = tmp_path / nome
    caminho.write_bytes(("\n".join(linhas) + "\n").encode("latin1"))
    return caminho


def test_ler_largura_fixa_devolve_texto_sem_espacos(tmp_path):
    arquivo = _escrever(tmp_path, [_linha_consumo(), _linha_consumo(V9015="")])
    df = ler_largura_fixa(arquivo, {"quadro": (22, 2), "hora": (39, 3), "dia": (529, 15)})
    assert list(df.columns) == ["quadro", "hora", "dia"]
    assert df.iloc[0].tolist() == ["72", "20H", "Sábado"]
    assert df.iloc[1]["hora"] == ""
    assert all(df[c].map(type).eq(str).all() for c in df.columns)


def test_parse_consumo_2017_converte_campos(tmp_path):
    arquivo = _escrever(tmp_path, [_linha_consumo(UF="35")])
    df = parse_consumo_2017(arquivo)
    linha = df.iloc[0]
    assert (linha.uf, linha.quadro, linha.cod_alimento) == (35, 72, 8500903)
    assert (linha.hora, linha.local, linha.dia_semana) == (20, "em_casa", 6)
    assert linha.edicao == "2017-2018"
    assert linha.estrato == "35-3501"
    assert (linha.cod_upa, linha.num_dom, linha.num_uc, linha.cod_informante) == (
        "350000001",
        3,
        1,
        2,
    )
    assert linha.id_pessoa == "2017-2018|350000001|3|1|2"
    assert linha.id_dia == "2017-2018|350000001|3|1|2|72"
    assert linha.peso_final == pytest.approx(2390.48652303)
    assert linha.renda_total_uc == pytest.approx(3855.34)
    validar_consumo(df)


@pytest.mark.parametrize("codigo", ["9", " "])
def test_local_indeterminado(tmp_path, codigo):
    df = parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(V9018=codigo)]))
    assert df.iloc[0]["local"] == "indeterminado"


@pytest.mark.parametrize("codigo", ["2", "3", "4", "5", "6", "7"])
def test_local_fora_de_casa(tmp_path, codigo):
    df = parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(V9018=codigo)]))
    assert df.iloc[0]["local"] == "fora_de_casa"


def test_local_desconhecido_falha(tmp_path):
    with pytest.raises(ValueError, match="V9018"):
        parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(V9018="8")]))


def test_hora_vazia_vira_nulo(tmp_path):
    df = parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(V9015=""), _linha_consumo()]))
    assert pd.isna(df.iloc[0]["hora"])
    assert df.iloc[1]["hora"] == 20
    assert str(df["hora"].dtype) == "Int64"
    validar_consumo(df)


def test_hora_com_h_e_um_digito(tmp_path):
    df = parse_consumo_2017(
        _escrever(tmp_path, [_linha_consumo(V9015="8H"), _linha_consumo(V9015="0H")])
    )
    assert df["hora"].tolist() == [8, 0]


@pytest.mark.parametrize(
    ("texto", "numero"),
    [
        ("Segunda-Feira", 1),
        ("Terça-Feira", 2),
        ("Quarta-Feira", 3),
        ("Quinta-Feira", 4),
        ("Sexta-Feira", 5),
        ("Sábado", 6),
        ("Domingo", 7),
        ("SABADO", 6),
        ("terca-feira", 2),
    ],
)
def test_dia_da_semana_sem_acento_e_sem_caixa(tmp_path, texto, numero):
    df = parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(DIA_SEMANA=texto)]))
    assert df.iloc[0]["dia_semana"] == numero


def test_dia_da_semana_desconhecido_falha(tmp_path):
    with pytest.raises(ValueError, match="DIA_SEMANA"):
        parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(DIA_SEMANA="Feriado")]))


def test_renda_vazia_nao_passa_em_silencio(tmp_path):
    with pytest.raises(ValueError, match="RENDA_TOTAL"):
        parse_consumo_2017(_escrever(tmp_path, [_linha_consumo(RENDA_TOTAL="")]))


def test_parse_morador_2017_converte_campos(tmp_path):
    arquivo = _escrever(
        tmp_path,
        [
            _linha_morador(COD_INFORMANTE=" 1", V0403=" 64", V0404="1"),
            _linha_morador(COD_INFORMANTE=" 2", V0403="  7", V0404="2", TIPO_SITUACAO_REG="2"),
            _linha_morador(NUM_UC="2", COD_INFORMANTE=" 1", V0403="101", V0404="2"),
        ],
    )
    df = parse_morador_2017(arquivo)
    assert df["idade"].tolist() == [64, 7, 101]
    assert df["sexo"].tolist() == ["masculino", "feminino", "feminino"]
    assert df["situacao"].tolist() == ["urbano", "rural", "urbano"]
    assert df["n_moradores_uc"].tolist() == [2, 2, 1]
    assert df.iloc[0]["id_pessoa"] == "2017-2018|350000001|3|1|1"
    assert df.iloc[0]["estrato"] == "35-3501"
    validar_morador(df)


def test_sexo_desconhecido_falha(tmp_path):
    with pytest.raises(ValueError, match="V0404"):
        parse_morador_2017(_escrever(tmp_path, [_linha_morador(V0404="3")]))


@pytest.mark.dados
def test_dias_pessoa_2017():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet")
    assert _numero(df["id_dia"].nunique()) == 85_018


@pytest.mark.dados
def test_soma_dos_pesos_2017_e_plausivel():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet")
    total = df.drop_duplicates("id_pessoa")["peso_final"].sum()
    assert 160e6 < _numero(total) < 200e6


@pytest.mark.dados
def test_domingos_e_sabados_2017():
    dias = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet").drop_duplicates("id_dia")
    assert _numero((dias["dia_semana"] == 7).sum()) == 16_585
    assert _numero((dias["dia_semana"] == 6).sum()) == 3_862


@pytest.mark.dados
def test_parquets_2017_seguem_o_esquema_comum():
    validar_consumo(pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet"))
    validar_morador(pd.read_parquet(DADOS_INTERIM / "morador_2017-2018.parquet"))
