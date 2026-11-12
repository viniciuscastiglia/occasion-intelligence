"""Testes do parser da POF 2008-2009: unitários com linhas montadas e conferência nos dados reais."""

from pathlib import Path
from typing import cast

import pandas as pd
import pandera.errors
import pytest

from occasion.config import DADOS_INTERIM
from occasion.ingest.parse import _gravar_interim, parse_consumo_2008, parse_morador_2008
from occasion.ingest.schema import validar_consumo, validar_morador

# Posições (base 1, tamanho) escritas aqui de propósito, independentes de layouts.py.
_CONSUMO = {
    "COD_UF": (3, 2),
    "NUM_SEQ": (5, 3),
    "NUM_DV": (8, 1),
    "NUM_DOM": (9, 2),
    "NUM_UC": (11, 1),
    "COD_INFORMANTE": (12, 2),
    "ESTRATO_POF": (14, 2),
    "PESO_FINAL": (31, 15),
    "NUM_QUADRO": (46, 2),
    "V9018": (48, 1),
    "V9015": (49, 2),
    "V9001": (64, 7),
    "RENDA_TOTAL": (115, 16),
    "DIA_SEMANA": (152, 1),
}
_MORADOR = {
    "COD_UF": (3, 2),
    "NUM_SEQ": (5, 3),
    "NUM_DV": (8, 1),
    "NUM_DOM": (9, 2),
    "NUM_UC": (11, 1),
    "COD_INFORMANTE": (12, 2),
    "ESTRATO_POF": (14, 2),
    "IDADE_ANOS": (60, 3),
    "V0405": (76, 2),
    "COD_UPA": (247, 6),
    "TIPO_SITUACAO_REG": (253, 1),
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
        "COD_UF": "35",
        "NUM_SEQ": "012",
        "NUM_DV": "3",
        "NUM_DOM": " 4",
        "NUM_UC": "1",
        "COD_INFORMANTE": " 2",
        "ESTRATO_POF": "05",
        "PESO_FINAL": "002058.05826960",
        "NUM_QUADRO": "71",
        "V9018": "2",
        "V9015": "20",
        "V9001": "8500903",
        "RENDA_TOTAL": "0000000000478.47",
        "DIA_SEMANA": "6",
    }
    return _linha(_CONSUMO, 152, **{**base, **sobrescritas})


def _linha_morador(**sobrescritas: str) -> str:
    base = {
        "COD_UF": "35",
        "NUM_SEQ": " 12",
        "NUM_DV": "3",
        "NUM_DOM": " 4",
        "NUM_UC": "1",
        "COD_INFORMANTE": " 2",
        "ESTRATO_POF": " 5",
        "IDADE_ANOS": " 53",
        "V0405": " 1",
        "COD_UPA": "350123",
        "TIPO_SITUACAO_REG": "1",
    }
    return _linha(_MORADOR, 263, **{**base, **sobrescritas})


def _numero(valor: object) -> float:
    """Converte o escalar devolvido por uma agregação do pandas (o pyright o vê como Series)."""
    return float(cast(float, valor))


def _escrever(tmp_path: Path, linhas: list[str], nome: str = "arquivo.txt") -> Path:
    caminho = tmp_path / nome
    caminho.write_bytes(("\n".join(linhas) + "\n").encode("latin1"))
    return caminho


def test_parse_consumo_2008_converte_campos(tmp_path):
    df = parse_consumo_2008(_escrever(tmp_path, [_linha_consumo()]))
    linha = df.iloc[0]
    assert linha.cod_upa == "350123"
    assert (linha.hora, linha.local, linha.dia_semana) == (20, "fora_de_casa", 6)
    assert (linha.uf, linha.quadro, linha.cod_alimento) == (35, 71, 8500903)
    assert (linha.num_dom, linha.num_uc, linha.cod_informante) == (4, 1, 2)
    assert linha.edicao == "2008-2009"
    assert linha.estrato == "35-05"
    assert linha.id_pessoa == "2008-2009|350123|4|1|2"
    assert linha.id_dia == "2008-2009|350123|4|1|2|71"
    assert linha.peso_final == pytest.approx(2058.0582696)
    assert linha.renda_total_uc == pytest.approx(478.47)
    validar_consumo(df)


def test_cod_upa_preserva_zeros_a_esquerda(tmp_path):
    df = parse_consumo_2008(_escrever(tmp_path, [_linha_consumo(NUM_SEQ="001", COD_UF="11")]))
    assert df.iloc[0]["cod_upa"] == "110013"


def test_decimais_sem_ponto_sao_divididos(tmp_path):
    linha = _linha_consumo(PESO_FINAL="000205805826960", RENDA_TOTAL="0000000000047847")
    df = parse_consumo_2008(_escrever(tmp_path, [linha]))
    assert df.iloc[0]["peso_final"] == pytest.approx(2058.0582696)
    assert df.iloc[0]["renda_total_uc"] == pytest.approx(478.47)


def test_dia_da_semana_numerico(tmp_path):
    linhas = [_linha_consumo(DIA_SEMANA=str(d)) for d in range(1, 8)]
    df = parse_consumo_2008(_escrever(tmp_path, linhas))
    assert df["dia_semana"].tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_dia_da_semana_desconhecido_falha(tmp_path):
    with pytest.raises(ValueError, match="DIA_SEMANA"):
        parse_consumo_2008(_escrever(tmp_path, [_linha_consumo(DIA_SEMANA="8")]))


def test_local_dentro_e_fora(tmp_path):
    df = parse_consumo_2008(
        _escrever(tmp_path, [_linha_consumo(V9018="1"), _linha_consumo(V9018="2")])
    )
    assert df["local"].tolist() == ["em_casa", "fora_de_casa"]


@pytest.mark.parametrize("codigo", ["9", " ", "3"])
def test_local_fora_do_dicionario_vira_indeterminado(tmp_path, codigo):
    df = parse_consumo_2008(_escrever(tmp_path, [_linha_consumo(V9018=codigo)]))
    assert df.iloc[0]["local"] == "indeterminado"


def test_hora_vazia_vira_nulo_e_aceita_formato_com_h(tmp_path):
    linhas = [_linha_consumo(V9015=""), _linha_consumo(V9015=" 8"), _linha_consumo(V9015="8H")]
    df = parse_consumo_2008(_escrever(tmp_path, linhas))
    assert pd.isna(df.iloc[0]["hora"])
    assert df["hora"].tolist()[1:] == [8, 8]
    assert str(df["hora"].dtype) == "Int64"
    validar_consumo(df)


def test_renda_vazia_nao_passa_em_silencio(tmp_path):
    with pytest.raises(ValueError, match="RENDA_TOTAL"):
        parse_consumo_2008(_escrever(tmp_path, [_linha_consumo(RENDA_TOTAL="")]))


def test_parse_morador_2008_converte_campos(tmp_path):
    arquivo = _escrever(
        tmp_path,
        [
            _linha_morador(COD_INFORMANTE=" 1", IDADE_ANOS=" 64", V0405=" 1"),
            _linha_morador(
                COD_INFORMANTE=" 2", IDADE_ANOS="  7", V0405=" 2", TIPO_SITUACAO_REG="2"
            ),
            _linha_morador(NUM_UC="2", COD_INFORMANTE=" 1", IDADE_ANOS="101", V0405=" 2"),
        ],
    )
    df = parse_morador_2008(arquivo)
    assert df["idade"].tolist() == [64, 7, 101]
    assert df["sexo"].tolist() == ["masculino", "feminino", "feminino"]
    assert df["situacao"].tolist() == ["urbano", "rural", "urbano"]
    assert df["n_moradores_uc"].tolist() == [2, 2, 1]
    assert df.iloc[0]["id_pessoa"] == "2008-2009|350123|4|1|1"
    assert df.iloc[0]["estrato"] == "35-05"
    validar_morador(df)


def test_cod_upa_do_consumo_igual_ao_do_morador(tmp_path):
    consumo = parse_consumo_2008(_escrever(tmp_path, [_linha_consumo()], "c.txt"))
    morador = parse_morador_2008(_escrever(tmp_path, [_linha_morador()], "m.txt"))
    assert consumo.iloc[0]["cod_upa"] == morador.iloc[0]["cod_upa"]
    assert consumo.iloc[0]["estrato"] == morador.iloc[0]["estrato"]
    assert consumo.iloc[0]["id_pessoa"] == morador.iloc[0]["id_pessoa"]


def test_sexo_desconhecido_falha(tmp_path):
    with pytest.raises(ValueError, match="V0405"):
        parse_morador_2008(_escrever(tmp_path, [_linha_morador(V0405=" 3")]))


def test_cod_upa_divergente_entre_campos_do_morador_falha(tmp_path):
    with pytest.raises(ValueError, match="COD_UPA"):
        parse_morador_2008(_escrever(tmp_path, [_linha_morador(COD_UPA="350124")]))


def test_morador_invalido_nao_grava_nenhum_parquet(tmp_path):
    consumo = parse_consumo_2008(_escrever(tmp_path, [_linha_consumo()], "c.txt"))
    morador = parse_morador_2008(_escrever(tmp_path, [_linha_morador()], "m.txt"))
    morador["n_moradores_uc"] = 5  # diverge do número de linhas da unidade de consumo
    saida = tmp_path / "saida"
    with pytest.raises(pandera.errors.SchemaErrors):
        _gravar_interim("2008-2009", consumo, morador, saida)
    assert not saida.exists() or list(saida.iterdir()) == []


@pytest.mark.dados
def test_dias_pessoa_2008():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    assert _numero(df["id_dia"].nunique()) == 66_903


@pytest.mark.dados
def test_soma_dos_pesos_2008_e_plausivel():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    total = df.drop_duplicates("id_pessoa")["peso_final"].sum()
    assert 140e6 < _numero(total) < 180e6


@pytest.mark.dados
def test_quadros_2008():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    assert set(df["quadro"].unique()) == {71, 72}


@pytest.mark.dados
def test_parquets_2008_seguem_o_esquema_comum():
    validar_consumo(pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet"))
    validar_morador(pd.read_parquet(DADOS_INTERIM / "morador_2008-2009.parquet"))
