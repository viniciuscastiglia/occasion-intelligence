"""Testes das ocasiões e da base pessoa x dia: unitários com itens montados e conferência nos dados reais."""

import pandas as pd
import pytest

from occasion.bases.ocasioes import montar_ocasioes, montar_pessoa_dia
from occasion.config import CODIGOS, DADOS_PROCESSADOS

EDICAO = "2017-2018"


def quadro_de_consumo(
    itens: list[tuple[int | None, int, str]], id_pessoa: str = f"{EDICAO}|111|1|1|1"
) -> pd.DataFrame:
    """Monta um consumo de um único dia com itens (hora, cod_alimento, local)."""
    n = len(itens)
    return pd.DataFrame(
        {
            "edicao": pd.array([EDICAO] * n, dtype="string"),
            "uf": [35] * n,
            "cod_upa": pd.array(["111"] * n, dtype="string"),
            "estrato": pd.array(["35-1"] * n, dtype="string"),
            "num_dom": [1] * n,
            "num_uc": [1] * n,
            "cod_informante": [1] * n,
            "id_pessoa": pd.array([id_pessoa] * n, dtype="string"),
            "quadro": [72] * n,
            "id_dia": pd.array([f"{id_pessoa}|72"] * n, dtype="string"),
            "cod_alimento": [i[1] for i in itens],
            "hora": pd.array([i[0] for i in itens], dtype="Int64"),
            "local": pd.array([i[2] for i in itens], dtype="string"),
            "dia_semana": [3] * n,
            "peso_final": [100.0] * n,
            "renda_total_uc": [2000.0] * n,
        }
    )


def quadro_de_morador(ids: list[str]) -> pd.DataFrame:
    n = len(ids)
    return pd.DataFrame(
        {
            "edicao": pd.array([EDICAO] * n, dtype="string"),
            "uf": [35] * n,
            "cod_upa": pd.array(["111"] * n, dtype="string"),
            "estrato": pd.array(["35-1"] * n, dtype="string"),
            "num_dom": [1] * n,
            "num_uc": [1] * n,
            "cod_informante": list(range(1, n + 1)),
            "id_pessoa": pd.array(ids, dtype="string"),
            "situacao": pd.array(["urbano"] * n, dtype="string"),
            "idade": [30] * n,
            "sexo": pd.array(["feminino"] * n, dtype="string"),
            "n_moradores_uc": [n] * n,
        }
    )


PIZZA = 8500903
GUARANA = 8200301
COLA = 8200101


def test_pizza_e_guarana_na_mesma_hora_marcam_a_ocasiao():
    consumo = quadro_de_consumo(
        [(20, PIZZA, "em_casa"), (20, GUARANA, "em_casa"), (12, COLA, "fora_de_casa")]
    )
    oc = montar_ocasioes(consumo, EDICAO).set_index("hora")
    assert oc.loc[20, ["tem_pizza", "tem_refri", "tem_guarana", "tem_cola"]].tolist() == [
        True,
        True,
        True,
        False,
    ]
    assert oc.loc[12, ["tem_pizza", "tem_refri", "tem_cola"]].tolist() == [False, True, True]


def test_guarana_e_cola_juntos_e_item_repetido_nao_contam_em_dobro():
    consumo = quadro_de_consumo(
        [(20, GUARANA, "em_casa"), (20, GUARANA, "em_casa"), (20, COLA, "em_casa")]
    )
    oc = montar_ocasioes(consumo, EDICAO)
    assert len(oc) == 1
    assert oc.iloc[0][["tem_guarana", "tem_cola"]].tolist() == [True, True]
    assert oc.iloc[0]["id_ocasiao"] == f"{EDICAO}|111|1|1|1|72|20"


def test_pizza_e_refri_em_horas_diferentes_nao_e_ocasiao_conjunta():
    consumo = quadro_de_consumo([(13, PIZZA, "em_casa"), (20, GUARANA, "em_casa")])
    oc = montar_ocasioes(consumo, EDICAO)
    assert not (oc.tem_pizza & oc.tem_refri).any()
    pdia = montar_pessoa_dia(consumo, quadro_de_morador([consumo.id_pessoa[0]]), oc)
    assert pdia.tem_pizza_dia.tolist() == [True]
    assert pdia.tem_pizza_refri.tolist() == [False]
    assert pdia.tem_pizza_guarana.tolist() == [False]


def test_local_da_ocasiao_e_o_do_item_de_pizza():
    consumo = quadro_de_consumo(
        [(20, COLA, "fora_de_casa"), (20, PIZZA, "em_casa"), (20, 8200102, "fora_de_casa")]
    )
    assert montar_ocasioes(consumo, EDICAO).local.tolist() == ["em_casa"]


def test_local_sem_pizza_e_o_do_primeiro_refrigerante_e_senao_o_do_primeiro_item():
    consumo = quadro_de_consumo(
        [(20, 6300101, "em_casa"), (20, COLA, "fora_de_casa"), (20, GUARANA, "em_casa")]
    )
    assert montar_ocasioes(consumo, EDICAO).local.tolist() == ["fora_de_casa"]
    sem_refri = quadro_de_consumo([(8, 6300101, "fora_de_casa"), (8, 6300201, "em_casa")])
    assert montar_ocasioes(sem_refri, EDICAO).local.tolist() == ["fora_de_casa"]


def test_pessoa_sem_morador_levanta_erro():
    consumo = quadro_de_consumo([(20, PIZZA, "em_casa")])
    oc = montar_ocasioes(consumo, EDICAO)
    with pytest.raises(ValueError, match="1"):
        montar_pessoa_dia(consumo, quadro_de_morador([f"{EDICAO}|111|1|1|2"]), oc)


def test_itens_sem_hora_sao_contados_e_nao_formam_ocasiao():
    oc = montar_ocasioes(
        quadro_de_consumo([(None, PIZZA, "em_casa"), (20, GUARANA, "em_casa")]), EDICAO
    )
    assert len(oc) == 1 and oc.attrs["itens_sem_hora"] == 1


def test_dia_com_pizza_sem_hora_conta_como_dia_com_pizza_mas_nao_como_conjunta():
    consumo = quadro_de_consumo([(None, PIZZA, "em_casa"), (20, GUARANA, "em_casa")])
    oc = montar_ocasioes(consumo, EDICAO)
    pdia = montar_pessoa_dia(consumo, quadro_de_morador([consumo.id_pessoa[0]]), oc)
    assert pdia.tem_pizza_dia.tolist() == [True]
    assert pdia.tem_pizza_refri.tolist() == [False]


def test_pessoa_dia_traz_morador_e_uma_linha_por_dia():
    consumo = quadro_de_consumo([(20, PIZZA, "em_casa"), (20, GUARANA, "em_casa")])
    pdia = montar_pessoa_dia(
        consumo, quadro_de_morador([consumo.id_pessoa[0]]), montar_ocasioes(consumo, EDICAO)
    )
    assert len(pdia) == 1
    linha = pdia.iloc[0]
    assert (linha.idade, linha.sexo, linha.dia_semana, linha.peso_final) == (
        30,
        "feminino",
        3,
        100.0,
    )
    assert linha.renda_total_uc == 2000.0
    assert (linha.tem_pizza_refri, linha.tem_pizza_guarana) == (True, True)


@pytest.mark.parametrize("coluna", ["dia_semana", "peso_final", "renda_total_uc"])
def test_coluna_do_dia_que_varia_dentro_do_dia_levanta_erro(coluna):
    consumo = quadro_de_consumo([(20, PIZZA, "em_casa"), (21, GUARANA, "em_casa")])
    consumo.loc[1, coluna] = consumo.loc[1, coluna] + 1
    with pytest.raises(ValueError, match=coluna):
        montar_pessoa_dia(
            consumo, quadro_de_morador([consumo.id_pessoa[0]]), montar_ocasioes(consumo, EDICAO)
        )


def test_listas_de_codigos_sao_disjuntas_e_cobrem_as_duas_edicoes():
    for edicao, listas in CODIGOS.items():
        assert set(listas) == {"pizza", "guarana", "cola", "outros_refri"}
        nomes = list(listas)
        for i, a in enumerate(nomes):
            for b in nomes[i + 1 :]:
                assert not listas[a] & listas[b], (edicao, a, b)
    assert set(CODIGOS) == {"2008-2009", "2017-2018"}


# A tabela antiga do CLAUDE.md trazia 783/176 para 2017-2018 porque aquelas contagens usaram a chave
# dia+hora+refeicao (V9017), que nao existe em 2008-2009. A ocasiao aqui e dia+hora nas duas edicoes.
# A tabela antiga do CLAUDE.md trazia 783/176 para 2017-2018 porque aquelas contagens usaram a chave
# dia+hora+refeicao (V9017), que nao existe em 2008-2009. A ocasiao aqui e dia+hora nas duas edicoes.
REFERENCIA = {"2017-2018": (85_018, 1_681, 793, 180), "2008-2009": (66_903, 1_149, 635, 101)}


@pytest.mark.dados
@pytest.mark.parametrize("edicao", ["2017-2018", "2008-2009"])
def test_numeros_de_referencia(edicao):
    pd_ = pd.read_parquet(DADOS_PROCESSADOS / f"pessoa_dia_{edicao}.parquet")
    obtido = (
        len(pd_),
        int(pd_.tem_pizza_dia.sum()),
        int(pd_.tem_pizza_refri.sum()),
        int(pd_.tem_pizza_guarana.sum()),
    )
    assert obtido == REFERENCIA[edicao]


@pytest.mark.dados
def test_ocasioes_com_refrigerante_2017():
    oc = pd.read_parquet(DADOS_PROCESSADOS / "ocasioes_2017-2018.parquet")
    assert oc.tem_refri.sum() == 15_863
