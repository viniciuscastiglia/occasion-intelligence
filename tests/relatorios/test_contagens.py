import pandas as pd
import pytest

from occasion.bases.acesso import carregar_base
from occasion.relatorios.contagens import contar


def test_contar_em_base_sintetica():
    linhas = []
    for dia, dia_semana in (("a", 1), ("b", 7)):
        for k in range(10):
            linhas.append(
                {
                    "id_dia": dia,
                    "dia_semana": dia_semana,
                    "y_pizza_refri": int(dia == "a" and k == 0),
                    "y_pizza_guarana": int(dia == "a" and k == 0),
                }
            )
    celulas = pd.DataFrame(linhas)
    ocasioes = pd.DataFrame({"id_dia": ["a", "a", "b"]})

    r = contar(celulas, ocasioes)

    assert r["dias_pessoa"] == 2
    assert r["celulas"] == 20
    assert r["dias_pizza_refri"] == 1
    assert r["dias_pizza_guarana"] == 1
    assert r["ocasioes_refri"] == 3
    assert sum(r["dias_por_dia_semana"].values()) == 2
    assert r["dias_por_dia_semana"]["1"] == 1
    assert r["dias_por_dia_semana"]["7"] == 1
    assert r["dias_por_dia_semana"]["3"] == 0


@pytest.mark.dados
def test_contagens_de_2008_batem_com_a_referencia():
    r = contar(carregar_base("celulas", "2008-2009"), carregar_base("ocasioes_refri", "2008-2009"))
    assert r["dias_pessoa"] == 66_903
    assert r["celulas"] == 669_030
    assert r["dias_pizza_refri"] == 651
    assert r["dias_pizza_guarana"] == 101
    assert r["ocasioes_refri"] == 19_182
