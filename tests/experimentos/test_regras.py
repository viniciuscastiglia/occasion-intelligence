from occasion.experimentos.regras import (
    correlacao_positiva,
    melhor_logistico,
    padrao,
    sinal_negativo,
    superioridade,
    vencedor_escada,
)
from occasion.validacao.bootstrap import Intervalo

POS = Intervalo(0.1, 0.01, 0.2, 1000)
NEG = Intervalo(-0.1, -0.2, -0.01, 1000)
EMPATE = Intervalo(0.1, -0.01, 0.2, 1000)


def test_regras():
    assert superioridade(Intervalo(0.1, 0.01, 0.2, 1000)) == "confirmada"
    assert superioridade(Intervalo(0.1, -0.01, 0.2, 1000)) == "refutada"
    assert sinal_negativo(Intervalo(-0.5, -0.9, -0.1, 1000)) == "confirmada"
    assert sinal_negativo(Intervalo(0.5, 0.1, 0.9, 1000)) == "efeito no sentido contrário"
    assert sinal_negativo(Intervalo(0.1, -0.2, 0.4, 1000)) == "os dados não mostram o efeito"
    assert padrao(Intervalo(0.1, -0.01, 0.2, 1000)) == "sem evidência contra o padrão"
    assert padrao(POS) == "padrão refutado"


def test_ic_inf_zero_nao_confirma():
    assert superioridade(Intervalo(0.1, 0.0, 0.2, 1000)) == "refutada"


def test_correlacao_positiva_exige_todos_os_ic_inf_positivos():
    assert correlacao_positiva([POS, POS]) == "confirmada"
    assert correlacao_positiva([POS, EMPATE]) == "refutada"


def test_melhor_logistico():
    assert melhor_logistico(POS) == "C"
    assert melhor_logistico(EMPATE) == "B"
    assert melhor_logistico(NEG) == "B"


def test_escada_sem_h1_exige_superar_a_tabela():
    ganhos = {("B", "A"): NEG, ("C", "A"): NEG, ("D", "A"): POS}
    vencedor, comparacoes = vencedor_escada(lambda c, r: ganhos[(c, r)])
    assert vencedor == "D" and [c["referencia"] for c in comparacoes] == ["A", "A", "A"]
    assert [c["candidato"] for c in comparacoes] == ["B", "C", "D"]
    assert [c["veredito"] for c in comparacoes] == ["refutada", "refutada", "confirmada"]
    assert comparacoes[2]["intervalo"] is POS


def test_escada_empate_fica_com_o_mais_simples():
    ganhos = {("B", "A"): POS, ("C", "B"): EMPATE, ("D", "B"): EMPATE}
    vencedor, comparacoes = vencedor_escada(lambda c, r: ganhos[(c, r)])
    assert vencedor == "B"
    assert [(c["candidato"], c["referencia"]) for c in comparacoes] == [
        ("B", "A"),
        ("C", "B"),
        ("D", "B"),
    ]


def test_escada_campeao_troca_a_referencia():
    ganhos = {("B", "A"): POS, ("C", "B"): POS, ("D", "C"): POS}
    vencedor, comparacoes = vencedor_escada(lambda c, r: ganhos[(c, r)])
    assert vencedor == "D"
    assert [c["referencia"] for c in comparacoes] == ["A", "B", "C"]
