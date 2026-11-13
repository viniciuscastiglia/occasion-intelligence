"""Testes do Decision Engine: regras explícitas, cortes inclusivos e texto sem termos proibidos."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from occasion.decision.engine import Decisao, decidir, justificar
from occasion.inference.contract import Fator

LOCAIS = ["em_casa", "fora_de_casa"]
UFS = ["SP", "BA", "PE", "RJ", "AM", "RS", "DF", "MT"]
FRASE_FINAL = (
    "A recomendação indica que a ativação combina com o contexto; ela não estima o efeito da ação."
)
TERMOS_PROIBIDOS = ["vende mais", "aumenta as vendas", "causa", "guaraná antarctica"]

CATEGORIAS_POSSIVEIS = {
    "Sustentar, via ação na pizzaria",
    "Sustentar, via combo em app de delivery",
    "Sustentar, via pizza e guaraná juntos no varejo",
    "Converter, via influencers e conteúdo digital",
    "Converter, via ação na pizzaria",
    "Converter, via combo em app de delivery",
    "Converter, via pizza e guaraná juntos no varejo",
    "Criar ocasião, via ação na pizzaria",
    "Criar ocasião, via combo em app de delivery",
    "Criar ocasião, via pizza e guaraná juntos no varejo",
    "Não priorizar",
}


def test_converter_jovem_vai_para_influencers():
    d = decidir(90, 20, "em_casa", "urbano", 24, "SP", frozenset())
    assert (d.objetivo, d.canal_principal, d.regra) == (
        "Converter",
        "influencers e conteúdo digital",
        "canal-2",
    )
    assert d.categoria == "Converter, via influencers e conteúdo digital"


def test_converter_nao_jovem_segue_para_regras_de_local():
    d = decidir(90, 20, "em_casa", "urbano", 30, "SP", frozenset())
    assert (d.canal_principal, d.regra) == ("combo em app de delivery", "canal-4")


def test_criar_ocasiao_jovem_nao_vai_para_influencers():
    d = decidir(10, 80, "fora_de_casa", "urbano", 20, "SP", frozenset())
    assert (d.objetivo, d.canal_principal) == ("Criar ocasião", "ação na pizzaria")


def test_sustentar_em_casa_urbano_vai_para_delivery():
    d = decidir(90, 80, "em_casa", "urbano", 24, "SP", frozenset())
    assert (d.objetivo, d.canal_principal) == ("Sustentar", "combo em app de delivery")
    assert d.regra == "canal-4"


def test_rural_em_casa_vai_para_varejo():
    d = decidir(90, 80, "em_casa", "rural", 40, "SP", frozenset())
    assert d.canal_principal == "pizza e guaraná juntos no varejo"
    assert d.regra == "canal-5"
    assert d.categoria == "Sustentar, via pizza e guaraná juntos no varejo"


def test_fora_de_casa_vai_para_pizzaria():
    d = decidir(90, 80, "fora_de_casa", "rural", 40, "SP", frozenset())
    assert d.canal_principal == "ação na pizzaria"
    assert d.regra == "canal-3"


def test_nao_priorizar_nao_tem_canal_nem_midia_regional():
    d = decidir(10, 10, "em_casa", "urbano", 40, "BA", frozenset({"BA"}))
    assert (d.canal_principal, d.canais_complementares, d.categoria) == (None, (), "Não priorizar")
    assert d.regra == "canal-1"


def test_midia_regional_e_complementar_quando_uf_esta_no_conjunto():
    d = decidir(90, 80, "em_casa", "urbano", 40, "BA", frozenset({"BA"}))
    assert d.canais_complementares == ("mídia regional",)
    d2 = decidir(90, 80, "em_casa", "urbano", 40, "SP", frozenset({"BA"}))
    assert d2.canais_complementares == ()


@pytest.mark.parametrize(
    "oc,pref,obj",
    [
        (75, 50, "Sustentar"),
        (74.9, 50, "Criar ocasião"),
        (75, 49.9, "Converter"),
        (74.9, 49.9, "Não priorizar"),
    ],
)
def test_cortes_sao_inclusivos(oc, pref, obj):
    assert decidir(oc, pref, "em_casa", "urbano", 40, "SP", frozenset()).objetivo == obj


def test_idade_jovem_e_estritamente_menor_que_30():
    assert decidir(90, 20, "em_casa", "urbano", 29, "SP", frozenset()).regra == "canal-2"
    assert decidir(90, 20, "em_casa", "urbano", 30, "SP", frozenset()).regra == "canal-4"


def test_justificar_cita_objetivo_regra_e_ate_tres_fatores_na_ordem():
    d = decidir(90, 80, "em_casa", "urbano", 40, "SP", frozenset())
    fatores = [
        Fator(
            parte="ocasiao",
            variavel=f"v{i}",
            valor=f"x{i}",
            direcao="aumenta",
            contribuicao=1.0 / (i + 1),
        )
        for i in range(5)
    ]
    texto = justificar(d, fatores)
    assert "Sustentar" in texto and "regra: em casa e área urbana" in texto
    assert "canal-" not in texto and "_" not in texto
    assert "v0" in texto and "v1" in texto and "v2" in texto
    assert "v3" not in texto and "v4" not in texto
    assert texto.index("v0") < texto.index("v1") < texto.index("v2")
    assert texto.endswith(FRASE_FINAL)


def test_justificar_com_lista_vazia_continua_valida():
    d = decidir(10, 10, "em_casa", "urbano", 40, "SP", frozenset())
    texto = justificar(d, [])
    assert "Não priorizar" in texto
    assert texto.endswith(FRASE_FINAL)


def test_decisao_e_imutavel():
    d = decidir(90, 80, "em_casa", "urbano", 40, "SP", frozenset())
    assert isinstance(d, Decisao)
    with pytest.raises(AttributeError):
        d.objetivo = "Converter"  # type: ignore[misc]


ESTRATEGIA_FATOR = st.builds(
    Fator,
    parte=st.sampled_from(["ocasiao", "preferencia", "direto"]),
    variavel=st.sampled_from(["faixa_horario", "idade", "uf", "local"]),
    valor=st.sampled_from(["noite", "24", "SP", "em_casa"]),
    direcao=st.sampled_from(["aumenta", "diminui"]),
    contribuicao=st.floats(-5, 5, allow_nan=False),
)


@given(
    oc=st.floats(0, 100),
    pref=st.floats(0, 100),
    local=st.sampled_from(LOCAIS),
    situacao=st.sampled_from(["urbano", "rural"]),
    idade=st.integers(18, 110),
    uf=st.sampled_from(UFS),
)
def test_toda_entrada_valida_cai_em_exatamente_uma_categoria(oc, pref, local, situacao, idade, uf):
    d = decidir(oc, pref, local, situacao, idade, uf, frozenset({"BA", "PE"}))
    assert d.categoria in CATEGORIAS_POSSIVEIS
    assert (d.canal_principal is None) == (d.objetivo == "Não priorizar")
    assert (d.canais_complementares == ()) == (
        d.objetivo == "Não priorizar" or uf not in {"BA", "PE"}
    )


@given(
    oc=st.floats(0, 100),
    pref=st.floats(0, 100),
    local=st.sampled_from(LOCAIS),
    situacao=st.sampled_from(["urbano", "rural"]),
    idade=st.integers(18, 110),
    uf=st.sampled_from(UFS),
    fatores=st.lists(ESTRATEGIA_FATOR, max_size=6),
)
def test_justificativa_nao_usa_termos_proibidos(oc, pref, local, situacao, idade, uf, fatores):
    d = decidir(oc, pref, local, situacao, idade, uf, frozenset({"BA", "PE"}))
    texto = justificar(d, fatores)
    assert not any(t in texto.lower() for t in TERMOS_PROIBIDOS)
    assert "_" not in texto and "canal-" not in texto
    assert texto.endswith(FRASE_FINAL)


def test_justificar_escreve_cada_fator_com_rotulo_valor_e_parte():
    d = decidir(90, 80, "em_casa", "urbano", 40, "SP", frozenset())
    fatores = [
        Fator(
            parte="ocasiao",
            variavel="faixa_horario",
            valor="noite",
            direcao="aumenta",
            contribuicao=1.0,
        ),
        Fator(
            parte="preferencia",
            variavel="regiao",
            valor="centro_oeste",
            direcao="diminui",
            contribuicao=-0.5,
        ),
    ]
    texto = justificar(d, fatores)
    assert "faixa de horário: noite (ocasião, puxa para cima)" in texto
    assert "região: Centro-Oeste (preferência, puxa para baixo)" in texto
    assert "_" not in texto and "canal-" not in texto
