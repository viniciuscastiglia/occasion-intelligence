"""Tela do Occasion Intelligence: contexto na barra lateral, resultado e painel do modelo.

A tela só chama `carregar_artefato` e `prever` e formata a saída; não calcula nada.
"""

import os
from pathlib import Path
from typing import Any, cast, get_args

import altair as alt
import pandas as pd
import streamlit as st

from occasion.config import MODELOS
from occasion.inference.contract import (
    UF,
    Contexto,
    DiaSemana,
    FaixaHorario,
    FaixaRenda,
    Fator,
    Local,
    Resultado,
    Sexo,
    Situacao,
)
from occasion.inference.predict import ArtefatoAusente, ArtefatoModelo, carregar_artefato, prever

COR_AUMENTA = "#2a78d6"
COR_DIMINUI = "#eb6834"

ROTULOS_DIA = {
    "segunda": "Segunda", "terca": "Terça", "quarta": "Quarta", "quinta": "Quinta",
    "sexta": "Sexta", "sabado": "Sábado", "domingo": "Domingo",
}  # fmt: skip
ROTULOS_HORARIO = {
    "madrugada": "Madrugada", "manha": "Manhã", "almoco": "Almoço", "tarde": "Tarde",
    "noite": "Noite",
}  # fmt: skip
ROTULOS_LOCAL = {"em_casa": "Em casa", "fora_de_casa": "Fora de casa"}
ROTULOS_SITUACAO = {"urbano": "Urbano", "rural": "Rural"}
ROTULOS_SEXO = {"masculino": "Masculino", "feminino": "Feminino"}
ROTULOS_RENDA = {
    "ate_meio_sm": "Até 1/2 salário mínimo por pessoa",
    "meio_a_1_sm": "De 1/2 a 1 salário mínimo",
    "de_1_a_2_sm": "De 1 a 2 salários mínimos",
    "de_2_a_5_sm": "De 2 a 5 salários mínimos",
    "mais_de_5_sm": "Mais de 5 salários mínimos",
}
ROTULOS_PARTE = {"ocasiao": "ocasião", "preferencia": "preferência", "direto": "direto"}
ROTULOS_VARIAVEL = {
    "dia_semana": "dia da semana", "faixa_horario": "faixa de horário", "local": "local",
    "uf": "UF", "regiao": "região", "situacao": "situação", "idade": "idade", "sexo": "sexo",
    "renda_pc_sm": "renda", "n_moradores_uc": "moradores", "tem_pizza": "tem pizza",
}  # fmt: skip
COR_CLASSIFICACAO = {"alta": "green", "média": "orange", "baixa": "red"}


def _formatar_opcao(rotulos: dict[str, str]) -> Any:
    return lambda codigo: rotulos.get(codigo, codigo)


@st.cache_resource
def _carregar(pasta: str) -> ArtefatoModelo:
    return carregar_artefato(Path(pasta))


def formulario() -> Contexto | None:
    """Barra lateral com os campos de `Contexto`; devolve o contexto ao clicar em Calcular."""
    with st.sidebar:
        st.header("Contexto")
        uf = st.selectbox("UF", get_args(UF), index=get_args(UF).index("SP"))
        situacao = st.radio(
            "Situação do domicílio",
            get_args(Situacao),
            format_func=_formatar_opcao(ROTULOS_SITUACAO),
            horizontal=True,
        )
        dia = st.selectbox(
            "Dia da semana",
            get_args(DiaSemana),
            index=get_args(DiaSemana).index("sabado"),
            format_func=_formatar_opcao(ROTULOS_DIA),
        )
        horario = st.selectbox(
            "Faixa de horário",
            get_args(FaixaHorario),
            index=get_args(FaixaHorario).index("noite"),
            format_func=_formatar_opcao(ROTULOS_HORARIO),
        )
        local = st.radio(
            "Local",
            get_args(Local),
            format_func=_formatar_opcao(ROTULOS_LOCAL),
            horizontal=True,
        )
        idade = st.number_input("Idade", min_value=18, max_value=110, value=28, step=1)
        sexo = st.radio(
            "Sexo",
            get_args(Sexo),
            format_func=_formatar_opcao(ROTULOS_SEXO),
            horizontal=True,
        )
        renda = st.selectbox(
            "Renda por pessoa",
            get_args(FaixaRenda),
            index=get_args(FaixaRenda).index("de_2_a_5_sm"),
            format_func=_formatar_opcao(ROTULOS_RENDA),
        )
        moradores = st.number_input("Moradores no domicílio", 1, 20, 3, step=1)
        if not st.button("Calcular", type="primary"):
            return None
    return Contexto(
        uf=cast(UF, uf),
        situacao=cast(Situacao, situacao),
        dia_semana=cast(DiaSemana, dia),
        faixa_horario=cast(FaixaHorario, horario),
        local=cast(Local, local),
        idade=int(idade),
        sexo=cast(Sexo, sexo),
        faixa_renda=cast(FaixaRenda, renda),
        numero_moradores=int(moradores),
    )


ROTULOS_REGIAO = {
    "norte": "Norte", "nordeste": "Nordeste", "sudeste": "Sudeste", "sul": "Sul",
    "centro_oeste": "Centro-Oeste",
}  # fmt: skip
_ROTULOS_DO_VALOR: dict[str, dict[str, str]] = {
    "dia_semana": ROTULOS_DIA,
    "faixa_horario": {c: r.lower() for c, r in ROTULOS_HORARIO.items()},
    "local": {c: r.lower() for c, r in ROTULOS_LOCAL.items()},
    "situacao": {c: r.lower() for c, r in ROTULOS_SITUACAO.items()},
    "sexo": {c: r.lower() for c, r in ROTULOS_SEXO.items()},
    "regiao": ROTULOS_REGIAO,
}
ALTURA_POR_FATOR = 36


def valor_legivel(variavel: str, valor: str) -> str:
    """Valor do fator em texto legível: sem códigos com sublinhado."""
    if variavel == "renda_pc_sm":
        try:
            return f"{float(valor):.1f} salário mínimo por pessoa".replace(".", ",")
        except ValueError:
            pass
    rotulos = _ROTULOS_DO_VALOR.get(variavel, {})
    return rotulos.get(valor, valor.replace("_", " "))


def rotulo_do_fator(f: Fator) -> str:
    variavel = ROTULOS_VARIAVEL.get(f.variavel, f.variavel)
    return f"{variavel}: {valor_legivel(f.variavel, f.valor)} ({ROTULOS_PARTE[f.parte]})"


def tabela_fatores(fatores: list[Fator]) -> pd.DataFrame:
    """Uma linha por fator, com rótulo legível único, contribuição e direção."""
    return pd.DataFrame(
        {
            "fator": [rotulo_do_fator(f) for f in fatores],
            "contribuicao": [f.contribuicao for f in fatores],
            "direção": [
                "aumenta a chance" if f.direcao == "aumenta" else "diminui a chance"
                for f in fatores
            ],
        }
    )


def grafico_fatores(fatores: list[Fator]) -> alt.Chart:
    """Barras horizontais das contribuições em log-chances; a cor e o texto dizem a direção."""
    tabela = tabela_fatores(fatores)
    return (
        alt.Chart(tabela)
        .mark_bar(size=18, cornerRadiusEnd=4)
        .encode(
            x=alt.X("contribuicao:Q", title="Contribuição (log-chances)"),
            y=alt.Y(
                "fator:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelLimit=400, labelOverlap=False),
            ),
            color=alt.Color(
                "direção:N",
                scale=alt.Scale(
                    domain=["aumenta a chance", "diminui a chance"],
                    range=[COR_AUMENTA, COR_DIMINUI],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["fator", "direção", alt.Tooltip("contribuicao:Q", format="+.3f")],
        )
        .properties(height=alt.Step(ALTURA_POR_FATOR))
    )


def mostrar_resultado(r: Resultado) -> None:
    st.subheader("Occasion Score")
    c1, c2, c3 = st.columns(3)
    c1.metric("Score (0 a 100)", f"{r.occasion_score:.0f}")
    c2.metric("Classificação", r.classification.capitalize())
    c3.metric("Probabilidade conjunta", f"{r.occasion_probability:.2%}")
    st.markdown(
        f"Posição entre os contextos da população de referência: "
        f"**ocasião** {r.score_components['ocasiao'].posicao:.0f}/100 "
        f"(pizza com refrigerante no mesmo horário) e "
        f"**preferência** {r.score_components['preferencia'].posicao:.0f}/100 "
        f"(refrigerante de guaraná, dado o contexto). "
        f"Preferência alta quer dizer guaraná acima do contexto mediano."
    )
    st.subheader("Fatores")
    st.altair_chart(grafico_fatores(r.top_factors), width="stretch")
    st.caption(
        "Cada barra é a contribuição do fator na parte indicada. A mesma variável pode "
        "aparecer nas duas partes com sinais opostos."
    )
    st.subheader("Recomendação")
    st.markdown(f"**Categoria:** {r.recommendation_category}")
    st.write(r.recommendation_reason)


def sobre_o_modelo(r: Resultado) -> None:
    m = r.model_metadata
    st.subheader("Sobre o modelo")
    st.markdown(
        f"- **Modelo:** {m.modelo}, degrau {m.degrau}\n"
        f"- **Treinado em:** {', '.join(m.treinado_em)}\n"
        f"- **Commit da receita:** `{m.receita_commit}`\n"
        f"- **Instante de previsão:** {m.instante_de_previsao}\n"
        f"- **Métricas:** {m.descricao_metricas}\n"
        f"- **Versão do contrato:** {m.versao_contrato}"
    )
    st.markdown("**Limites da fonte (POF/IBGE)**")
    st.markdown(
        "- Guaraná é o sabor registrado na pesquisa, não uma marca.\n"
        "- Não há data de calendário: só o dia da semana.\n"
        "- Não há canal: a base só distingue em casa de fora de casa.\n"
        "- As edições são 2008-2009 (treino) e 2017-2018 (teste); o score descreve "
        "associação nos dados, não efeito de nenhuma ativação."
    )
    st.caption(m.aviso)


def main() -> None:
    st.set_page_config(page_title="Occasion Intelligence", layout="wide")
    st.title("Occasion Intelligence: pizza com refrigerante de guaraná")
    pasta = os.environ.get("OCCASION_MODELOS", str(MODELOS))
    try:
        artefato = _carregar(pasta)
    except ArtefatoAusente as erro:
        st.error(str(erro))
        st.stop()
    contexto = formulario()
    if contexto is not None:
        st.session_state["resultado"] = prever(contexto, artefato)
    resultado: Resultado | None = st.session_state.get("resultado")
    if resultado is None:
        st.info("Preencha o contexto na barra lateral e clique em Calcular.")
        return
    mostrar_resultado(resultado)
    sobre_o_modelo(resultado)
    with st.expander("JSON do contrato"):
        st.json(resultado.model_dump())


if __name__ == "__main__":
    main()
