"""Tela do Occasion Intelligence: contexto na barra lateral, resultado e painel do modelo.

A tela só chama `carregar_artefato` e `prever` e formata a saída; não calcula nada.
"""

import contextlib
import os
import re
from pathlib import Path
from typing import Any, cast, get_args

import altair as alt
import pandas as pd
import streamlit as st

from occasion.briefing.cliente import ClienteHTTP, configuracao_de
from occasion.briefing.limite import LimiteGlobal
from occasion.briefing.servico import LIMITE_POR_SESSAO, Briefing, gerar_briefing
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
from occasion.inference.grade import tabela_do_mapa
from occasion.inference.predict import ArtefatoAusente, ArtefatoModelo, carregar_artefato, prever
from occasion.rotulos import (
    ROTULOS_DIA,
    ROTULOS_HORARIO,
    ROTULOS_LOCAL,
    ROTULOS_RENDA,
    ROTULOS_SEXO,
    ROTULOS_SITUACAO,
    rotulo_da_parte,
    rotulo_da_variavel,
    valor_legivel,
)

COR_AUMENTA = "#2a78d6"
COR_DIMINUI = "#eb6834"
# Rampa sequencial de um matiz só (azul), do claro ao escuro, para o mapa de calor.
COR_MAPA_CLARA = "#e8f1fb"
COR_MAPA_ESCURA = "#0b3d7a"
LIMITE_TEXTO_CLARO = 55
VARIAVEIS_DO_BRIEFING = ("BRIEFING_BASE_URL", "BRIEFING_MODELO", "BRIEFING_CHAVE")
MAXIMO_DE_MAPAS_EM_CACHE = 500
# Caracteres que o Markdown do Streamlit interpreta (inclusive `$` do LaTeX e `:` dos ícones).
_MARKDOWN_ATIVO = re.compile(r"([\\`*_{}\[\]()#+\-.!|$~<>:])")


def _formatar_opcao(rotulos: dict[str, str]) -> Any:
    return lambda codigo: rotulos.get(codigo, codigo)


@st.cache_resource
def _carregar(pasta: str) -> ArtefatoModelo:
    return carregar_artefato(Path(pasta))


@st.cache_resource
def _cache_de_briefings() -> dict[str, Briefing]:
    """Um dicionário do processo, compartilhado entre as sessões."""
    return {}


@st.cache_resource
def _limite_global() -> LimiteGlobal:
    """Limite de chamadas ao provedor do processo inteiro, somado ao limite por sessão."""
    return LimiteGlobal()


def texto_sem_markdown(texto: str) -> str:
    """Escapa o Markdown e o LaTeX do texto do LLM, para que ele apareça como foi escrito."""
    return _MARKDOWN_ATIVO.sub(r"\\\1", texto)


def fonte_da_configuracao() -> dict[str, str]:
    """`st.secrets` e `os.environ` combinados; a variável de ambiente prevalece.

    Sem `secrets.toml`, `st.secrets` lança exceção: aqui isso vale como fonte vazia.
    """
    fonte: dict[str, str] = {}
    for nome in VARIAVEIS_DO_BRIEFING:
        # Sem arquivo de segredos: FileNotFoundError (StreamlitSecretNotFoundError herda dele).
        with contextlib.suppress(FileNotFoundError, KeyError):
            fonte[nome] = str(st.secrets[nome])
    fonte.update({n: os.environ[n] for n in VARIAVEIS_DO_BRIEFING if n in os.environ})
    return fonte


@st.cache_data(show_spinner="Calculando o mapa de calor...", max_entries=MAXIMO_DE_MAPAS_EM_CACHE)
def _scores_do_mapa(contexto_json: str, pasta: str) -> pd.DataFrame:
    """Tabela do mapa; a chave é o JSON do contexto com dia e faixa normalizados."""
    contexto = Contexto.model_validate_json(contexto_json)
    return tabela_do_mapa(contexto, _carregar(pasta))


def tabela_do_mapa_na_tela(contexto: Contexto, pasta: str) -> pd.DataFrame:
    """Scores em cache (independentes de dia e faixa do contexto) e a marca da célula atual."""
    normalizado = contexto.model_copy(
        update={"dia_semana": get_args(DiaSemana)[0], "faixa_horario": get_args(FaixaHorario)[0]}
    )
    tabela = _scores_do_mapa(normalizado.model_dump_json(), pasta).copy()
    tabela["atual"] = (tabela["dia_semana"] == contexto.dia_semana) & (
        tabela["faixa_horario"] == contexto.faixa_horario
    )
    return tabela


def grafico_mapa(tabela: pd.DataFrame) -> alt.LayerChart:
    """Mapa de calor dia x faixa: escala de 0 a 100, valor em cada célula, contorno na atual."""
    dados = tabela.assign(
        dia=tabela["dia_semana"].replace(ROTULOS_DIA),
        faixa=tabela["faixa_horario"].replace(ROTULOS_HORARIO),
    )
    eixo_x = alt.X(
        "faixa:N",
        sort=[ROTULOS_HORARIO[f] for f in get_args(FaixaHorario)],
        title=None,
        axis=alt.Axis(orient="top", labelAngle=0),
    )
    eixo_y = alt.Y("dia:N", sort=[ROTULOS_DIA[d] for d in get_args(DiaSemana)], title=None)
    base = alt.Chart(dados).encode(x=eixo_x, y=eixo_y)
    celulas = base.mark_rect(stroke="white", strokeWidth=2).encode(
        color=alt.Color(
            "score:Q",
            scale=alt.Scale(domain=[0, 100], range=[COR_MAPA_CLARA, COR_MAPA_ESCURA]),
            legend=alt.Legend(title="Posição (0 a 100)"),
        ),
        tooltip=[
            alt.Tooltip("dia:N", title="Dia"),
            alt.Tooltip("faixa:N", title="Faixa"),
            alt.Tooltip("score:Q", title="Score", format=".0f"),
        ],
    )
    valores = base.mark_text(fontSize=13).encode(
        text=alt.Text("score:Q", format=".0f"),
        color=alt.condition(
            alt.datum.score > LIMITE_TEXTO_CLARO, alt.value("white"), alt.value("#1f2a37")
        ),
    )
    contorno = (
        base.transform_filter(alt.datum.atual)
        .mark_rect(fill=None, stroke="#eb6834", strokeWidth=4)
        .encode()
    )
    return cast(alt.LayerChart, celulas + valores + contorno).properties(height=alt.Step(40))


def mostrar_mapa(contexto: Contexto, pasta: str) -> None:
    st.subheader("Mapa de calor: dia e faixa de horário")
    st.altair_chart(grafico_mapa(tabela_do_mapa_na_tela(contexto, pasta)), width="stretch")
    st.caption(
        "Posição de cada combinação entre os contextos da população de referência, com os "
        "demais campos fixos. O contorno marca o contexto calculado."
    )


def mostrar_briefing(contexto: Contexto, resultado: Resultado) -> None:
    """Botão e texto do briefing; usa o contexto do último cálculo, nunca o formulário."""
    st.session_state.setdefault("chamadas_llm", 0)
    if st.button("Gerar briefing"):
        config = configuracao_de(fonte_da_configuracao())
        cliente = ClienteHTTP(config) if config else None
        with st.spinner("Gerando briefing..."):
            briefing, chamou = gerar_briefing(
                contexto,
                resultado,
                cliente,
                _cache_de_briefings(),
                st.session_state["chamadas_llm"],
                _limite_global(),
            )
        st.session_state["chamadas_llm"] += int(chamou)
        st.session_state["briefing"] = briefing
    briefing: Briefing | None = st.session_state.get("briefing")
    if briefing is None:
        st.caption(f"Até {LIMITE_POR_SESSAO} chamadas ao provedor por sessão.")
        return
    st.markdown(texto_sem_markdown(briefing.texto))
    if briefing.fonte == "llm":
        st.caption(f"Gerado por {briefing.modelo} e conferido automaticamente")
    else:
        st.caption(f"Texto reserva: {briefing.motivo}")


def formulario() -> tuple[Contexto, bool]:
    """Barra lateral com os campos de `Contexto`; devolve o contexto atual e se Calcular foi clicado."""
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
        calculado = st.button("Calcular", type="primary")
    contexto = Contexto(
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
    return contexto, calculado


ALTURA_POR_FATOR = 36


def rotulo_do_fator(f: Fator) -> str:
    variavel = rotulo_da_variavel(f.variavel)
    return f"{variavel}: {valor_legivel(f.variavel, f.valor)} ({rotulo_da_parte(f.parte)})"


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
    c3.metric("Probabilidade conjunta", f"{r.occasion_probability:.2%}".replace(".", ","))
    c3.caption("Chance estimada de pizza com refrigerante de guaraná na célula.")
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


_ROTULO_METRICA = {
    "pr_auc": "PR-AUC",
    "log_loss": "log loss",
    "roc_auc": "ROC-AUC",
    "brier": "Brier",
    "pr_auc_degrau_a": "PR-AUC do degrau A",
}


def _linha_metricas(metricas: dict[str, float] | None) -> str:
    """Os números do teste temporal da receita, se o artefato os trouxer."""
    if not metricas:
        return ""
    itens = "; ".join(
        f"{_ROTULO_METRICA.get(k, k)} {v:.4g}".replace(".", ",") for k, v in metricas.items()
    )
    return f"- **Métricas do teste temporal da receita (2017-2018):** {itens}\n"


def limitacao_das_edicoes(modelo: str) -> str:
    """O que dizer das edições, conforme o modelo seja o avaliado ou o servido."""
    if modelo == "servido":
        return (
            "A receita foi avaliada com treino em 2008-2009 e teste em 2017-2018; o modelo "
            "servido foi reajustado com as duas edições e não tem teste temporal próprio"
        )
    return "As edições são 2008-2009 (treino) e 2017-2018 (teste)"


def sobre_o_modelo(r: Resultado) -> None:
    m = r.model_metadata
    st.subheader("Sobre o modelo")
    st.markdown(
        f"- **Modelo:** {m.modelo}, degrau {m.degrau}\n"
        f"- **Treinado em:** {', '.join(m.treinado_em)}\n"
        f"- **Commit da receita:** `{m.receita_commit}`\n"
        f"- **Instante de previsão:** {m.instante_de_previsao}\n"
        f"- **Métricas:** {m.descricao_metricas}\n"
        f"{_linha_metricas(m.metricas_da_receita)}"
        f"- **Versão do contrato:** {m.versao_contrato}"
    )
    st.markdown("**Limites da fonte (POF/IBGE)**")
    st.markdown(
        "- Guaraná é o sabor registrado na pesquisa, não uma marca.\n"
        "- Não há data de calendário: só o dia da semana.\n"
        "- Não há canal: a base só distingue em casa de fora de casa.\n"
        "- A base é anterior à expansão do delivery.\n"
        "- O método de coleta difere entre as edições: registro alimentar em 2008-2009 e "
        "recordatório de 24 horas em 2017-2018.\n"
        "- O local de 2008-2009 é a fonte do alimento (dentro ou fora do domicílio), e não o "
        "local de refeição de 2017-2018.\n"
        f"- {limitacao_das_edicoes(m.modelo)}; o score descreve "
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
    contexto, calculado = formulario()
    if calculado:
        st.session_state["resultado"] = prever(contexto, artefato)
        st.session_state["contexto_usado"] = contexto
        st.session_state.pop("briefing", None)
    resultado: Resultado | None = st.session_state.get("resultado")
    if resultado is None:
        st.info("Preencha o contexto na barra lateral e clique em Calcular.")
        return
    if st.session_state.get("contexto_usado") != contexto:
        st.warning("O contexto mudou desde o último cálculo. Clique em Calcular de novo.")
    contexto_usado: Contexto = st.session_state["contexto_usado"]
    mostrar_resultado(resultado)
    mostrar_briefing(contexto_usado, resultado)
    mostrar_mapa(contexto_usado, pasta)
    sobre_o_modelo(resultado)
    with st.expander("JSON do contrato"):
        st.json(resultado.model_dump())


if __name__ == "__main__":
    main()
