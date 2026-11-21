"""Relatório das hipóteses H1 a H10 (pré-registro, seção 5), consolidado dos JSONs das etapas.

Esta camada só lê e organiza: o veredito é copiado do JSON de cada etapa, nunca recalculado aqui
(as regras vivem em `regras.py` e foram aplicadas na etapa). Enunciado, teste e métrica são os
textos do pré-registro. Hipótese cujo JSON não existe sai com `veredito = "não executada"` e o
motivo. Grava `reports/hipoteses.json` com uma entrada por hipótese (H1 a H3 uma vez por parte; H9
uma vez por comparação).
"""

import json
from pathlib import Path
from typing import Any

from occasion.config import RELATORIOS

NAO_EXECUTADA = "não executada"
PARTES = ("ocasiao", "preferencia")

_ENUNCIADOS: dict[str, str] = {
    "H1": "em cada parte, a regressão logística prevê melhor que a tabela de taxas.",
    "H2": (
        "a logística com curvas para idade e renda e com as interações pré-declaradas prevê "
        "melhor que a logística simples."
    ),
    "H3": "o LightGBM prevê melhor que o melhor modelo logístico.",
    "H4": (
        "dado o mesmo contexto, a presença de pizza reduz a chance de o refrigerante ser guaraná."
    ),
    "H5": (
        "treinar com `PESO_FINAL` dá previsões melhores para a população do que treinar sem peso."
    ),
    "H6": (
        "o produto das duas partes prevê pizza com guaraná melhor que um modelo treinado direto "
        "nesse evento."
    ),
    "H7": "o produto das duas partes corresponde à frequência observada de pizza com guaraná.",
    "H8": "a receita congelada ordena os contextos de 2017-2018 melhor que a tabela de taxas.",
    "H9": (
        "treinar com 2008-2009 somado não piora a previsão de 2017-2018 em relação a treinar só "
        "com 2017-2018."
    ),
    "H10": (
        "por UF, a participação do guaraná no interesse de busca (guaraná contra cola) tem "
        "correlação positiva com a participação do guaraná nas ocasiões com refrigerante da POF, "
        "no período de cada edição."
    ),
}

_TESTES: dict[str, str] = {
    "H1": (
        "degrau B contra degrau A, os dois com hiperparâmetros escolhidos nas dobras internas, "
        "comparados nas previsões fora da dobra externa de 2008-2009."
    ),
    "H2": "degrau C (com `C` e o conjunto de interações escolhidos nas dobras internas) contra degrau B.",
    "H3": "degrau D contra o melhor modelo logístico (C, se H2 for confirmada, senão B).",
    "H4": (
        "coeficiente de `tem_pizza` no degrau B da preferência, ajustado em 2008-2009 inteira, "
        "com intervalo de 1000 réplicas do bootstrap de setores."
    ),
    "H5": (
        "degrau B de cada parte, com e sem peso no treino, na validação aninhada. A métrica é "
        "sempre ponderada."
    ),
    "H6": (
        "o produto fora da dobra contra o modelo direto, os dois contra `y_pizza_guarana` nas "
        "células."
    ),
    "H7": (
        "sigmoide ajustada nas previsões internas de cada dobra externa e aplicada às previsões "
        "fora dela; avaliação só fora da dobra externa."
    ),
    "H8": (
        "avaliação temporal: receita congelada treinada em 2008-2009 e testada uma única vez em "
        "2017-2018 (depois da tag `receita-congelada`), com o score contra `y_pizza_guarana` nas "
        "células, comparado ao produto das duas tabelas do degrau A."
    ),
    "H9": (
        "validação cruzada dentro de 2017-2018 com a receita congelada, em dois braços: recente "
        "(só 2017-2018) e base inteira (mais toda 2008-2009, sempre no treino)."
    ),
    "H10": "correlação de postos (Spearman) entre as 27 UFs, uma vez por edição.",
}

_METRICAS: dict[str, str] = {
    "H1": "ganho de B sobre A no log loss ponderado, com intervalo de 95%.",
    "H2": "ganho de C sobre B no log loss ponderado, com intervalo de 95%.",
    "H3": "ganho de D sobre o melhor logístico no log loss ponderado, com intervalo de 95%.",
    "H4": "o coeficiente, em log-chances, com intervalo de 95% (encolhido pela regularização).",
    "H5": "ganho da versão sem peso no log loss ponderado, com intervalo de 95%.",
    "H6": "ganho do direto sobre o produto no log loss ponderado, com intervalo de 95%.",
    "H7": "ganho da versão corrigida no log loss ponderado, com intervalo de 95%.",
    "H8": "ganho da receita sobre a referência na PR-AUC ponderada, com intervalo de 95%.",
    "H9": "ganho do braço recente sobre a base inteira no log loss ponderado, com intervalo de 95%.",
    "H10": "correlação de postos de cada edição, com intervalo de 95% por reamostragem de UFs.",
}

MOTIVO_H10 = (
    "o bloco do Google Trends foi cortado pelo autor em 20/11/2026, antes de qualquer cálculo com "
    "a POF: a comparação por estado tem resolução de porcentagem inteira (não ordena os estados), "
    "os tópicos misturam buscas que não são sobre a bebida e 15 dos 27 estados vêm sem valor para "
    "o guaraná em 2008-2009. Nenhum número do interesse de busca entra no score, na recomendação "
    "ou nos resultados."
)


def _ler(pasta: Path, relativo: str) -> dict[str, Any] | None:
    caminho = pasta / relativo
    if not caminho.exists():
        return None
    return json.loads(caminho.read_text(encoding="utf-8"))


def _chave(dados: dict[str, Any], fonte: str, *caminho: str) -> Any:
    """Lê `dados["metricas"][...caminho]`; chave ausente vira erro que diz o arquivo e a chave."""
    atual: Any = dados
    for nome in ("metricas", *caminho):
        if not isinstance(atual, dict) or nome not in atual:
            raise KeyError(f"{fonte}: chave ausente `{'.'.join(('metricas', *caminho))}` ({nome})")
        atual = atual[nome]
    return atual


def _entrada(
    id_: str,
    parte: str | None,
    fonte: str,
    intervalo: dict[str, Any] | None,
    veredito: str,
    consequencia: str,
    **extras: Any,
) -> dict[str, Any]:
    estimativa = ic_inf = ic_sup = largura = None
    if intervalo is not None:
        estimativa = intervalo.get("estimativa")
        ic_inf = intervalo.get("ic_inf")
        ic_sup = intervalo.get("ic_sup")
        if ic_inf is not None and ic_sup is not None:
            largura = ic_sup - ic_inf
    entrada: dict[str, Any] = {
        "id": id_,
        "parte": parte,
        "enunciado": _ENUNCIADOS[id_],
        "teste": _TESTES[id_],
        "metrica": _METRICAS[id_],
        "estimativa": estimativa,
        "ic_inf": ic_inf,
        "ic_sup": ic_sup,
        "largura": largura,
        "veredito": veredito,
        "consequencia": consequencia,
        "fonte": fonte,
    }
    entrada.update(extras)
    return entrada


def _ausente(id_: str, parte: str | None, fonte: str, motivo: str) -> dict[str, Any]:
    return _entrada(
        id_,
        parte,
        fonte,
        None,
        NAO_EXECUTADA,
        "nenhuma: a hipótese não foi testada.",
        motivo=motivo,
    )


def _sem_arquivo(id_: str, parte: str | None, fonte: str) -> dict[str, Any]:
    return _ausente(id_, parte, fonte, f"o arquivo {fonte} não existe (a etapa não rodou).")


def _consequencia_escada(id_: str, parte: str, veredito: str, vencedor: str | None) -> str:
    confirmada = veredito == "confirmada"
    final = f" Degrau vencedor da escada na parte {parte}: {vencedor}." if vencedor else ""
    if id_ == "H1":
        texto = (
            f"a logística (B) supera a tabela de taxas na parte {parte}."
            if confirmada
            else f"a logística (B) não supera a tabela de taxas na parte {parte}: o ML simples "
            "não acrescentou."
        )
    elif id_ == "H2":
        texto = (
            f"curvas e interações (C) superam a logística simples na parte {parte}."
            if confirmada
            else f"curvas e interações (C) não superam a logística simples na parte {parte}."
        )
    else:
        texto = (
            f"o LightGBM (D) supera o melhor logístico na parte {parte}."
            if confirmada
            else f"o LightGBM (D) não supera o melhor logístico na parte {parte}."
        )
    return texto + final


def _escada(pasta: Path) -> list[dict[str, Any]]:
    saida: list[dict[str, Any]] = []
    for id_ in ("H1", "H2", "H3"):
        for parte in PARTES:
            fonte = f"experimentos/escada_{parte}.json"
            dados = _ler(pasta, fonte)
            if dados is None:
                saida.append(_sem_arquivo(id_, parte, fonte))
                continue
            comp = dados["metricas"]["comparacoes"][id_]
            saida.append(
                _entrada(
                    id_,
                    parte,
                    fonte,
                    comp["intervalo"],
                    comp["veredito"],
                    _consequencia_escada(
                        id_, parte, comp["veredito"], dados["metricas"].get("vencedor")
                    ),
                    candidato=comp.get("candidato"),
                    referencia=comp.get("referencia"),
                    vencedor=dados["metricas"].get("vencedor"),
                )
            )
    return saida


def _h4(pasta: Path) -> dict[str, Any]:
    fonte = "experimentos/h4_tem_pizza.json"
    dados = _ler(pasta, fonte)
    if dados is None:
        return _sem_arquivo("H4", "preferencia", fonte)
    return _entrada(
        "H4",
        "preferencia",
        fonte,
        _chave(dados, fonte, "intervalo"),
        _chave(dados, fonte, "veredito"),
        "nenhuma: a variável `tem_pizza` permanece no modelo em todos os casos. "
        "Mede associação, não efeito causal da pizza sobre a escolha.",
    )


def _h5(pasta: Path) -> list[dict[str, Any]]:
    fonte = "experimentos/h5_peso.json"
    dados = _ler(pasta, fonte)
    saida: list[dict[str, Any]] = []
    for parte in PARTES:
        if dados is None:
            saida.append(_sem_arquivo("H5", parte, fonte))
            continue
        m = dados["metricas"][parte]
        saida.append(
            _entrada(
                "H5",
                parte,
                fonte,
                m["ganho_sem_peso"],
                m["veredito"],
                f"treino {'com' if m['usar_peso'] else 'sem'} peso na parte {parte} "
                "(a calibração continua usando `peso_final`).",
                usar_peso=m["usar_peso"],
            )
        )
    return saida


def _h6_h7(pasta: Path) -> list[dict[str, Any]]:
    fonte = "experimentos/h6_h7_produto.json"
    dados = _ler(pasta, fonte)
    if dados is None:
        return [_sem_arquivo("H6", None, fonte), _sem_arquivo("H7", None, fonte)]
    m = dados["metricas"]
    h6, h7 = m["H6"], m["H7"]
    conseq6 = (
        "nenhuma: o produto das duas partes continua calculando o score."
        if h6["veredito"] == "sem evidência contra o padrão"
        else "o modelo direto assume o cálculo do score; as duas partes ficam como componentes."
    )
    conseq7 = (
        "a sigmoide de calibração do produto entra na receita."
        if h7.get("calibrar_score")
        else "nenhuma: a sigmoide de correção não entra na receita e o produto fica como está; "
        "isso não é um achado de descalibração."
    )
    return [
        _entrada(
            "H6",
            None,
            fonte,
            h6["ganho_do_direto"],
            h6["veredito"],
            conseq6,
            graus_de_liberdade_nominais_do_direto=h6.get("graus_de_liberdade_nominais_do_direto"),
            limite_riley_do_direto=h6.get("limite_riley_do_direto"),
        ),
        _entrada(
            "H7",
            None,
            fonte,
            h7["ganho_corrigida"],
            h7["veredito"],
            conseq7,
            inclinacao=h7.get("inclinacao"),
            leitura=_leitura_h7(h7),
        ),
    ]


def _numero(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def _leitura_h7(h7: dict[str, Any]) -> str:
    """Como ler o veredito de H7: a regra testa a correção, não a calibração do produto."""
    base = (
        "a regra testa se a correção sigmoide melhora o log loss do produto; 'refutada' quer "
        "dizer que a correção não entrou, e não que o produto esteja descalibrado"
    )
    inc = h7.get("inclinacao")
    if not inc or inc.get("ic_inf") is None or inc.get("ic_sup") is None:
        return f"{base}."
    return (
        f"{base}. A inclinação da calibração, {_numero(inc['estimativa'])} "
        f"[{_numero(inc['ic_inf'])}; {_numero(inc['ic_sup'])}], não mostra descalibração."
    )


def _consequencia_h8(veredito: str) -> str:
    base = "nenhuma: é a avaliação final e a receita não muda depois dela."
    if veredito == "refutada":
        return (
            f"{base} A receita não superou a tabela de taxas na PR-AUC de 2017-2018; a "
            "limitação vai para a tela e para o README."
        )
    if veredito == "confirmada":
        return f"{base} A receita superou a tabela de taxas na PR-AUC de 2017-2018."
    return f"{base} Veredito: {veredito}."


def _h8(pasta: Path) -> dict[str, Any]:
    fonte = "teste_final.json"
    dados = _ler(pasta, fonte)
    if dados is None:
        return _sem_arquivo("H8", None, fonte)
    veredito = _chave(dados, fonte, "H8", "veredito")
    return _entrada(
        "H8",
        None,
        fonte,
        _chave(dados, fonte, "H8", "ganho"),
        veredito,
        _consequencia_h8(veredito),
        pr_auc=_chave(dados, fonte, "H8", "pr_auc"),
        referencia=_chave(dados, fonte, "H8", "referencia"),
    )


def _h9(pasta: Path) -> list[dict[str, Any]]:
    fonte = "pos_teste.json"
    dados = _ler(pasta, fonte)
    nomes = ("ocasiao", "preferencia", "preferencia_com_pizza")
    if dados is None:
        return [_sem_arquivo("H9", n, fonte) for n in nomes]
    servido = _chave(dados, fonte, "H9", "treino_servido")
    conseq = (
        "o modelo servido é treinado com as duas edições (a agregação das três comparações "
        "mantém a base inteira)."
        if servido == "base_inteira"
        else "o modelo servido é treinado só com 2017-2018 (alguma comparação recusou a base "
        "inteira)."
    )
    return [
        _entrada(
            "H9",
            nome,
            fonte,
            _chave(dados, fonte, "H9", "comparacoes", nome, "ganho"),
            _chave(dados, fonte, "H9", "comparacoes", nome, "veredito"),
            conseq,
            treino_servido=servido,
        )
        for nome in nomes
    ]


def _h10() -> dict[str, Any]:
    return _ausente("H10", None, "docs/registro-de-decisoes.md", MOTIVO_H10)


def consolidar(pasta: Path = RELATORIOS) -> dict[str, Any]:
    """Lê os JSONs das etapas em `pasta` e devolve as entradas de H1 a H10 (vereditos copiados)."""
    hipoteses: list[dict[str, Any]] = [
        *_escada(pasta),
        _h4(pasta),
        *_h5(pasta),
        *_h6_h7(pasta),
        _h8(pasta),
        *_h9(pasta),
        _h10(),
    ]
    return {"hipoteses": hipoteses}


def rodar_hipoteses() -> Path:
    """Grava `reports/hipoteses.json` a partir dos JSONs de `reports/`."""
    resultado = consolidar(RELATORIOS)
    caminho = RELATORIOS / "hipoteses.json"
    caminho.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return caminho
