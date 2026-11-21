"""Teste final (H8, seção 7 do pré-registro): a receita congelada medida em 2017-2018.

`rodar_teste_final` só lê o alvo de 2017-2018 depois de conferir, nesta ordem:

1. a tag `receita-congelada` existe e a assinatura da receita na árvore de trabalho é a da tag
   (`exigir_receita_intacta`);
2. o artefato é `models/avaliado/artefato.joblib` (não o servido nem o baseline), montado com
   essa assinatura e com os degraus de `reports/receita.json`;
3. os relatórios de validação de 2008-2009 existem, terminaram bem e são os da receita.

Nada aqui muda a receita nem o artefato: o degrau A de H8 é ajustado de novo em 2008-2009, em
memória, com a configuração de `receita.referencia_a` e o uso de peso decidido em H5.

**Posições.** As posições são sempre em relação à população de referência de 2008-2009: as do
score e da ocasião da receita vêm da referência guardada no artefato (a mesma regra de
`pontuar`); as do degrau A, de uma referência equivalente construída nas células de 2008-2009
com as tabelas A ajustadas aqui. A PR-AUC não depende das posições; a concentração no décimo
superior e o corte da classe alta dependem.

Grava `reports/teste_final.json`. Se ele já existir, grava `reports/teste_final_reexecucao.json`
com `identico_ao_original` (tudo menos `data_hora` e `commit`), sem tocar o original (seção 6.7
do design: rodar de novo é permitido e deve reproduzir).
"""

import json
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from occasion.avaliacao.erros import contextos_com_mais_erro
from occasion.avaliacao.segmentos import colunas_de_segmento, metricas_por_segmento
from occasion.bases.acesso import carregar_base
from occasion.bases.assinatura import TesteBloqueado, exigir_receita_intacta
from occasion.config import EDICAO_TESTE, EDICAO_TREINO, RAIZ, SEMENTE
from occasion.experimentos.especificacao import ParteModelo
from occasion.experimentos.registro import Natureza, _para_json
from occasion.experimentos.regras import superioridade
from occasion.inference.predict import NOME_ARQUIVO, ArtefatoModelo
from occasion.inference.treino import ajustar_parte, descricao_do_degrau
from occasion.models.base import ModeloParte
from occasion.pipeline import commit_da_receita
from occasion.receita import Receita, ReceitaParte, carregar_receita
from occasion.score.score import Parte, ReferenciaScore, construir_referencia, probabilidades
from occasion.validacao.bootstrap import B_PADRAO, Intervalo, intervalo_ganho
from occasion.validacao.metricas import concentracao_topo, pr_auc_ponderada, resumo

__all__ = ["NOME", "posicoes_na_referencia", "rodar_teste_final"]

NOME = "teste_final"
NATUREZA: Natureza = "confirmatório"
ALVO_SCORE = "y_pizza_guarana"
ALVO_OCASIAO = "y_pizza_refri"
ALVO_PREFERENCIA = "y_guarana"
CAMPOS_IGNORADOS_NA_REEXECUCAO: frozenset[str] = frozenset({"data_hora", "commit"})
_METRICAS_DA_QUEDA = ("log_loss", "brier", "pr_auc", "roc_auc", "prevalencia", "concentracao_topo")
_METRICAS_DO_SEGMENTO = ("n", "positivos", "log_loss", "pr_auc", "o_e")
_DECLARACAO_QUEDA = (
    "a queda em relação à validação é o efeito conjunto de dez anos e da mudança do método de "
    "coleta (registro alimentar em 2008-2009, recordatório de 24 horas em 2017-2018); o projeto "
    "não separa os dois"
)


# ---------------------------------------------------------------------------------- utilitários


def posicoes_na_referencia(ref: ReferenciaScore, valores: np.ndarray, qual: Parte) -> np.ndarray:
    """Posição de 0 a 100 de cada valor na referência: a regra de `ReferenciaScore.posicao`, em lote."""
    serie: np.ndarray = getattr(ref, qual)
    acumulado = ref.peso_acumulado[qual]
    indice = np.searchsorted(serie, np.asarray(valores, dtype=float), side="right")
    return np.where(indice == 0, 0.0, acumulado[np.maximum(indice - 1, 0)] * 100.0)


def _sem_nan(p: np.ndarray, o_que: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError(f"previsão com NaN ou infinito em {o_que}")
    return p


def _prevalencia(base: pd.DataFrame, alvo: str) -> float:
    w = base["peso_final"].to_numpy(dtype=float)
    return float(np.sum(w * base[alvo].to_numpy(dtype=float)) / np.sum(w))


def _dados(base: pd.DataFrame, alvo: str, edicao: str) -> dict[str, Any]:
    return {
        "edicao": edicao,
        "n": len(base),
        "positivos": int(np.sum(base[alvo].to_numpy() == 1)),
        "setores": len(np.unique(base["cod_upa"].to_numpy())),
    }


def _ler_metricas(caminho: Path, raiz: Path) -> dict[str, Any]:
    relativo = caminho.relative_to(raiz)
    if not caminho.exists():
        raise ValueError(f"falta {relativo}: a validação de 2008-2009 precisa estar completa")
    metricas = json.loads(caminho.read_text(encoding="utf-8"))["metricas"]
    if metricas.get("status") != "ok":
        raise ValueError(f"{relativo} é de uma execução que falhou")
    return metricas


# ---------------------------------------------------------------------------------- conferências


def _carregar_avaliado(raiz: Path, receita: Receita, assinatura: str) -> ArtefatoModelo:
    """O artefato avaliado, recusado se não for o da receita congelada."""
    caminho = raiz / "models" / "avaliado" / NOME_ARQUIVO
    if not caminho.exists():
        raise TesteBloqueado(
            f"falta {caminho.relative_to(raiz)}: rode `make avaliado` com a árvore limpa, "
            "depois da tag"
        )
    # gerado pelo próprio pipeline (`make avaliado`) nesta máquina; nunca vem de fora
    artefato = joblib.load(caminho)
    if not isinstance(artefato, ArtefatoModelo):
        raise TesteBloqueado(f"{caminho.relative_to(raiz)} não é um ArtefatoModelo")
    if artefato.assinatura_receita != assinatura:
        raise TesteBloqueado(
            f"a assinatura do artefato ({artefato.assinatura_receita}) não é a da receita "
            f"congelada ({assinatura}): rode `make avaliado` de novo, com a árvore limpa"
        )
    esperado = descricao_do_degrau(receita)
    if artefato.metadados.degrau != esperado:
        raise TesteBloqueado(
            f"o degrau do artefato ({artefato.metadados.degrau!r}) não é o da receita "
            f"({esperado!r}): o artefato não é o avaliado"
        )
    if artefato.metadados.modelo != "avaliado" or artefato.metadados.treinado_em != [EDICAO_TREINO]:
        raise TesteBloqueado(
            f"o artefato não é o avaliado treinado só em {EDICAO_TREINO}: "
            f"{artefato.metadados.modelo}, {artefato.metadados.treinado_em}"
        )
    return artefato


def _relatorios_de_validacao(raiz: Path, receita: Receita) -> dict[str, dict[str, Any]]:
    """As métricas de 2008-2009 lado a lado com o teste; recusa relatórios de outra receita."""
    pasta = raiz / "reports" / "experimentos"
    rel = {
        nome: _ler_metricas(pasta / f"{nome}.json", raiz)
        for nome in ("escada_ocasiao", "escada_preferencia", "h6_h7_produto", "cortes")
    }
    rel["avaliacao_2008"] = _ler_metricas(raiz / "reports" / "avaliacao_2008.json", raiz)
    for parte, rp in (("ocasiao", receita.ocasiao), ("preferencia", receita.preferencia)):
        if rel[f"escada_{parte}"]["vencedor"] != rp.degrau:
            raise ValueError(f"escada_{parte}.json não é da receita (vencedor diferente)")
    prod = rel["h6_h7_produto"]
    if prod["score"] != receita.score or rel["cortes"]["score"] != receita.score:
        raise ValueError("h6_h7_produto.json ou cortes.json não são do score da receita")
    if bool(prod["H7"]["calibrar_score"]) != receita.calibrar_score:
        raise ValueError("h6_h7_produto.json discorda da receita na calibração do score")
    return rel


# ---------------------------------------------------------------------------------- comparações


def _comparar(
    base: pd.DataFrame,
    alvo: str,
    candidato: np.ndarray,
    referencia: np.ndarray,
    nome_referencia: str,
    b: int,
) -> dict[str, Any]:
    """Ganho da receita sobre a referência na PR-AUC ponderada, com setores dentro da UF."""
    y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy(dtype=float)
    ic: Intervalo = intervalo_ganho(
        pr_auc_ponderada, True, y, candidato, referencia, w,
        base["cod_upa"].to_numpy(), base["uf"].to_numpy(), b=b, semente=SEMENTE,
    )  # fmt: skip
    return {
        "referencia": nome_referencia,
        "pr_auc": {
            "receita": pr_auc_ponderada(y, candidato, w),
            "referencia": pr_auc_ponderada(y, referencia, w),
        },
        "ganho": ic,
        "veredito": superioridade(ic),
    }


def _ajustar_a(
    base: pd.DataFrame, alvo: str, parte: ParteModelo, receita: Receita, rp: ReceitaParte
) -> ModeloParte:
    """Tabela do degrau A da parte, com a configuração de `referencia_a` e o peso de H5."""
    config = ReceitaParte(
        degrau="A", parametros=receita.referencia_a[parte], usar_peso=rp.usar_peso, calibrar=False
    )
    return ajustar_parte(base, alvo, parte, config, np.empty(0))


def _diferenca(teste: Any, validacao: Any) -> float | None:
    """`teste - validacao`, ou `None` se um dos lados não existe."""
    if teste is None or validacao is None:
        return None
    return float(teste) - float(validacao)


def _lado_a_lado(
    validacao: dict[str, Any] | None, teste: dict[str, Any], fonte: str | None, motivo: str = ""
) -> dict[str, Any]:
    t = {k: teste.get(k) for k in _METRICAS_DA_QUEDA}
    if validacao is None:
        return {"fonte": None, "validacao": None, "teste": t, "diferenca": None, "motivo": motivo}
    v = {k: validacao.get(k) for k in _METRICAS_DA_QUEDA}
    diferenca = {k: _diferenca(t[k], v[k]) for k in _METRICAS_DA_QUEDA}
    return {"fonte": fonte, "validacao": v, "teste": t, "diferenca": diferenca}


def _queda_por_segmento(
    validacao: list[dict[str, Any]], teste: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Cada segmento do teste ao lado do mesmo segmento na validação (`None` se ele não existia)."""
    por_chave = {(str(v["segmento"]), str(v["valor"])): v for v in validacao}
    linhas: list[dict[str, Any]] = []
    for t in teste:
        v = por_chave.get((str(t["segmento"]), str(t["valor"])))
        lado_v = None if v is None else {k: v.get(k) for k in _METRICAS_DO_SEGMENTO}
        lado_t = {k: t.get(k) for k in _METRICAS_DO_SEGMENTO}
        diferenca = {
            k: None if lado_v is None else _diferenca(lado_t[k], lado_v[k])
            for k in ("log_loss", "pr_auc", "o_e")
        }
        linhas.append(
            {
                "segmento": t["segmento"],
                "valor": t["valor"],
                "validacao": lado_v,
                "teste": lado_t,
                "diferenca": diferenca,
            }
        )
    return linhas


# ---------------------------------------------------------------------------------- gravação


def _versoes() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "pandas": version("pandas"),
        "scikit-learn": version("scikit-learn"),
        "lightgbm": version("lightgbm"),
    }


def _gravar(raiz: Path, registro: dict[str, Any]) -> Path:
    """Grava o original; se ele existir, grava a reexecução com a comparação, sem sobrescrevê-lo."""
    pasta = raiz / "reports"
    pasta.mkdir(parents=True, exist_ok=True)
    conteudo: dict[str, Any] = json.loads(json.dumps(_para_json(registro)))
    destino = pasta / f"{NOME}.json"
    if destino.exists():
        original: dict[str, Any] = json.loads(destino.read_text(encoding="utf-8"))
        chaves = (set(original) | set(conteudo)) - CAMPOS_IGNORADOS_NA_REEXECUCAO
        diferentes = sorted(k for k in chaves if original.get(k) != conteudo.get(k))
        conteudo["identico_ao_original"] = not diferentes
        conteudo["chaves_diferentes"] = diferentes
        destino = pasta / f"{NOME}_reexecucao.json"
    destino.write_text(
        json.dumps(conteudo, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    return destino


# ---------------------------------------------------------------------------------- teste final


def rodar_teste_final(raiz: Path = RAIZ, *, b: int = B_PADRAO) -> Path:
    """Roda o teste final (H8 e o relatado à parte) e devolve o caminho do relatório gravado.

    Todos os caminhos saem de `raiz`. `b` é o número de réplicas do bootstrap (1000 no teste
    real; menor só nos testes do código).
    """
    assinatura = exigir_receita_intacta(raiz)
    receita = carregar_receita(raiz / "reports" / "receita.json")
    artefato = _carregar_avaliado(raiz, receita, assinatura)
    validacao = _relatorios_de_validacao(raiz, receita)
    print("teste final: receita, artefato e validação conferidos", flush=True)

    pasta = raiz / "data" / "processed"
    cel08 = carregar_base("celulas", EDICAO_TREINO, pasta=pasta, raiz=raiz)
    oc08 = carregar_base("ocasioes_refri", EDICAO_TREINO, pasta=pasta, raiz=raiz)
    cel17 = carregar_base("celulas", EDICAO_TESTE, pasta=pasta, raiz=raiz)
    oc17 = carregar_base("ocasioes_refri", EDICAO_TESTE, pasta=pasta, raiz=raiz)
    com_pizza = (oc17["tem_pizza"] == 1).to_numpy()
    oc17_pizza = cast(pd.DataFrame, oc17[com_pizza])

    # degrau A (referência de H8), ajustado em 2008-2009 aqui mesmo, e as constantes
    a_oc = _ajustar_a(cel08, ALVO_OCASIAO, "ocasiao", receita, receita.ocasiao)
    a_pref = _ajustar_a(oc08, ALVO_PREFERENCIA, "preferencia", receita, receita.preferencia)
    ref_a = construir_referencia(cel08, a_oc, a_pref)
    constante = {
        "score": _prevalencia(cel08, ALVO_SCORE),
        "ocasiao": _prevalencia(cel08, ALVO_OCASIAO),
        "preferencia": _prevalencia(oc08, ALVO_PREFERENCIA),
    }

    # previsões em 2017-2018
    print("teste final: previsões em 2017-2018", flush=True)
    p_rec = probabilidades(
        cel17, artefato.modelo_p1, artefato.modelo_p2,
        artefato.calibracao_score, artefato.modelo_direto,
    )  # fmt: skip
    p_a = probabilidades(cel17, a_oc, a_pref)
    previsto: dict[str, dict[str, np.ndarray]] = {
        "score": {
            "receita": _sem_nan(p_rec["p_produto"].to_numpy(), "score"),
            "degrau_a": _sem_nan(p_a["p_produto"].to_numpy(), "score do degrau A"),
        },
        "ocasiao": {
            "receita": _sem_nan(p_rec["p_ocasiao"].to_numpy(), "ocasião"),
            "degrau_a": _sem_nan(p_a["p_ocasiao"].to_numpy(), "ocasião do degrau A"),
        },
    }
    pref_rec = _sem_nan(artefato.modelo_p2.predict_proba(oc17), "preferência")
    pref_a = _sem_nan(a_pref.predict_proba(oc17), "preferência do degrau A")
    previsto["preferencia_todas"] = {"receita": pref_rec, "degrau_a": pref_a}
    previsto["preferencia_com_pizza"] = {
        "receita": pref_rec[com_pizza],
        "degrau_a": pref_a[com_pizza],
    }
    posicao = {
        "score": {
            "receita": posicoes_na_referencia(
                artefato.referencia, previsto["score"]["receita"], "produto"
            ),
            "degrau_a": posicoes_na_referencia(ref_a, previsto["score"]["degrau_a"], "produto"),
        },
        "ocasiao": {
            "receita": posicoes_na_referencia(
                artefato.referencia, previsto["ocasiao"]["receita"], "ocasiao"
            ),
            "degrau_a": posicoes_na_referencia(ref_a, previsto["ocasiao"]["degrau_a"], "ocasiao"),
        },
    }
    avaliacoes: dict[str, tuple[pd.DataFrame, str, float, str]] = {
        "score": (cel17, ALVO_SCORE, constante["score"], "score"),
        "ocasiao": (cel17, ALVO_OCASIAO, constante["ocasiao"], "ocasiao"),
        "preferencia_todas": (oc17, ALVO_PREFERENCIA, constante["preferencia"], "preferencia"),
        "preferencia_com_pizza": (
            oc17_pizza, ALVO_PREFERENCIA, constante["preferencia"], "preferencia",
        ),
    }  # fmt: skip

    # H8: o score contra o produto das tabelas A (ou contra a constante, se a receita for a tabela)
    so_tabelas = (
        receita.score == "produto"
        and receita.ocasiao.degrau == "A"
        and receita.preferencia.degrau == "A"
    )
    print(f"teste final: H8 com {b} réplicas", flush=True)
    if so_tabelas:
        ref_h8, nome_h8 = np.full(len(cel17), constante["score"]), "constante"
    else:
        ref_h8, nome_h8 = previsto["score"]["degrau_a"], "produto das tabelas do degrau A"
    h8 = _comparar(cel17, ALVO_SCORE, previsto["score"]["receita"], ref_h8, nome_h8, b)

    # relatado à parte: a mesma comparação em cada parte (a parte que já é a tabela A é
    # comparada com a constante, a mesma regra do score)
    partes: dict[str, Any] = {}
    degrau_da_parte = {
        "ocasiao": receita.ocasiao.degrau,
        "preferencia_todas": receita.preferencia.degrau,
        "preferencia_com_pizza": receita.preferencia.degrau,
    }
    for nome, degrau in degrau_da_parte.items():
        print(f"teste final: comparação da parte {nome}", flush=True)
        base, alvo, const, _ = avaliacoes[nome]
        if degrau == "A":
            ref, nome_ref = np.full(len(base), const), "constante"
        else:
            ref, nome_ref = previsto[nome]["degrau_a"], "tabela do degrau A"
        partes[nome] = _comparar(base, alvo, previsto[nome]["receita"], ref, nome_ref, b)

    # resumos, concentração no décimo superior, segmentos e erros
    resumos: dict[str, Any] = {}
    concentracao: dict[str, Any] = {}
    segmentos: dict[str, Any] = {}
    erros: dict[str, Any] = {}
    for nome, (base, alvo, const, _) in avaliacoes.items():
        y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy(dtype=float)
        p_const = np.full(len(base), const)
        pos = posicao.get(nome, {})
        resumos[nome] = {
            "receita": resumo(y, previsto[nome]["receita"], w, posicao=pos.get("receita")),
            "degrau_a": resumo(y, previsto[nome]["degrau_a"], w, posicao=pos.get("degrau_a")),
            "constante": resumo(y, p_const, w),
        }
        if pos:
            concentracao[nome] = {
                "receita": concentracao_topo(y, pos["receita"], w),
                "degrau_a": concentracao_topo(y, pos["degrau_a"], w),
                "constante": concentracao_topo(y, p_const, w),
            }
        segmentos[nome] = metricas_por_segmento(
            y, previsto[nome]["receita"], w, colunas_de_segmento(base)
        )
        erros[nome] = contextos_com_mais_erro(base, y, previsto[nome]["receita"])

    # a queda em relação à validação de 2008-2009
    av = validacao["avaliacao_2008"]
    esc_oc, esc_pref = validacao["escada_ocasiao"], validacao["escada_preferencia"]
    sem_a = "o produto das tabelas do degrau A não foi medido fora da dobra em 2008-2009"
    queda: dict[str, Any] = {
        "declaracao": _DECLARACAO_QUEDA,
        "score": {
            "receita": _lado_a_lado(
                validacao["cortes"]["resumo"], resumos["score"]["receita"], "cortes.json"
            ),
            "degrau_a": _lado_a_lado(None, resumos["score"]["degrau_a"], None, sem_a),
            "validacao_h6_h7": {
                "H6": {k: validacao["h6_h7_produto"]["H6"][k] for k in ("log_loss", "pr_auc", "roc_auc")},
                "H7": {
                    k: validacao["h6_h7_produto"]["H7"][k]
                    for k in ("log_loss_antes", "log_loss_depois")
                },
            },
        },
        "ocasiao": {
            "receita": _lado_a_lado(
                av["ocasiao"]["resumo"], resumos["ocasiao"]["receita"], "avaliacao_2008.json"
            ),
            "degrau_a": _lado_a_lado(
                esc_oc["degraus"]["A"]["resumo"], resumos["ocasiao"]["degrau_a"],
                "escada_ocasiao.json",
            ),
        },
        "preferencia_todas": {
            "receita": _lado_a_lado(
                av["preferencia"]["todas"]["resumo"], resumos["preferencia_todas"]["receita"],
                "avaliacao_2008.json",
            ),
            "degrau_a": _lado_a_lado(
                esc_pref["degraus"]["A"]["resumo"], resumos["preferencia_todas"]["degrau_a"],
                "escada_preferencia.json",
            ),
        },
        "preferencia_com_pizza": {
            "receita": _lado_a_lado(
                av["preferencia"]["com_pizza"]["resumo"],
                resumos["preferencia_com_pizza"]["receita"], "avaliacao_2008.json",
            ),
            "degrau_a": _lado_a_lado(
                esc_pref["degraus"]["A"]["resumo_com_pizza"],
                resumos["preferencia_com_pizza"]["degrau_a"], "escada_preferencia.json",
            ),
        },
    }  # fmt: skip
    segmentos_da_validacao = {
        "score": av["score"]["segmentos"],
        "ocasiao": av["ocasiao"]["segmentos"],
        "preferencia_todas": av["preferencia"]["todas"]["segmentos"],
        "preferencia_com_pizza": av["preferencia"]["com_pizza"]["segmentos"],
    }
    queda_por_segmento = {
        nome: _queda_por_segmento(segmentos_da_validacao[nome], segmentos[nome])
        for nome in segmentos
    }

    configuracao: dict[str, Any] = {
        "hipotese": "H8: a receita congelada ordena os contextos de 2017-2018 melhor que a "
        "tabela de taxas",
        "metrica": "ganho da receita sobre a referência na PR-AUC ponderada (peso_final) contra "
        f"{ALVO_SCORE} nas células de {EDICAO_TESTE}",
        "regra": "superioridade: o intervalo do ganho precisa excluir zero a favor da receita",
        "referencia": nome_h8,
        "degrau_a": "construir('A', parte, receita.referencia_a[parte]) ajustado em "
        f"{EDICAO_TREINO}, com o usar_peso de H5 (receita.<parte>.usar_peso), sem sigmoide",
        "constante": constante,
        "constante_declaracao": "prevalência ponderada de 2008-2009 de cada alvo; no recorte de "
        "pizza da preferência, a de todas as ocasiões com refrigerante (a base em que a parte "
        "é treinada)",
        "partes": "relatadas à parte, sem entrar na regra; a parte que já é a tabela A é "
        "comparada com a constante",
        "posicoes": "em relação à referência de 2008-2009: a do artefato para a receita; para o "
        "degrau A, uma referência equivalente nas células de 2008-2009 com as tabelas A; a "
        "preferência nas ocasiões com refrigerante não tem posição (a referência é de células)",
        "bootstrap": {"b": b, "unidade": "cod_upa", "estrato": "uf", "nivel": 0.95},
        "receita": {
            "ocasiao": receita.ocasiao.model_dump(),
            "preferencia": receita.preferencia.model_dump(),
            "direto": None if receita.direto is None else receita.direto.model_dump(),
            "score": receita.score,
            "calibrar_score": receita.calibrar_score,
            "referencia_a": receita.referencia_a,
        },
        "artefato": {
            "degrau": artefato.metadados.degrau,
            "receita_commit": artefato.metadados.receita_commit,
            "calibracao_score": artefato.calibracao_score,
        },
        "queda": _DECLARACAO_QUEDA,
    }
    dados: dict[str, Any] = {
        "celulas_teste": _dados(cel17, ALVO_SCORE, EDICAO_TESTE),
        "ocasioes_refri_teste": _dados(oc17, ALVO_PREFERENCIA, EDICAO_TESTE),
        "ocasioes_refri_teste_com_pizza": _dados(oc17_pizza, ALVO_PREFERENCIA, EDICAO_TESTE),
        "celulas_treino": _dados(cel08, ALVO_SCORE, EDICAO_TREINO),
        "ocasioes_refri_treino": _dados(oc08, ALVO_PREFERENCIA, EDICAO_TREINO),
        "receita": "reports/receita.json",
        "artefato": f"models/avaliado/{NOME_ARQUIVO}",
        "validacao": [
            "reports/experimentos/escada_ocasiao.json",
            "reports/experimentos/escada_preferencia.json",
            "reports/experimentos/h6_h7_produto.json",
            "reports/experimentos/cortes.json",
            "reports/avaliacao_2008.json",
        ],
    }
    registro: dict[str, Any] = {
        "nome": NOME,
        "natureza": NATUREZA,
        "configuracao": configuracao,
        "metricas": {
            "H8": h8,
            "partes": partes,
            "concentracao_topo": concentracao,
            "resumos": resumos,
            "segmentos": segmentos,
            "erros": erros,
            "queda": queda,
            "queda_por_segmento": queda_por_segmento,
        },
        "dados": dados,
        "assinatura_receita": assinatura,
        "commit": commit_da_receita(raiz),
        "semente": SEMENTE,
        "data_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versoes": _versoes(),
    }
    destino = _gravar(raiz, registro)
    print(f"teste final: H8 {h8['veredito']}; gravado em {destino}", flush=True)
    return destino
