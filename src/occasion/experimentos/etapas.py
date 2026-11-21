"""Etapas dos experimentos do Plano 2 sobre 2008-2009: dobras, peso (H5), escada, H4, controle,
calibração, produto (H6 e H7), cortes, a receita e a avaliação (segmentos, erros, explicabilidade
e região fora).

Cada etapa lê as bases por `carregar_base` (só `EDICAO_TREINO`; nada de 2017-2018 passa por
aqui), grava o relatório com `registrar` em `reports/` e os objetos pesados com `salvar_resultado`
em `data/experimentos/`. As pastas são lidas dos nomes deste módulo na hora da chamada, para o
teste poder redirecioná-las.

Resultados aninhados salvos (`salvar_resultado`):
- `aninhada_<parte>_B_com_peso` e `aninhada_<parte>_B_sem_peso`: as duas versões de H5;
- `aninhada_<parte>_<degrau>` (A a D): os degraus da escada; o B é o B de H5 com o peso decidido.
  O B é gravado como `{"resultado", "hash_dobras"}`; A, C e D, como o `ResultadoAninhado` direto
  (para eles, o hash conferido é o de `escada_<parte>.json`);
- `aninhada_direto_<degrau>`: o modelo direto de H6, com o hash das dobras;
- `previsoes_score`: o score fora da dobra decidido em H6 e H7 e as duas partes nas células;
- `selecao_<parte>`: parâmetros e previsões fora da dobra de `selecao_final` do vencedor.
"""

import hashlib
import json
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from occasion.avaliacao.erros import contextos_com_mais_erro
from occasion.avaliacao.explicacao import estabilidade, importancia_global
from occasion.avaliacao.segmentos import colunas_de_segmento, metricas_por_segmento
from occasion.bases.acesso import carregar_base
from occasion.config import (
    DADOS_EXPERIMENTOS,
    DADOS_PROCESSADOS,
    EDICAO_TREINO,
    FAIXAS_HORARIO,
    RELATORIOS,
    SEMENTE,
)
from occasion.decision.engine import CORTE_OCASIAO, CORTE_PREFERENCIA
from occasion.experimentos.aninhada import ResultadoAninhado, selecao_final, validacao_aninhada
from occasion.experimentos.especificacao import (
    INTERACOES,
    LIMITE_RILEY_DIRETO,
    ParteModelo,
    graus_de_liberdade,
    variaveis_da_parte,
)
from occasion.experimentos.registro import (
    Natureza,
    carregar_resultado,
    registrar,
    salvar_resultado,
)
from occasion.experimentos.regras import (
    melhor_logistico,
    padrao,
    sinal_negativo,
    superioridade,
    vencedor_escada,
)
from occasion.features.variaveis import DOMINIOS
from occasion.inference.treino import ajustar_parte
from occasion.models.base import ModeloParte, logit, sigmoide
from occasion.models.escada import Degrau
from occasion.models.logistica import ModeloLogistico
from occasion.receita import Receita, ReceitaParte, carregar_receita, salvar_receita
from occasion.score.score import CORTE_ALTA, CORTE_MEDIA
from occasion.validacao.bootstrap import B_PADRAO, Intervalo, intervalo_estatistica, intervalo_ganho
from occasion.validacao.dobras import dobra_das_linhas, escolher_dobras, tabela_de_upas
from occasion.validacao.metricas import (
    ajustar_sigmoide,
    calibracao,
    log_loss_ponderado,
    posicoes_ponderadas,
    pr_auc_ponderada,
    resumo,
    roc_auc_ponderada,
)

Parte = Literal["ocasiao", "preferencia"]
PARTES: tuple[Parte, ...] = ("ocasiao", "preferencia")

# Parte -> (base, coluna do alvo): a ocasião usa as células e a preferência, as ocasiões com
# refrigerante. Na preferência, as células entram como base extra (prevista com `tem_pizza = 1`).
_BASE_E_ALVO: dict[Parte, tuple[Literal["celulas", "ocasioes_refri"], str]] = {
    "ocasiao": ("celulas", "y_pizza_refri"),
    "preferencia": ("ocasioes_refri", "y_guarana"),
}
_AVISO_H4 = (
    "o coeficiente de tem_pizza fica encolhido pela regularização L2 (C fixo, o escolhido na "
    "seleção final): lê-se o sinal e a ordem de grandeza, não o tamanho exato do efeito"
)


# ---------------------------------------------------------------------------------- utilitários


def _pasta_relatorios() -> Path:
    return RELATORIOS / "experimentos"


def _celulas() -> pd.DataFrame:
    return carregar_base("celulas", EDICAO_TREINO, pasta=DADOS_PROCESSADOS)


def _ocasioes() -> pd.DataFrame:
    return carregar_base("ocasioes_refri", EDICAO_TREINO, pasta=DADOS_PROCESSADOS)


def _dobras() -> pd.DataFrame:
    return pd.read_parquet(DADOS_EXPERIMENTOS / "dobras.parquet")


def _base_da_parte(
    parte: Parte, celulas: pd.DataFrame, ocasioes: pd.DataFrame
) -> tuple[pd.DataFrame, str, dict[str, pd.DataFrame]]:
    nome, alvo = _BASE_E_ALVO[parte]
    base = celulas if nome == "celulas" else ocasioes
    return base, alvo, ({"celulas": celulas} if parte == "preferencia" else {})


def _usar_peso_decidido() -> dict[str, bool]:
    """O `usar_peso` de cada parte, lido do relatório de H5."""
    h5 = _h5()
    return {parte: bool(h5["metricas"][parte]["usar_peso"]) for parte in PARTES}


def _dados(base: pd.DataFrame, alvo: str) -> dict[str, Any]:
    y = base[alvo].to_numpy()
    return {
        "edicao": EDICAO_TREINO,
        "n": len(base),
        "positivos": int(np.sum(y == 1)),
        "setores": len(np.unique(base["cod_upa"].to_numpy())),
    }


def _ic_ganho(
    base: pd.DataFrame, alvo: str, p_candidato: np.ndarray, p_referencia: np.ndarray
) -> Intervalo:
    """Ganho do candidato no log loss ponderado, com bootstrap de setores dentro da UF."""
    return intervalo_ganho(
        log_loss_ponderado,
        False,
        base[alvo].to_numpy(),
        p_candidato,
        p_referencia,
        base["peso_final"].to_numpy(),
        base["cod_upa"].to_numpy(),
        base["uf"].to_numpy(),
    )


def _hash_das_dobras(dobras: pd.DataFrame) -> str:
    """sha256 de `cod_upa` e `dobra`, ordenados por setor: identifica a atribuição das dobras."""
    linhas = sorted(f"{u}|{d}" for u, d in zip(dobras["cod_upa"], dobras["dobra"], strict=True))
    return hashlib.sha256("\n".join(linhas).encode("utf-8")).hexdigest()


def _info_dobras(dobras: pd.DataFrame) -> dict[str, Any]:
    """Quais dobras o experimento usou: o hash e a semente gravada por `etapa_dobras`."""
    caminho = RELATORIOS / "dobras.json"
    semente = None
    if caminho.exists():
        semente = json.loads(caminho.read_text(encoding="utf-8"))["metricas"].get("semente_usada")
    return {"hash": _hash_das_dobras(dobras), "semente": semente}


def _h5() -> dict[str, Any]:
    """O relatório de H5; recusa o de uma execução que falhou."""
    h5 = json.loads((_pasta_relatorios() / "h5_peso.json").read_text(encoding="utf-8"))
    if h5["metricas"].get("status") != "ok":
        raise ValueError("h5_peso.json é de uma execução que falhou: rode `make peso` de novo")
    return h5


def _conferir_dobras_de_h5(dobras: pd.DataFrame) -> None:
    """Recusa dobras diferentes das que H5 usou: as comparações pareadas deixariam de valer."""
    usado = _h5()["dados"]["dobras"]["hash"]
    if usado != _hash_das_dobras(dobras):
        raise ValueError(
            "dobras.parquet mudou depois de H5 (hash diferente do de h5_peso.json): "
            "rode `make peso` de novo antes desta etapa"
        )


@dataclass
class _Parcial:
    """O que o experimento já sabe; vai para o JSON com `status` ok ou falhou."""

    configuracao: dict[str, Any]
    metricas: dict[str, Any] = field(default_factory=dict)
    dados: dict[str, Any] = field(default_factory=dict)
    semente: int = SEMENTE


@contextmanager
def _experimento(
    nome: str, natureza: Natureza, configuracao: dict[str, Any], pasta: Path
) -> Iterator[_Parcial]:
    """Grava o relatório da etapa; se ela falhar, grava `status: falhou`, o erro e o parcial."""
    parcial = _Parcial(configuracao)
    try:
        yield parcial
    except Exception as erro:
        parcial.metricas["status"] = "falhou"
        parcial.metricas["erro"] = {"tipo": type(erro).__name__, "mensagem": str(erro)}
        registrar(
            nome, natureza, parcial.configuracao, parcial.metricas, parcial.dados,
            pasta=pasta, semente=parcial.semente,
        )  # fmt: skip
        raise
    parcial.metricas["status"] = "ok"
    registrar(
        nome, natureza, parcial.configuracao, parcial.metricas, parcial.dados,
        pasta=pasta, semente=parcial.semente,
    )  # fmt: skip


def _sem_nan(p: np.ndarray, o_que: str) -> np.ndarray:
    if np.isnan(p).any():
        raise ValueError(f"previsão fora da dobra com NaN em {o_que}")
    return p


# ---------------------------------------------------------------------------------- dobras


def etapa_dobras() -> None:
    """Dobras externas por setor, conferidas nos quatro alvos; grava parquet e `dobras.json`."""
    config = {
        "k_externo": 5,
        "unidade": "cod_upa, espalhado por estrato",
        "semente_inicial": SEMENTE,
    }
    with _experimento("dobras", "exploratório", config, RELATORIOS) as r:
        celulas, ocasioes = _celulas(), _ocasioes()
        r.dados.update(
            celulas=_dados(celulas, "y_pizza_refri"), ocasioes_refri=_dados(ocasioes, "y_guarana")
        )
        fora = set(ocasioes["cod_upa"]) - set(celulas["cod_upa"])
        if fora:
            raise ValueError(f"setores das ocasiões ausentes das células: {sorted(fora)[:10]}")
        com_pizza = cast(pd.DataFrame, ocasioes[ocasioes["tem_pizza"] == 1])
        alvos = {
            "y_pizza_refri": (celulas, "y_pizza_refri"),
            "y_pizza_guarana": (celulas, "y_pizza_guarana"),
            "y_guarana": (ocasioes, "y_guarana"),
            "y_guarana_com_pizza": (com_pizza, "y_guarana"),
        }
        dobras, registro_dobras = escolher_dobras(tabela_de_upas(celulas), alvos)
        DADOS_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
        dobras.to_parquet(DADOS_EXPERIMENTOS / "dobras.parquet")
        semente = int(registro_dobras["semente"])
        escolhida = next(t for t in registro_dobras["tentativas"] if t["semente"] == semente)
        r.semente = semente
        r.configuracao["tentativas_feitas"] = len(registro_dobras["tentativas"])
        r.dados["hash_dobras"] = _hash_das_dobras(dobras)
        r.metricas.update(
            semente_usada=semente,
            conferencia=escolhida["resultado"],
            positivos_por_dobra={
                nome: v["positivos_por_dobra"]
                for nome, v in escolhida["resultado"].items()
                if nome != "geral"
            },
            tentativas=registro_dobras["tentativas"],
        )
    print(
        f"dobras: semente {semente}, {len(registro_dobras['tentativas'])} tentativa(s)", flush=True
    )


# ---------------------------------------------------------------------------------- H5


def etapa_peso() -> None:
    """H5: o degrau B com e sem `peso_final` no treino; o padrão é treinar com peso."""
    config = {
        "degrau": "B",
        "metrica": "log loss ponderado fora da dobra",
        "referencia": "treino com peso_final (o padrão)",
        "alternativa": "treino sem peso",
        "bootstrap": {"b": B_PADRAO, "estrato": "uf"},
        "resultados_salvos": "aninhada_<parte>_B_com_peso, aninhada_<parte>_B_sem_peso, "
        "aninhada_<parte>_B (o decidido, com o hash das dobras)",
    }
    with _experimento("h5_peso", "confirmatório", config, _pasta_relatorios()) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        r.dados["dobras"] = _info_dobras(dobras)
        for parte in PARTES:
            base, alvo, extras = _base_da_parte(parte, celulas, ocasioes)
            resultados: dict[bool, ResultadoAninhado] = {}
            tempos: dict[bool, float] = {}
            for usar in (True, False):
                rotulo = "com_peso" if usar else "sem_peso"
                print(f"H5 {parte}: B {rotulo}", flush=True)
                t0 = time.perf_counter()
                resultados[usar] = validacao_aninhada(
                    base, alvo, "B", cast(ParteModelo, parte), dobras,
                    usar_peso=usar, bases_extras=extras,
                )  # fmt: skip
                tempos[usar] = time.perf_counter() - t0
                salvar_resultado(f"aninhada_{parte}_B_{rotulo}", resultados[usar])
            com = _sem_nan(resultados[True].oof, f"{parte} com peso")
            sem = _sem_nan(resultados[False].oof, f"{parte} sem peso")
            ic = _ic_ganho(base, alvo, sem, com)
            veredito = padrao(ic)
            usar_peso = veredito != "padrão refutado"
            salvar_resultado(
                f"aninhada_{parte}_B",
                {"resultado": resultados[usar_peso], "hash_dobras": r.dados["dobras"]["hash"]},
            )
            y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
            r.metricas[parte] = {
                "usar_peso": usar_peso,
                "veredito": veredito,
                "ganho_sem_peso": ic,
                "log_loss_com_peso": log_loss_ponderado(y, com, w),
                "log_loss_sem_peso": log_loss_ponderado(y, sem, w),
                "parametros_por_dobra": {
                    "com_peso": resultados[True].parametros,
                    "sem_peso": resultados[False].parametros,
                },
                "tempo_s": {"com_peso": tempos[True], "sem_peso": tempos[False]},
            }
            r.dados[parte] = _dados(base, alvo)
            print(f"H5 {parte}: usar_peso={usar_peso} ({veredito})", flush=True)


# ---------------------------------------------------------------------------------- escada


def _progresso(parte: str, degrau: str) -> Callable[[int], None]:
    def avisar(k: int) -> None:
        print(f"escada {parte}: degrau {degrau}, dobra externa {k + 1} de 5", flush=True)

    return avisar


def _resumos(base: pd.DataFrame, alvo: str, oof: np.ndarray, parte: Parte) -> dict[str, Any]:
    y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
    saida: dict[str, Any] = {"resumo": resumo(y, oof, w)}
    if parte == "preferencia":
        com_pizza = base["tem_pizza"].to_numpy() == 1
        saida["resumo_com_pizza"] = resumo(y[com_pizza], oof[com_pizza], w[com_pizza])
    return saida


def etapa_escada(parte: Parte) -> None:
    """H1 a H3: A, B, C e D por validação aninhada, com o peso decidido em H5."""
    config: dict[str, Any] = {
        "parte": parte,
        "alvo": _BASE_E_ALVO[parte][1],
        "base": _BASE_E_ALVO[parte][0],
        "metrica_da_comparacao": "log loss ponderado fora da dobra (ganho positivo favorece o candidato)",
        "bootstrap": {"b": B_PADRAO, "estrato": "uf", "unidade": "cod_upa"},
        "b_de_h5": "reaproveitado de aninhada_<parte>_B",
        "resultados_salvos": f"aninhada_{parte}_<degrau>",
        "tempo_b_s": "tempo do B medido em H5",
    }
    with _experimento(f"escada_{parte}", "confirmatório", config, _pasta_relatorios()) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        base, alvo, extras = _base_da_parte(parte, celulas, ocasioes)
        r.dados.update(_dados(base, alvo))
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        usar_peso = _usar_peso_decidido()[parte]
        r.configuracao["usar_peso"] = usar_peso
        tempo_b = _h5()["metricas"][parte]["tempo_s"]["com_peso" if usar_peso else "sem_peso"]
        concluidos: dict[str, Any] = {}
        r.metricas["degraus_concluidos"] = concluidos

        resultados: dict[str, ResultadoAninhado] = {}
        tempos: dict[str, float] = {}
        for degrau in ("A", "B", "C", "D"):
            if degrau == "B":
                print(f"escada {parte}: B reaproveitado de H5", flush=True)
                salvo = carregar_resultado(f"aninhada_{parte}_B")
                if salvo["hash_dobras"] != r.dados["dobras"]["hash"]:
                    raise ValueError(
                        f"aninhada_{parte}_B.joblib usou outras dobras que o dobras.parquet atual: "
                        "rode `make peso` de novo"
                    )
                resultados["B"] = salvo["resultado"]
                tempos["B"] = float(tempo_b)
            else:
                print(f"escada {parte}: degrau {degrau}", flush=True)
                t0 = time.perf_counter()
                resultados[degrau] = validacao_aninhada(
                    base, alvo, cast(Degrau, degrau), cast(ParteModelo, parte), dobras,
                    usar_peso=usar_peso, bases_extras=extras, progresso=_progresso(parte, degrau),
                )  # fmt: skip
                tempos[degrau] = time.perf_counter() - t0
                salvar_resultado(f"aninhada_{parte}_{degrau}", resultados[degrau])
                print(f"escada {parte}: degrau {degrau} em {tempos[degrau]:.0f} s", flush=True)
            concluidos[degrau] = {
                "tempo_s": tempos[degrau],
                "parametros_por_dobra": resultados[degrau].parametros,
            }

        oof = {d: _sem_nan(res.oof, f"{parte} degrau {d}") for d, res in resultados.items()}
        cache: dict[tuple[str, str], Intervalo] = {}

        def comparar(candidato: str, referencia: str) -> Intervalo:
            if (candidato, referencia) not in cache:
                cache[(candidato, referencia)] = _ic_ganho(
                    base, alvo, oof[candidato], oof[referencia]
                )
            return cache[(candidato, referencia)]

        def hipotese(candidato: str, referencia: str) -> dict[str, Any]:
            ic = comparar(candidato, referencia)
            return {
                "candidato": candidato,
                "referencia": referencia,
                "intervalo": ic,
                "veredito": superioridade(ic),
            }

        h1, h2 = hipotese("B", "A"), hipotese("C", "B")
        h3 = hipotese("D", melhor_logistico(comparar("C", "B")))
        vencedor, sequencial = vencedor_escada(comparar)
        r.metricas.pop("degraus_concluidos")
        r.metricas.update(
            degraus={
                d: {
                    **_resumos(base, alvo, oof[d], parte),
                    "parametros_por_dobra": resultados[d].parametros,
                    "tempo_s": tempos[d],
                }
                for d in ("A", "B", "C", "D")
            },
            comparacoes={"H1": h1, "H2": h2, "H3": h3},
            comparacoes_sequenciais=sequencial,
            vencedor=vencedor,
        )


# ---------------------------------------------------------------------------------- H4


def etapa_h4(b: int = B_PADRAO) -> None:
    """H4: sinal do coeficiente de `tem_pizza` no B da preferência, com bootstrap de setores."""
    config: dict[str, Any] = {
        "degrau": "B",
        "parte": "preferencia",
        "bootstrap": {"b": b, "estrato": "uf", "unidade": "cod_upa"},
        "pesos_na_replica": "(peso_final se usar_peso, senão 1) x multiplicador, média 1",
        "aviso": _AVISO_H4,
    }
    with _experimento("h4_tem_pizza", "confirmatório", config, _pasta_relatorios()) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        r.dados.update(_dados(ocasioes, "y_guarana"))
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        usar_peso = _usar_peso_decidido()["preferencia"]
        r.configuracao["usar_peso"] = usar_peso
        print("H4: seleção final do B da preferência", flush=True)
        params, _, _ = selecao_final(
            ocasioes, "y_guarana", "B", "preferencia", dobras,
            usar_peso=usar_peso, bases_extras={"celulas": celulas},
        )  # fmt: skip
        C = float(params["C"])
        r.configuracao["C"] = C
        X = cast(pd.DataFrame, ocasioes[variaveis_da_parte("preferencia")])
        y = cast(pd.Series, ocasioes["y_guarana"])
        w0 = ocasioes["peso_final"].to_numpy() if usar_peso else np.ones(len(ocasioes))

        def coeficiente(multiplicador: np.ndarray) -> float:
            modelo = ModeloLogistico("preferencia", C=C).fit(X, y, pd.Series(w0 * multiplicador))
            return modelo.coeficiente("tem_pizza")

        print(f"H4: bootstrap de {b} réplicas", flush=True)
        ic = intervalo_estatistica(
            coeficiente, ocasioes["cod_upa"].to_numpy(), ocasioes["uf"].to_numpy(), b=b
        )
        r.metricas.update(
            coeficiente_tem_pizza=ic.estimativa,
            intervalo=ic,
            veredito=sinal_negativo(ic),
            razao_de_chances=float(np.exp(ic.estimativa)),
        )


# ---------------------------------------------------------------------------------- embaralhado


def etapa_embaralhado(
    partes: Sequence[Parte] = PARTES, usar_peso: Mapping[str, bool] | None = None
) -> None:
    """Controle de vazamento: B com o alvo permutado contra a prevalência ponderada do treino.

    `usar_peso` fixa o uso de peso por parte; sem ele, vale o decidido em H5.
    """
    config = {
        "controle": "vazamento (não é uma hipótese)",
        "degrau": "B",
        "partes": list(partes),
        "alvo": "permutado entre as linhas, com np.random.default_rng(SEMENTE)",
        "referencia": "constante igual à prevalência ponderada (peso_final) do treino de cada dobra",
        "esperado": "refutada: o modelo não pode superar a constante",
    }
    with _experimento("rotulos_embaralhados", "exploratório", config, _pasta_relatorios()) as r:
        dobras = _dobras()
        r.dados["dobras"] = _info_dobras(dobras)
        celulas, ocasioes = _celulas(), _ocasioes()
        uso = dict(usar_peso) if usar_peso is not None else _usar_peso_decidido()
        for parte in partes:
            base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
            rng = np.random.default_rng(SEMENTE)
            permutado = base.assign(**{alvo: rng.permutation(base[alvo].to_numpy())})
            print(f"embaralhado {parte}: B", flush=True)
            res = validacao_aninhada(
                permutado, alvo, "B", cast(ParteModelo, parte), dobras, usar_peso=uso[parte]
            )
            y, w = permutado[alvo].to_numpy(), permutado["peso_final"].to_numpy()
            por_linha = dobra_das_linhas(permutado, dobras)
            constante = np.full(len(permutado), np.nan)
            for k in range(int(por_linha.max()) + 1):
                treino = por_linha != k
                constante[~treino] = np.sum(w[treino] * y[treino]) / np.sum(w[treino])
            modelo = _sem_nan(res.oof, f"{parte} embaralhado")
            ic = _ic_ganho(permutado, alvo, modelo, _sem_nan(constante, f"{parte} constante"))
            r.metricas[parte] = {
                "veredito": superioridade(ic),
                "ganho_do_modelo_sobre_a_constante": ic,
                "log_loss_modelo": log_loss_ponderado(y, modelo, w),
                "log_loss_constante": log_loss_ponderado(y, constante, w),
                "usar_peso": uso[parte],
            }
            r.dados[parte] = _dados(base, alvo)
            print(f"embaralhado {parte}: {r.metricas[parte]['veredito']}", flush=True)


# ------------------------------------------------- calibração, produto (H6 e H7), cortes e receita

ALVO_SCORE = "y_pizza_guarana"  # o fato observado contra o qual o score é medido, nas células
_REGRA_DA_SIGMOIDE = (
    "em cada dobra externa k, ajustar_sigmoide (com peso_final) nas previsões internas de k, "
    "aplicada às previsões fora da dobra k; a dobra k nunca entra no ajuste da própria sigmoide"
)
_BOOTSTRAP = {"b": B_PADRAO, "estrato": "uf", "unidade": "cod_upa"}


def _relatorio(nome: str, dobras: pd.DataFrame) -> dict[str, Any]:
    """Relatório de uma etapa anterior; recusa o que falhou ou o feito com outras dobras."""
    rel = json.loads((_pasta_relatorios() / f"{nome}.json").read_text(encoding="utf-8"))
    if rel["metricas"].get("status") != "ok":
        raise ValueError(f"{nome}.json é de uma execução que falhou: rode a etapa de novo")
    if rel.get("dados", {}).get("dobras", {}).get("hash") != _hash_das_dobras(dobras):
        raise ValueError(
            f"{nome}.json usou outras dobras que o dobras.parquet atual: rode a etapa de novo"
        )
    return rel


def _vencedor(parte: Parte, dobras: pd.DataFrame) -> Degrau:
    return cast(Degrau, _relatorio(f"escada_{parte}", dobras)["metricas"]["vencedor"])


def _carregar_aninhada(nome: str, hash_dobras: str) -> ResultadoAninhado:
    """Resultado aninhado salvo, nos dois formatos: com o hash das dobras (conferido) ou direto."""
    salvo = carregar_resultado(nome)
    if isinstance(salvo, dict):
        if salvo.get("hash_dobras") != hash_dobras:
            raise ValueError(
                f"{nome}.joblib usou outras dobras que o dobras.parquet atual: rode a etapa de novo"
            )
        salvo = salvo["resultado"]
    if not isinstance(salvo, ResultadoAninhado):
        raise TypeError(f"{nome}.joblib não guarda um ResultadoAninhado: {type(salvo).__name__}")
    return salvo


def _sigmoides(
    internos: Sequence[tuple[np.ndarray, np.ndarray]],
    y: np.ndarray,
    w: np.ndarray,
    por_linha: np.ndarray,
) -> list[tuple[float, float]]:
    """Uma sigmoide `(inclinacao, intercepto)` por dobra externa, só com as previsões internas.

    `internos[k]` são as posições e as previsões internas da dobra externa `k`; uma posição que
    pertença à própria dobra `k` seria vazamento e é recusada.
    """
    saida = []
    for k, (posicoes, p) in enumerate(internos):
        if np.any(por_linha[posicoes] == k):
            raise ValueError(
                f"previsão interna da dobra externa {k} inclui linhas da própria dobra"
            )
        p = _sem_nan(np.asarray(p, dtype=float), f"previsão interna da dobra externa {k}")
        saida.append(ajustar_sigmoide(p, y[posicoes], w[posicoes]))
    return saida


def _sigmoides_por_dobra(
    res: ResultadoAninhado, y: np.ndarray, w: np.ndarray, por_linha: np.ndarray
) -> list[tuple[float, float]]:
    """As sigmoides de um resultado aninhado, nas previsões internas da própria base."""
    return _sigmoides([(i.indices, i.p) for i in res.interno], y, w, por_linha)


def _aplicar(p: np.ndarray, sigmoide_k: tuple[float, float]) -> np.ndarray:
    inclinacao, intercepto = sigmoide_k
    return np.asarray(sigmoide(inclinacao * logit(p) + intercepto), dtype=float)


def _calibrar_por_dobra(
    p: np.ndarray, por_linha: np.ndarray, sigmoides: Sequence[tuple[float, float]]
) -> np.ndarray:
    """Aplica a sigmoide da dobra `k` às previsões fora da dobra das linhas da dobra `k`."""
    saida = np.full(len(p), np.nan)
    for k, sig in enumerate(sigmoides):
        dentro = por_linha == k
        saida[dentro] = _aplicar(p[dentro], sig)
    return saida


def _sigmoides_em_json(sigmoides: Sequence[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"inclinacao": a, "intercepto": b} for a, b in sigmoides]


@dataclass
class _PrevisoesCelulas:
    """Uma parte (ou o score) nas células: fora da dobra e, por dobra externa, as internas."""

    fora: np.ndarray
    interno: list[tuple[np.ndarray, np.ndarray]]  # dobra k: posições nas células e previsões


def _nas_celulas(
    res: ResultadoAninhado,
    parte: ParteModelo,
    sigmoides: Sequence[tuple[float, float]] | None,
    por_linha_celulas: np.ndarray,
) -> _PrevisoesCelulas:
    """As previsões de `res` nas células; com `sigmoides`, calibradas dobra a dobra.

    Na preferência, as células são a base extra (prevista com `tem_pizza = 1`). A sigmoide da
    dobra `k` é a mesma para a previsão fora da dobra `k` e para as internas de `k`.
    """
    if parte == "preferencia":
        fora = res.oof_extras["celulas"]
        interno = [i.extras["celulas"] for i in res.interno]
    else:
        fora = res.oof
        interno = [(i.indices, i.p) for i in res.interno]
    fora = _sem_nan(np.asarray(fora, dtype=float), f"{parte} nas células")
    if len(fora) != len(por_linha_celulas):
        raise ValueError(f"o resultado salvo de {parte} não tem uma previsão por célula")
    if sigmoides is None:
        return _PrevisoesCelulas(fora, [(pos, np.asarray(p, float)) for pos, p in interno])
    return _PrevisoesCelulas(
        _calibrar_por_dobra(fora, por_linha_celulas, sigmoides),
        [(pos, _aplicar(np.asarray(p, float), sigmoides[k])) for k, (pos, p) in enumerate(interno)],
    )


def _produto(oc: _PrevisoesCelulas, pref: _PrevisoesCelulas) -> _PrevisoesCelulas:
    """Ocasião vezes preferência, fora da dobra e nas internas de cada dobra externa.

    As internas precisam estar nas mesmas células nas duas partes (as dobras internas saem da
    tabela de setores); se não estiverem, o produto misturaria modelos que viram o setor.
    """
    if len(oc.interno) != len(pref.interno):
        raise ValueError("as duas partes não têm o mesmo número de dobras externas")
    interno = []
    for k, ((pos_o, p_o), (pos_p, p_p)) in enumerate(zip(oc.interno, pref.interno, strict=True)):
        if not np.array_equal(pos_o, pos_p):
            raise ValueError(
                f"as previsões internas da dobra externa {k} não estão nas mesmas linhas "
                "nas duas partes"
            )
        interno.append((pos_o, p_o * p_p))
    return _PrevisoesCelulas(oc.fora * pref.fora, interno)


def _previsoes_da_parte(
    parte: Parte,
    celulas: pd.DataFrame,
    ocasioes: pd.DataFrame,
    dobras: pd.DataFrame,
    hash_dobras: str,
    calibrar: bool,
) -> _PrevisoesCelulas:
    """O vencedor da escada da parte, nas células, com a calibração decidida."""
    degrau = _vencedor(parte, dobras)
    res = _carregar_aninhada(f"aninhada_{parte}_{degrau}", hash_dobras)
    base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
    sig = None
    if calibrar:
        sig = _sigmoides_por_dobra(
            res, base[alvo].to_numpy(), base["peso_final"].to_numpy(),
            dobra_das_linhas(base, dobras),
        )  # fmt: skip
    return _nas_celulas(res, parte, sig, dobra_das_linhas(celulas, dobras))


def _graus_do_direto(degrau: Degrau) -> int | dict[str, int] | None:
    """Graus de liberdade nominais do direto (B e C); A e D não têm a contagem logística."""
    if degrau == "B":
        return graus_de_liberdade("direto", "B")
    if degrau == "C":
        return {c: graus_de_liberdade("direto", "C", c) for c in INTERACOES["direto"]}
    return None


def _relativo(caminho: Path) -> str:
    """Caminho relativo à raiz do repositório (a pasta acima de `reports/`)."""
    return caminho.relative_to(RELATORIOS.parent).as_posix()


# ---------------------------------------------------------------------------------- calibração


def etapa_calibracao() -> None:
    """Sigmoide no vencedor da escada de cada parte; entra se `superioridade` confirmar o ganho."""
    config = {
        "o_que": "correção sigmoide no vencedor da escada de cada parte (seção 8.2 do design)",
        "ajuste": _REGRA_DA_SIGMOIDE,
        "metrica_da_comparacao": "ganho da versão calibrada no log loss ponderado fora da dobra",
        "regra": "superioridade",
        "bootstrap": _BOOTSTRAP,
        "niveis_de_calibracao": "média (o_e e intercepto_cil), inclinação e curva por faixas",
    }
    with _experimento("calibracao", "confirmatório", config, _pasta_relatorios()) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        for parte in PARTES:
            degrau = _vencedor(parte, dobras)
            res = _carregar_aninhada(f"aninhada_{parte}_{degrau}", r.dados["dobras"]["hash"])
            base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
            y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
            por_linha = dobra_das_linhas(base, dobras)
            antes = _sem_nan(res.oof, f"{parte} degrau {degrau}")
            if len(antes) != len(base):
                raise ValueError(f"aninhada_{parte}_{degrau} não tem uma previsão por linha")
            sigmoides = _sigmoides_por_dobra(res, y, w, por_linha)
            depois = _sem_nan(
                _calibrar_por_dobra(antes, por_linha, sigmoides), f"{parte} calibrada"
            )
            ic = _ic_ganho(base, alvo, depois, antes)
            veredito = superioridade(ic)
            m: dict[str, Any] = {
                "vencedor": degrau,
                "calibrar": veredito == "confirmada",
                "veredito": veredito,
                "ganho_calibrada": ic,
                "log_loss_antes": log_loss_ponderado(y, antes, w),
                "log_loss_depois": log_loss_ponderado(y, depois, w),
                "antes": calibracao(y, antes, w),
                "depois": calibracao(y, depois, w),
                "sigmoide_por_dobra": _sigmoides_em_json(sigmoides),
            }
            if parte == "preferencia":
                pizza = base["tem_pizza"].to_numpy() == 1
                m["antes_com_pizza"] = calibracao(y[pizza], antes[pizza], w[pizza])
                m["depois_com_pizza"] = calibracao(y[pizza], depois[pizza], w[pizza])
            r.metricas[parte] = m
            r.dados[parte] = _dados(base, alvo)
            print(f"calibração {parte} (degrau {degrau}): {veredito}", flush=True)


# ---------------------------------------------------------------------------------- H6 e H7


def etapa_produto(b: int = B_PADRAO) -> None:
    """H6 (produto contra o modelo direto) e H7 (calibração do que calcula o score).

    `b` é o número de réplicas do intervalo da inclinação de H7, que só é relatado.
    """
    config: dict[str, Any] = {
        "alvo": f"{ALVO_SCORE} nas células",
        "produto": "ocasião fora da dobra x preferência fora da dobra nas células "
        "(tem_pizza = 1), cada parte com a calibração decidida em calibracao.json",
        "H6": {
            "direto": "validação aninhada do degrau vencedor da ocasião, parte 'direto', com o "
            "peso de H5 e a calibração decididos para a ocasião",
            "metrica": "ganho do direto sobre o produto no log loss ponderado; PR-AUC ao lado",
            "regra": "padrao (o padrão é o produto)",
        },
        "H7": {
            "ajuste": _REGRA_DA_SIGMOIDE + "; no produto, as internas são o produto das "
            "internas das partes, nas mesmas células",
            "regra": "superioridade no log loss ponderado",
            "inclinacao": f"intervalo com {b} réplicas, reajustando ajustar_sigmoide em cada "
            "uma; só relatado",
        },
        "bootstrap": _BOOTSTRAP,
    }
    with _experimento("h6_h7_produto", "confirmatório", config, _pasta_relatorios()) as r:
        dobras = _dobras()
        cal = _relatorio("calibracao", dobras)
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        hash_dobras = r.dados["dobras"]["hash"]
        celulas, ocasioes = _celulas(), _ocasioes()
        for p in PARTES:
            if cal["metricas"][p]["vencedor"] != _vencedor(p, dobras):
                raise ValueError(
                    f"calibracao.json e escada_{p}.json discordam do vencedor: "
                    "rode `make calibracao` de novo"
                )
        calibrar = {p: bool(cal["metricas"][p]["calibrar"]) for p in PARTES}
        r.configuracao["calibrar_partes"] = calibrar
        oc, pref = (
            _previsoes_da_parte(p, celulas, ocasioes, dobras, hash_dobras, calibrar[p])
            for p in PARTES
        )
        produto = _produto(oc, pref)
        y, w = celulas[ALVO_SCORE].to_numpy(), celulas["peso_final"].to_numpy()
        por_linha = dobra_das_linhas(celulas, dobras)
        r.dados.update(_dados(celulas, ALVO_SCORE))

        # H6: o modelo direto do mesmo degrau do vencedor da ocasião
        degrau = _vencedor("ocasiao", dobras)
        usar_peso = _usar_peso_decidido()["ocasiao"]
        print(f"H6: modelo direto, degrau {degrau}", flush=True)
        t0 = time.perf_counter()
        res_direto = validacao_aninhada(
            celulas, ALVO_SCORE, degrau, "direto", dobras,
            usar_peso=usar_peso,
            progresso=lambda k: print(f"H6: direto {degrau}, dobra externa {k + 1} de 5", flush=True),
        )  # fmt: skip
        tempo = time.perf_counter() - t0
        salvar_resultado(
            f"aninhada_direto_{degrau}", {"resultado": res_direto, "hash_dobras": hash_dobras}
        )
        sig_direto = (
            _sigmoides_por_dobra(res_direto, y, w, por_linha) if calibrar["ocasiao"] else None
        )
        direto = _nas_celulas(res_direto, "direto", sig_direto, por_linha)
        ic6 = _ic_ganho(celulas, ALVO_SCORE, direto.fora, produto.fora)
        v6 = padrao(ic6)
        score: Literal["produto", "direto"] = "direto" if v6 == "padrão refutado" else "produto"
        r.metricas["H6"] = {
            "degrau": degrau,
            "usar_peso": usar_peso,
            "calibrar": calibrar["ocasiao"],
            "ganho_do_direto": ic6,
            "veredito": v6,
            "log_loss": {
                "produto": log_loss_ponderado(y, produto.fora, w),
                "direto": log_loss_ponderado(y, direto.fora, w),
            },
            "pr_auc": {
                "produto": pr_auc_ponderada(y, produto.fora, w),
                "direto": pr_auc_ponderada(y, direto.fora, w),
            },
            "roc_auc": {
                "produto": roc_auc_ponderada(y, produto.fora, w),
                "direto": roc_auc_ponderada(y, direto.fora, w),
            },
            "graus_de_liberdade_nominais_do_direto": _graus_do_direto(degrau),
            "limite_riley_do_direto": LIMITE_RILEY_DIRETO,
            "parametros_por_dobra_do_direto": res_direto.parametros,
            "tempo_direto_s": tempo,
        }
        print(f"H6: {v6}; o score é o {score}", flush=True)

        # H7: a sigmoide sobre o que calcula o score
        sobre = direto if score == "direto" else produto
        sigmoides = _sigmoides(sobre.interno, y, w, por_linha)
        corrigida = _sem_nan(_calibrar_por_dobra(sobre.fora, por_linha, sigmoides), "score")
        ic7 = _ic_ganho(celulas, ALVO_SCORE, corrigida, sobre.fora)
        v7 = superioridade(ic7)
        fora = sobre.fora
        cod_upa, uf = celulas["cod_upa"].to_numpy(), celulas["uf"].to_numpy()
        print(f"H7: intervalo da inclinação com {b} réplicas", flush=True)
        inclinacao = intervalo_estatistica(
            lambda m: ajustar_sigmoide(fora, y, w * m)[0], cod_upa, uf, b=b
        )
        r.metricas["H7"] = {
            "sobre": score,
            "veredito": v7,
            "calibrar_score": v7 == "confirmada",
            "ganho_corrigida": ic7,
            "inclinacao": inclinacao,
            "log_loss_antes": log_loss_ponderado(y, fora, w),
            "log_loss_depois": log_loss_ponderado(y, corrigida, w),
            "antes": calibracao(y, fora, w),
            "depois": calibracao(y, corrigida, w),
            "sigmoide_por_dobra": _sigmoides_em_json(sigmoides),
        }
        r.metricas["score"] = score
        salvar_resultado(
            "previsoes_score",
            {
                "score": corrigida if v7 == "confirmada" else fora,
                "tipo": score,
                "calibrar_score": v7 == "confirmada",
                "ocasiao": oc.fora,
                "preferencia": pref.fora,
                "hash_dobras": hash_dobras,
            },
        )
        print(f"H7: {v7}", flush=True)


# ---------------------------------------------------------------------------------- cortes


def _quadrantes(
    pos_oc: np.ndarray, pos_pref: np.ndarray, corte_oc: float, corte_pref: float
) -> np.ndarray:
    """Quadrante de cada célula (0 a 3): ocasião alta vale 2, preferência alta vale 1."""
    return 2 * (pos_oc >= corte_oc).astype(int) + (pos_pref >= corte_pref).astype(int)


def etapa_cortes() -> None:
    """Relata o score no corte de 80 e a sensibilidade dos cortes do Decision Engine (±10)."""
    config: dict[str, Any] = {
        "cortes": {
            "alta": CORTE_ALTA,
            "media": CORTE_MEDIA,
            "ocasiao": CORTE_OCASIAO,
            "preferencia": CORTE_PREFERENCIA,
        },
        "posicoes": "posicoes_ponderadas (peso_final) nas células, sobre a previsão fora da dobra",
        "sensibilidade": "quadrante = (posição da parte 1 >= corte da ocasião, posição da parte 2 "
        ">= corte da preferência), com as partes do produto, que são as que o Decision Engine "
        "usa; parcela ponderada das células que muda de quadrante com cada corte a ±10",
        "decisao": "só relata: os cortes ficam os do pré-registro",
    }
    with _experimento("cortes", "exploratório", config, _pasta_relatorios()) as r:
        dobras = _dobras()
        prod = _relatorio("h6_h7_produto", dobras)
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        salvo = carregar_resultado("previsoes_score")
        if salvo["hash_dobras"] != r.dados["dobras"]["hash"]:
            raise ValueError("previsoes_score.joblib usou outras dobras: rode `make produto`")
        if salvo["tipo"] != prod["metricas"]["score"]:
            raise ValueError("previsoes_score.joblib não é o de h6_h7_produto.json")
        celulas = _celulas()
        r.dados.update(_dados(celulas, ALVO_SCORE))
        y, w = celulas[ALVO_SCORE].to_numpy(), celulas["peso_final"].to_numpy()
        score = _sem_nan(np.asarray(salvo["score"], float), "score")
        posicao = posicoes_ponderadas(score, w)
        total = float(np.sum(w))
        alta, media = posicao >= CORTE_ALTA, (posicao >= CORTE_MEDIA) & (posicao < CORTE_ALTA)
        r.metricas["score"] = salvo["tipo"]
        r.metricas["resumo"] = resumo(y, score, w, posicao=posicao)
        r.metricas["classificacao"] = {
            "alta": float(np.sum(w[alta]) / total),
            "media": float(np.sum(w[media]) / total),
            "baixa": float(np.sum(w[~alta & ~media]) / total),
        }

        pos_oc = posicoes_ponderadas(np.asarray(salvo["ocasiao"], float), w)
        pos_pref = posicoes_ponderadas(np.asarray(salvo["preferencia"], float), w)
        base_q = _quadrantes(pos_oc, pos_pref, CORTE_OCASIAO, CORTE_PREFERENCIA)
        sensibilidade: dict[str, Any] = {
            "quadrantes": {
                rotulo: float(np.sum(w[base_q == q]) / total)
                for q, rotulo in enumerate(
                    (
                        "ocasiao_baixa_preferencia_baixa",
                        "ocasiao_baixa_preferencia_alta",
                        "ocasiao_alta_preferencia_baixa",
                        "ocasiao_alta_preferencia_alta",
                    )
                )
            }
        }
        for qual, corte in (("ocasiao", CORTE_OCASIAO), ("preferencia", CORTE_PREFERENCIA)):
            sensibilidade[qual] = {"corte": corte}
            for chave, delta in (("menos_10", -10.0), ("mais_10", 10.0)):
                cortes = (
                    (corte + delta, CORTE_PREFERENCIA)
                    if qual == "ocasiao"
                    else (CORTE_OCASIAO, corte + delta)
                )
                muda = _quadrantes(pos_oc, pos_pref, *cortes) != base_q
                sensibilidade[qual][chave] = float(np.sum(w[muda]) / total)
        r.metricas["sensibilidade"] = sensibilidade
        if salvo["tipo"] == "direto":
            r.metricas["aviso_direto"] = (
                "o score é o modelo direto (H6), mas a sensibilidade usa as partes do produto, "
                "porque o Decision Engine decide pelas partes"
            )
        print("cortes: relatório gravado", flush=True)


# ---------------------------------------------------------------------------------- receita

_RELATORIOS_DA_RECEITA = (
    "h5_peso",
    "escada_ocasiao",
    "escada_preferencia",
    "calibracao",
    "h6_h7_produto",
    "cortes",
)


def etapa_receita() -> None:
    """Monta a receita: `selecao_final` do vencedor de cada parte e do degrau A (referência de H8)."""
    config = {
        "o_que": "selecao_final do degrau vencedor de cada parte (e do direto, se H6 o promover) "
        "e do degrau A, a referência de H8",
        "metrica": "log loss ponderado fora da dobra nas 5 dobras externas",
        "resultados_salvos": "selecao_<parte> (parâmetros e previsões, com o hash das dobras)",
        "receita": "reports/receita.json",
    }
    with _experimento("selecao_final", "exploratório", config, _pasta_relatorios()) as r:
        dobras = _dobras()
        rel = {nome: _relatorio(nome, dobras) for nome in _RELATORIOS_DA_RECEITA}
        r.dados["dobras"] = _info_dobras(dobras)
        hash_dobras = r.dados["dobras"]["hash"]
        celulas, ocasioes = _celulas(), _ocasioes()
        usar_peso = {p: bool(rel["h5_peso"]["metricas"][p]["usar_peso"]) for p in PARTES}
        calibrar = {p: bool(rel["calibracao"]["metricas"][p]["calibrar"]) for p in PARTES}
        score = cast(Literal["produto", "direto"], rel["h6_h7_produto"]["metricas"]["score"])
        if rel["cortes"]["metricas"].get("score") != score:
            raise ValueError(
                "cortes.json é de outro score que h6_h7_produto.json: rode `make cortes` de novo"
            )
        calibrar_score = bool(rel["h6_h7_produto"]["metricas"]["H7"]["calibrar_score"])
        partes: list[ParteModelo] = ["ocasiao", "preferencia"]
        if score == "direto":
            partes.append("direto")

        escolhas: dict[ParteModelo, ReceitaParte] = {}
        referencia_a: dict[ParteModelo, dict[str, Any]] = {}
        interacoes: dict[ParteModelo, list[tuple[str, str]]] = {
            p: [] for p in ("ocasiao", "preferencia", "direto")
        }
        for parte in partes:
            fonte: Parte = "ocasiao" if parte == "direto" else parte
            degrau = cast(Degrau, rel[f"escada_{fonte}"]["metricas"]["vencedor"])
            if rel["calibracao"]["metricas"][fonte]["vencedor"] != degrau:
                raise ValueError(f"calibracao.json e escada_{fonte}.json discordam do vencedor")
            if parte == "direto":
                base, alvo, extras = celulas, ALVO_SCORE, {}
            else:
                base, alvo, extras = _base_da_parte(fonte, celulas, ocasioes)
            print(f"receita {parte}: seleção final do degrau {degrau}", flush=True)
            params, oof, oof_extras = selecao_final(
                base, alvo, degrau, parte, dobras,
                usar_peso=usar_peso[fonte], bases_extras=extras,
            )  # fmt: skip
            salvar_resultado(
                f"selecao_{parte}",
                {
                    "degrau": degrau,
                    "parametros": params,
                    "oof": oof,
                    "oof_extras": oof_extras,
                    "hash_dobras": hash_dobras,
                },
            )
            escolhas[parte] = ReceitaParte(
                degrau=degrau, parametros=params,
                usar_peso=usar_peso[fonte], calibrar=calibrar[fonte],
            )  # fmt: skip
            if degrau == "C":
                interacoes[parte] = list(INTERACOES[parte][params["interacoes"]])
            print(f"receita {parte}: seleção final do degrau A", flush=True)
            params_a, _, _ = selecao_final(
                base, alvo, "A", parte, dobras, usar_peso=usar_peso[fonte]
            )
            referencia_a[parte] = params_a
            y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
            r.metricas[parte] = {
                "degrau": degrau,
                "parametros": params,
                "log_loss_fora_da_dobra": log_loss_ponderado(y, _sem_nan(oof, parte), w),
                "referencia_a": params_a,
            }
            r.dados[parte] = _dados(base, alvo)

        pasta = _pasta_relatorios()
        experimentos = [_relativo(RELATORIOS / "dobras.json")]
        experimentos += [_relativo(pasta / f"{nome}.json") for nome in _RELATORIOS_DA_RECEITA]
        experimentos.append(_relativo(pasta / "selecao_final.json"))
        receita = Receita(
            faixas_horario={nome: list(faixa) for nome, faixa in FAIXAS_HORARIO.items()},
            variaveis={p: variaveis_da_parte(p) for p in ("ocasiao", "preferencia", "direto")},
            interacoes=interacoes,
            ocasiao=escolhas["ocasiao"],
            preferencia=escolhas["preferencia"],
            score=score,
            direto=escolhas.get("direto"),
            calibrar_score=calibrar_score,
            referencia_a=referencia_a,
            cortes={
                "alta": CORTE_ALTA,
                "media": CORTE_MEDIA,
                "ocasiao": CORTE_OCASIAO,
                "preferencia": CORTE_PREFERENCIA,
            },
            experimentos=experimentos,
        )
        caminho = salvar_receita(receita, RELATORIOS / "receita.json")
        r.metricas["receita"] = _relativo(caminho)
        print(f"receita gravada em {caminho}", flush=True)


# ---------------------------------------------------------------------------------- avaliação

_DECLARACAO_REGIAO_FORA = (
    "análise secundária, que não muda a receita: para cada grande região, cada parte é treinada "
    "nas outras quatro, com os parâmetros e o uso de peso da receita, e avaliada na região "
    "deixada de fora; as UFs dessa região ficam ausentes do treino. Os dois lados estão sem "
    "calibração (sem a sigmoide): o lado 'treinada_nas_outras_quatro' é a receita sem a sigmoide, "
    "e o lado 'fora_da_dobra' é a previsão crua do vencedor (dobras por setor), nas linhas da "
    "mesma região. O score é o produto das duas partes (a preferência com tem_pizza = 1) ou o "
    "modelo direto, conforme a receita. Os parâmetros da receita foram escolhidos com a região "
    "deixada de fora incluída na seleção, o que deixa o lado 'treinada_nas_outras_quatro' um "
    "pouco otimista."
)


def _conferir_receita_e_selecoes(
    receita: Receita, dobras: pd.DataFrame, hash_dobras: str
) -> dict[ParteModelo, dict[str, Any]]:
    """Recusa uma receita que não é a dos relatórios e das seleções em uso; devolve as seleções."""
    h5 = _usar_peso_decidido()
    cal = _relatorio("calibracao", dobras)["metricas"]
    prod = _relatorio("h6_h7_produto", dobras)["metricas"]
    selecoes: dict[ParteModelo, dict[str, Any]] = {}
    partes: list[tuple[ParteModelo, ReceitaParte, Parte]] = [
        ("ocasiao", receita.ocasiao, "ocasiao"),
        ("preferencia", receita.preferencia, "preferencia"),
    ]
    if receita.score == "direto" and receita.direto is not None:
        partes.append(("direto", receita.direto, "ocasiao"))
    for parte, rp, fonte in partes:
        if (
            rp.degrau != _vencedor(fonte, dobras)
            or rp.usar_peso != h5[fonte]
            or rp.calibrar != bool(cal[fonte]["calibrar"])
        ):
            raise ValueError(
                f"receita.json não confere com os relatórios na parte {parte!r}: "
                "rode `make receita` de novo"
            )
        sel = carregar_resultado(f"selecao_{parte}")
        if (
            sel.get("hash_dobras") != hash_dobras
            or sel["degrau"] != rp.degrau
            or sel["parametros"] != rp.parametros
        ):
            raise ValueError(f"selecao_{parte}.joblib não é a da receita: rode `make receita`")
        selecoes[parte] = sel
    if receita.score != prod["score"] or receita.calibrar_score != bool(
        prod["H7"]["calibrar_score"]
    ):
        raise ValueError("receita.json e h6_h7_produto.json discordam do score")
    return selecoes


def _avaliar(base: pd.DataFrame, alvo: str, p: np.ndarray, posicao: bool = False) -> dict[str, Any]:
    """Resumo, métricas por segmento e contextos com mais erro de uma previsão fora da dobra."""
    y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
    return {
        "resumo": resumo(y, p, w, posicao=posicoes_ponderadas(p, w) if posicao else None),
        "segmentos": metricas_por_segmento(y, p, w, colunas_de_segmento(base)),
        "erros": contextos_com_mais_erro(base, y, p),
    }


def _oof_decidido(
    res: ResultadoAninhado, base: pd.DataFrame, alvo: str, dobras: pd.DataFrame, calibrar: bool
) -> np.ndarray:
    """A previsão fora da dobra do vencedor, com a sigmoide por dobra se a calibração entrou."""
    oof = _sem_nan(np.asarray(res.oof, dtype=float), alvo)
    if not calibrar:
        return oof
    por_linha = dobra_das_linhas(base, dobras)
    sig = _sigmoides_por_dobra(res, base[alvo].to_numpy(), base["peso_final"].to_numpy(), por_linha)
    return _sem_nan(_calibrar_por_dobra(oof, por_linha, sig), f"{alvo} calibrada")


def _explicar(
    final: ModeloParte, res: ResultadoAninhado, base: pd.DataFrame, parte: ParteModelo
) -> dict[str, Any]:
    """Importância no modelo final (2008-2009 inteira) e estabilidade nos modelos das dobras."""
    X = cast(pd.DataFrame, base[variaveis_da_parte(parte)])
    w = base["peso_final"].to_numpy()
    return {
        # lista, não dicionário: o registro grava as chaves em ordem alfabética
        "importancia_global": [
            {"variavel": v, "importancia": x} for v, x in importancia_global(final, X, w).items()
        ],
        "estabilidade": estabilidade(res.modelos, X, w),
    }


def _fora_da_regiao(
    parte: ParteModelo,
    rp: ReceitaParte,
    alvo: str,
    base: pd.DataFrame,
    extra: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Para cada região, a parte treinada nas outras quatro (sem sigmoide) e prevista nela.

    Devolve a previsão em `base` e, se houver `extra` (as células, na preferência, com
    `tem_pizza = 1`), nele também.
    """
    sem_sigmoide = rp.model_copy(update={"calibrar": False})
    p = np.full(len(base), np.nan)
    p_extra = None if extra is None else np.full(len(extra), np.nan)
    colunas = variaveis_da_parte(parte)
    for regiao in DOMINIOS["regiao"]:
        dentro = (base["regiao"] == regiao).to_numpy()
        if not dentro.any():
            continue
        print(f"região fora {parte}: {regiao}", flush=True)
        treino = cast(pd.DataFrame, base[~dentro])
        modelo = ajustar_parte(treino, alvo, parte, sem_sigmoide, np.empty(0))
        p[dentro] = modelo.predict_proba(cast(pd.DataFrame, base[dentro][colunas]))
        if extra is not None and p_extra is not None:
            fora_extra = (extra["regiao"] == regiao).to_numpy()
            if fora_extra.any():
                x = cast(pd.DataFrame, extra[fora_extra].assign(tem_pizza=1)[colunas])
                p_extra[fora_extra] = modelo.predict_proba(x)
    return _sem_nan(p, f"{parte} região fora"), p_extra


def _linhas_por_regiao(base: pd.DataFrame, alvo: str, p: np.ndarray) -> list[dict[str, Any]]:
    regiao = cast(pd.Series, base["regiao"])
    return metricas_por_segmento(
        base[alvo].to_numpy(), p, base["peso_final"].to_numpy(), {"regiao": regiao}
    )


def etapa_avaliacao_2008() -> None:
    """Avaliação do modelo em 2008-2009: segmentos, erros, explicabilidade e região fora.

    Lê a receita e os resultados salvos pelas etapas anteriores (conferindo dobras e vencedores)
    e grava `reports/avaliacao_2008.json`. Não muda a receita nem usa 2017-2018.
    """
    config: dict[str, Any] = {
        "o_que": "segmentos, contextos com mais erro, importância global, estabilidade e região "
        "fora, para a ocasião, a preferência (todas e com pizza) e o score",
        "previsoes": "fora da dobra do vencedor de cada parte, com a calibração decidida; o score "
        "é o salvo em previsoes_score (a calibração de H7 já aplicada)",
        "pesos": "peso_final em todas as métricas",
        "segmentos": "dia_semana, faixa_horario, regiao, faixa_etaria, sexo, faixa_renda, local",
        "erros": "contextos por regiao, dia_semana, faixa_horario e local, z = (O - E) / "
        "sqrt(soma de w² p (1 - p)), w com média 1; só contextos com 500 linhas ou mais",
        "importancia_global": "lista da maior para a menor; média ponderada de |contribuição| no modelo final, treinado em "
        "2008-2009 inteira (com a sigmoide, se a receita a usa)",
        "estabilidade": "contribuição média por nível nos modelos das dobras externas do vencedor "
        "(sem a sigmoide, que só reescala); instável se os sinais divergem e o maior valor "
        "absoluto passa de 0,05 log-chances",
        "regiao_fora": _DECLARACAO_REGIAO_FORA,
    }
    with _experimento("avaliacao_2008", "exploratório", config, RELATORIOS) as r:
        celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
        r.dados["dobras"] = _info_dobras(dobras)
        _conferir_dobras_de_h5(dobras)
        hash_dobras = r.dados["dobras"]["hash"]
        receita = carregar_receita(RELATORIOS / "receita.json")
        selecoes = _conferir_receita_e_selecoes(receita, dobras, hash_dobras)
        r.dados.update(
            celulas=_dados(celulas, ALVO_SCORE), ocasioes_refri=_dados(ocasioes, "y_guarana")
        )
        r.dados["receita"] = _relativo(RELATORIOS / "receita.json")
        r.configuracao["receita"] = {
            "ocasiao": receita.ocasiao.degrau,
            "preferencia": receita.preferencia.degrau,
            "score": receita.score,
            "calibrar": {
                "ocasiao": receita.ocasiao.calibrar,
                "preferencia": receita.preferencia.calibrar,
            },
            "calibrar_score": receita.calibrar_score,
        }
        rps: dict[Parte, ReceitaParte] = {
            "ocasiao": receita.ocasiao,
            "preferencia": receita.preferencia,
        }
        resultados: dict[Parte, ResultadoAninhado] = {}
        oof: dict[Parte, np.ndarray] = {}
        finais: dict[Parte, ModeloParte] = {}
        for parte in PARTES:
            base, alvo, _ = _base_da_parte(parte, celulas, ocasioes)
            rp = rps[parte]
            resultados[parte] = _carregar_aninhada(f"aninhada_{parte}_{rp.degrau}", hash_dobras)
            oof[parte] = _oof_decidido(resultados[parte], base, alvo, dobras, rp.calibrar)
            print(f"avaliação {parte}: modelo final", flush=True)
            finais[parte] = ajustar_parte(base, alvo, parte, rp, selecoes[parte]["oof"])

        # ocasião
        m_oc: dict[str, Any] = {
            "degrau": receita.ocasiao.degrau,
            "calibrado": receita.ocasiao.calibrar,
        }
        m_oc.update(_avaliar(celulas, "y_pizza_refri", oof["ocasiao"]))
        m_oc.update(_explicar(finais["ocasiao"], resultados["ocasiao"], celulas, "ocasiao"))
        r.metricas["ocasiao"] = m_oc

        # preferência: todas as ocasiões com refrigerante e o recorte com pizza
        com_pizza = (ocasioes["tem_pizza"] == 1).to_numpy()
        m_pref: dict[str, Any] = {
            "degrau": receita.preferencia.degrau,
            "calibrado": receita.preferencia.calibrar,
            "todas": _avaliar(ocasioes, "y_guarana", oof["preferencia"]),
            "com_pizza": _avaliar(
                cast(pd.DataFrame, ocasioes[com_pizza]), "y_guarana", oof["preferencia"][com_pizza]
            ),
        }
        m_pref.update(
            _explicar(finais["preferencia"], resultados["preferencia"], ocasioes, "preferencia")
        )
        r.metricas["preferencia"] = m_pref

        # score: o fora da dobra decidido em H6 e H7
        salvo = carregar_resultado("previsoes_score")
        if (
            salvo["hash_dobras"] != hash_dobras
            or salvo["tipo"] != receita.score
            or bool(salvo["calibrar_score"]) != receita.calibrar_score
        ):
            raise ValueError("previsoes_score.joblib não é o da receita: rode `make produto`")
        score = _sem_nan(np.asarray(salvo["score"], dtype=float), "score")
        m_score: dict[str, Any] = {"tipo": receita.score, "calibrado": receita.calibrar_score}
        m_score.update(_avaliar(celulas, ALVO_SCORE, score, posicao=True))
        res_direto: ResultadoAninhado | None = None
        final_direto: ModeloParte | None = None
        if receita.score == "direto" and receita.direto is not None:
            res_direto = _carregar_aninhada(f"aninhada_direto_{receita.direto.degrau}", hash_dobras)
            final_direto = ajustar_parte(
                celulas, ALVO_SCORE, "direto", receita.direto, selecoes["direto"]["oof"]
            )
            m_score["explicacao_do_direto"] = _explicar(final_direto, res_direto, celulas, "direto")
        else:
            m_score["explicacao"] = (
                "o score é o produto: ver a importância e a estabilidade de cada parte"
            )
        r.metricas["score"] = m_score

        # região fora (seção 6.4), sem sigmoide dos dois lados
        regiao: dict[str, Any] = {"declaracao": _DECLARACAO_REGIAO_FORA}
        p_oc, _ = _fora_da_regiao("ocasiao", receita.ocasiao, "y_pizza_refri", celulas)
        p_pref, p_pref_cel = _fora_da_regiao(
            "preferencia", receita.preferencia, "y_guarana", ocasioes, extra=celulas
        )
        if final_direto is not None and receita.direto is not None:
            p_score, _ = _fora_da_regiao("direto", receita.direto, ALVO_SCORE, celulas)
            cru_score = _sem_nan(
                np.asarray(cast(ResultadoAninhado, res_direto).oof, float), "direto"
            )
        else:
            p_score = p_oc * _sem_nan(cast(np.ndarray, p_pref_cel), "preferência nas células")
            cru_score = _sem_nan(
                np.asarray(resultados["ocasiao"].oof, float), "ocasião"
            ) * _sem_nan(
                np.asarray(resultados["preferencia"].oof_extras["celulas"], float),
                "preferência nas células",
            )
        for nome, base, alvo, fora, cru in (
            (
                "ocasiao",
                celulas,
                "y_pizza_refri",
                p_oc,
                np.asarray(resultados["ocasiao"].oof, float),
            ),
            (
                "preferencia",
                ocasioes,
                "y_guarana",
                p_pref,
                np.asarray(resultados["preferencia"].oof, float),
            ),
            ("score", celulas, ALVO_SCORE, p_score, cru_score),
        ):
            regiao[nome] = {
                "treinada_nas_outras_quatro": _linhas_por_regiao(base, alvo, fora),
                "fora_da_dobra": _linhas_por_regiao(base, alvo, _sem_nan(cru, nome)),
            }
        r.metricas["regiao_fora"] = regiao
    print("avaliação 2008-2009: relatório gravado", flush=True)
