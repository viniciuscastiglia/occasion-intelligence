"""Etapas dos experimentos do Plano 2 sobre 2008-2009: dobras, peso (H5), escada, H4 e controle.

Cada etapa lê as bases por `carregar_base` (só `EDICAO_TREINO`; nada de 2017-2018 passa por
aqui), grava o relatório com `registrar` em `reports/` e os objetos pesados com `salvar_resultado`
em `data/experimentos/`. As pastas são lidas dos nomes deste módulo na hora da chamada, para o
teste poder redirecioná-las.

Resultados aninhados salvos (`salvar_resultado`):
- `aninhada_<parte>_B_com_peso` e `aninhada_<parte>_B_sem_peso`: as duas versões de H5;
- `aninhada_<parte>_<degrau>` (A a D): os degraus da escada; o B é o B de H5 com o peso decidido.
"""

import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from occasion.bases.acesso import carregar_base
from occasion.config import (
    DADOS_EXPERIMENTOS,
    DADOS_PROCESSADOS,
    EDICAO_TREINO,
    RELATORIOS,
    SEMENTE,
)
from occasion.experimentos.aninhada import ResultadoAninhado, selecao_final, validacao_aninhada
from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.experimentos.registro import carregar_resultado, registrar, salvar_resultado
from occasion.experimentos.regras import (
    melhor_logistico,
    padrao,
    sinal_negativo,
    superioridade,
    vencedor_escada,
)
from occasion.models.escada import Degrau
from occasion.models.logistica import ModeloLogistico
from occasion.validacao.bootstrap import B_PADRAO, Intervalo, intervalo_estatistica, intervalo_ganho
from occasion.validacao.dobras import dobra_das_linhas, escolher_dobras, tabela_de_upas
from occasion.validacao.metricas import log_loss_ponderado, resumo

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
    h5 = json.loads((_pasta_relatorios() / "h5_peso.json").read_text(encoding="utf-8"))
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


def _sem_nan(p: np.ndarray, o_que: str) -> np.ndarray:
    if np.isnan(p).any():
        raise ValueError(f"previsão fora da dobra com NaN em {o_que}")
    return p


# ---------------------------------------------------------------------------------- dobras


def etapa_dobras() -> None:
    """Dobras externas por setor, conferidas nos quatro alvos; grava parquet e `dobras.json`."""
    celulas, ocasioes = _celulas(), _ocasioes()
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
    registrar(
        "dobras",
        "exploratório",
        {
            "k_externo": 5,
            "unidade": "cod_upa, espalhado por estrato",
            "tentativas_feitas": len(registro_dobras["tentativas"]),
            "semente_inicial": SEMENTE,
        },
        {
            "semente_usada": semente,
            "conferencia": escolhida["resultado"],
            "positivos_por_dobra": {
                nome: v["positivos_por_dobra"]
                for nome, v in escolhida["resultado"].items()
                if nome != "geral"
            },
            "tentativas": registro_dobras["tentativas"],
        },
        {
            "celulas": _dados(celulas, "y_pizza_refri"),
            "ocasioes_refri": _dados(ocasioes, "y_guarana"),
        },
        pasta=RELATORIOS,
        semente=semente,
    )
    print(
        f"dobras: semente {semente}, {len(registro_dobras['tentativas'])} tentativa(s)", flush=True
    )


# ---------------------------------------------------------------------------------- H5


def etapa_peso() -> None:
    """H5: o degrau B com e sem `peso_final` no treino; o padrão é treinar com peso."""
    celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
    metricas: dict[str, Any] = {}
    dados: dict[str, Any] = {}
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
        salvar_resultado(f"aninhada_{parte}_B", resultados[usar_peso])
        y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
        metricas[parte] = {
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
        dados[parte] = _dados(base, alvo)
        print(f"H5 {parte}: usar_peso={usar_peso} ({veredito})", flush=True)
    registrar(
        "h5_peso",
        "confirmatório",
        {
            "degrau": "B",
            "metrica": "log loss ponderado fora da dobra",
            "referencia": "treino com peso_final (o padrão)",
            "alternativa": "treino sem peso",
            "bootstrap": {"b": B_PADRAO, "estrato": "uf"},
            "resultados_salvos": "aninhada_<parte>_B_com_peso, aninhada_<parte>_B_sem_peso, "
            "aninhada_<parte>_B (o decidido)",
        },
        metricas,
        dados,
        pasta=_pasta_relatorios(),
    )


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
    celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
    base, alvo, extras = _base_da_parte(parte, celulas, ocasioes)
    usar_peso = _usar_peso_decidido()[parte]
    h5 = json.loads((_pasta_relatorios() / "h5_peso.json").read_text(encoding="utf-8"))
    tempo_b = h5["metricas"][parte]["tempo_s"]["com_peso" if usar_peso else "sem_peso"]

    resultados: dict[str, ResultadoAninhado] = {}
    tempos: dict[str, float] = {}
    for degrau in ("A", "B", "C", "D"):
        if degrau == "B":
            print(f"escada {parte}: B reaproveitado de H5", flush=True)
            resultados["B"] = carregar_resultado(f"aninhada_{parte}_B")
            tempos["B"] = float(tempo_b)
            continue
        print(f"escada {parte}: degrau {degrau}", flush=True)
        t0 = time.perf_counter()
        resultados[degrau] = validacao_aninhada(
            base, alvo, cast(Degrau, degrau), cast(ParteModelo, parte), dobras,
            usar_peso=usar_peso, bases_extras=extras, progresso=_progresso(parte, degrau),
        )  # fmt: skip
        tempos[degrau] = time.perf_counter() - t0
        salvar_resultado(f"aninhada_{parte}_{degrau}", resultados[degrau])
        print(f"escada {parte}: degrau {degrau} em {tempos[degrau]:.0f} s", flush=True)

    oof = {d: _sem_nan(r.oof, f"{parte} degrau {d}") for d, r in resultados.items()}
    cache: dict[tuple[str, str], Intervalo] = {}

    def comparar(candidato: str, referencia: str) -> Intervalo:
        if (candidato, referencia) not in cache:
            cache[(candidato, referencia)] = _ic_ganho(base, alvo, oof[candidato], oof[referencia])
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
    metricas = {
        "degraus": {
            d: {
                **_resumos(base, alvo, oof[d], parte),
                "parametros_por_dobra": resultados[d].parametros,
                "tempo_s": tempos[d],
            }
            for d in ("A", "B", "C", "D")
        },
        "comparacoes": {"H1": h1, "H2": h2, "H3": h3},
        "comparacoes_sequenciais": sequencial,
        "vencedor": vencedor,
    }
    registrar(
        f"escada_{parte}",
        "confirmatório",
        {
            "parte": parte,
            "alvo": alvo,
            "base": _BASE_E_ALVO[parte][0],
            "usar_peso": usar_peso,
            "metrica_da_comparacao": "log loss ponderado fora da dobra (ganho positivo favorece o candidato)",
            "bootstrap": {"b": B_PADRAO, "estrato": "uf", "unidade": "cod_upa"},
            "b_de_h5": "reaproveitado de aninhada_<parte>_B",
            "resultados_salvos": f"aninhada_{parte}_<degrau>",
            "tempo_b_s": "tempo do B medido em H5",
        },
        metricas,
        _dados(base, alvo),
        pasta=_pasta_relatorios(),
    )


# ---------------------------------------------------------------------------------- H4


def etapa_h4(b: int = B_PADRAO) -> None:
    """H4: sinal do coeficiente de `tem_pizza` no B da preferência, com bootstrap de setores."""
    celulas, ocasioes, dobras = _celulas(), _ocasioes(), _dobras()
    usar_peso = _usar_peso_decidido()["preferencia"]
    print("H4: seleção final do B da preferência", flush=True)
    params, _, _ = selecao_final(
        ocasioes, "y_guarana", "B", "preferencia", dobras,
        usar_peso=usar_peso, bases_extras={"celulas": celulas},
    )  # fmt: skip
    C = float(params["C"])
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
    registrar(
        "h4_tem_pizza",
        "confirmatório",
        {
            "degrau": "B",
            "parte": "preferencia",
            "C": C,
            "usar_peso": usar_peso,
            "bootstrap": {"b": b, "estrato": "uf", "unidade": "cod_upa"},
            "pesos_na_replica": "(peso_final se usar_peso, senão 1) x multiplicador, média 1",
            "aviso": _AVISO_H4,
        },
        {
            "coeficiente_tem_pizza": ic.estimativa,
            "intervalo": ic,
            "veredito": sinal_negativo(ic),
            "razao_de_chances": float(np.exp(ic.estimativa)),
        },
        _dados(ocasioes, "y_guarana"),
        pasta=_pasta_relatorios(),
    )


# ---------------------------------------------------------------------------------- embaralhado


def etapa_embaralhado(
    partes: Sequence[Parte] = PARTES, usar_peso: Mapping[str, bool] | None = None
) -> None:
    """Controle de vazamento: B com o alvo permutado contra a prevalência ponderada do treino.

    `usar_peso` fixa o uso de peso por parte; sem ele, vale o decidido em H5.
    """
    dobras = _dobras()
    celulas, ocasioes = _celulas(), _ocasioes()
    uso = dict(usar_peso) if usar_peso is not None else _usar_peso_decidido()
    metricas: dict[str, Any] = {}
    dados: dict[str, Any] = {}
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
        metricas[parte] = {
            "veredito": superioridade(ic),
            "ganho_do_modelo_sobre_a_constante": ic,
            "log_loss_modelo": log_loss_ponderado(y, modelo, w),
            "log_loss_constante": log_loss_ponderado(y, constante, w),
            "usar_peso": uso[parte],
        }
        dados[parte] = _dados(base, alvo)
        print(f"embaralhado {parte}: {metricas[parte]['veredito']}", flush=True)
    registrar(
        "rotulos_embaralhados",
        "exploratório",
        {
            "controle": "vazamento (não é uma hipótese)",
            "degrau": "B",
            "alvo": "permutado entre as linhas, com np.random.default_rng(SEMENTE)",
            "referencia": "constante igual à prevalência ponderada (peso_final) do treino de cada dobra",
            "esperado": "refutada: o modelo não pode superar a constante",
        },
        metricas,
        dados,
        pasta=_pasta_relatorios(),
    )
