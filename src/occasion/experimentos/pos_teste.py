"""Diagnóstico depois do teste (seção 6.8 do design) e H9 (pré-registro).

Só roda com o resultado do teste final commitado (`exigir_teste_commitado`) e a receita intacta.
A receita congelada (degraus e parâmetros de `reports/receita.json`, sem nova busca) é treinada em
validação cruzada dentro de 2017-2018 em dois braços:

- **recente:** só as dobras de treino de 2017-2018;
- **inteira:** as mesmas mais toda a 2008-2009, cujas linhas ficam sempre no treino (nas dobras
  externas e nas internas) e nunca são avaliadas.

As dobras são sorteadas só sobre os setores de 2017-2018 (a chave `(edicao, cod_upa)` fica
implícita, porque 2008-2009 nunca entra na tabela de setores), com o mesmo critério de 10% a 30%
dos positivos de cada alvo (`escolher_dobras`). O peso entra no treino como a receita manda e é
normalizado para média 1 dentro de cada conjunto de treino pelo próprio modelo; os pesos das duas
edições são de populações diferentes (cada uma soma a sua), e na base inteira entram como estão.

A calibração, se a receita a usar, é ajustada com as previsões fora da dobra interna só das linhas
de 2017-2018 (nos dois braços), com `peso_final`: a mesma regra do modelo servido (seção 12).
A receita atual não calibra, mas o ramo existe e é testado.

**Diagnóstico 6.8 (exploratório).** O braço recente ao lado do teste final: a diferença é o que se
perde ao atravessar dez anos e a troca de método de coleta, efeito conjunto que o projeto não
separa. Nenhum número já relatado muda.

**H9 (confirmatório).** Ganho do braço recente sobre a base inteira no log loss ponderado
(positivo favorece o braço recente), em três comparações (ocasião, preferência e recorte de pizza
da preferência; mais o modelo direto, se H6 o promoveu). `padrao` decide cada uma com a base
inteira como padrão; se qualquer uma recusar o padrão, `treino_servido` é `"2017-2018"` para as
duas partes; senão, `"base_inteira"`. O intervalo é o bootstrap de setores de 2017-2018 dentro da
UF (B = 1000). Os nós das curvas saem sem peso (declarado, não corrigido).
"""

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from occasion.bases.acesso import (
    carregar_base,
    exigir_receita_intacta,
    exigir_teste_commitado,
)
from occasion.config import (
    DADOS_PROCESSADOS,
    EDICAO_TESTE,
    EDICAO_TREINO,
    RAIZ,
    SEMENTE,
)
from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.experimentos.etapas import experimento, hash_das_dobras
from occasion.experimentos.regras import padrao
from occasion.inference.treino import ALVO_DA_PARTE
from occasion.models.base import ModeloParte
from occasion.models.calibracao import ModeloCalibrado
from occasion.models.escada import construir
from occasion.receita import Receita, ReceitaParte, carregar_receita
from occasion.validacao.bootstrap import B_PADRAO, intervalo_ganho
from occasion.validacao.dobras import (
    K_EXTERNO,
    dobra_das_linhas,
    dobras_internas,
    escolher_dobras,
    tabela_de_upas,
)
from occasion.validacao.metricas import ajustar_sigmoide, log_loss_ponderado, resumo

Braco = Literal["recente", "inteira"]
BRACOS: tuple[Braco, ...] = ("recente", "inteira")
NOME = "pos_teste"
_DIFERENCAS = ("log_loss", "brier", "pr_auc", "roc_auc")
_NOTA_PESO = (
    "o peso de treino é peso_final normalizado para média 1 dentro de cada conjunto de treino; "
    "os pesos de 2008-2009 e de 2017-2018 são de populações diferentes (cada edição soma a sua) "
    "e, na base inteira, entram como estão"
)
_DECLARACAO_QUEDA = (
    "a diferença entre o braço recente (validação cruzada dentro de 2017-2018) e o teste final é "
    "o efeito conjunto de dez anos e da mudança do método de coleta (registro alimentar em "
    "2008-2009, recordatório de 24 horas em 2017-2018); o projeto não separa os dois"
)
VALIDACAO_DE_H9 = (
    "H9 é validação cruzada por setores dentro de 2017-2018 (não é validação temporal); a "
    "avaliação temporal da receita (treino em 2008-2009, teste em 2017-2018) é H8, em "
    "reports/teste_final.json"
)
_BASE_DA_PARTE: dict[ParteModelo, Literal["celulas", "ocasioes"]] = {
    "ocasiao": "celulas",
    "direto": "celulas",
    "preferencia": "ocasioes",
}

__all__ = ["executar", "rodar_pos_teste", "treino_servido"]


# ---------------------------------------------------------------------------------- um braço


def _ajustar(
    parte: ParteModelo, receita_parte: ReceitaParte, alvo: str, base: pd.DataFrame
) -> ModeloParte:
    """Treina a parte com os parâmetros congelados na `base` (sem calibração)."""
    modelo = construir(receita_parte.degrau, parte, receita_parte.parametros)
    X = cast(pd.DataFrame, base[variaveis_da_parte(parte)])
    peso = cast(pd.Series, base["peso_final"])
    modelo.fit(X, cast(pd.Series, base[alvo]), peso if receita_parte.usar_peso else None)
    return modelo


def _treino(braco: Braco, base08: pd.DataFrame, treino17: pd.DataFrame) -> pd.DataFrame:
    if braco == "recente":
        return treino17.reset_index(drop=True)
    return pd.concat([base08, treino17], ignore_index=True)


def _ajustar_braco(
    braco: Braco,
    parte: ParteModelo,
    receita_parte: ReceitaParte,
    alvo: str,
    base08: pd.DataFrame,
    treino17: pd.DataFrame,
    interna: pd.DataFrame,
) -> ModeloParte:
    """O modelo da parte em um braço, com os setores de treino de 2017-2018 em `treino17`.

    `interna` é a tabela de dobras internas desses setores. As linhas de 2008-2009 (braço
    `inteira`) entram em todo ajuste e nunca são avaliadas; a calibração, se houver, usa só as
    previsões fora da dobra interna das linhas de 2017-2018.
    """
    modelo = _ajustar(parte, receita_parte, alvo, _treino(braco, base08, treino17))
    if not receita_parte.calibrar:
        return modelo
    por_linha = dobra_das_linhas(treino17, interna)
    k_interno = int(por_linha.max()) + 1
    p = np.full(len(treino17), np.nan)
    for j in range(k_interno):
        fora = por_linha == j
        dentro = _treino(braco, base08, cast(pd.DataFrame, treino17[~fora]))
        interno = _ajustar(parte, receita_parte, alvo, dentro)
        X = cast(pd.DataFrame, treino17.loc[fora, variaveis_da_parte(parte)])
        p[fora] = interno.predict_proba(X)
    inclinacao, intercepto = ajustar_sigmoide(
        p,
        treino17[alvo].to_numpy(dtype=float),
        treino17["peso_final"].to_numpy(dtype=float),  # a calibração mira a população
    )
    return cast(ModeloParte, ModeloCalibrado(modelo, inclinacao, intercepto))


# ---------------------------------------------------------------------------------- regra de H9


def treino_servido(vereditos: Sequence[str]) -> Literal["2017-2018", "base_inteira"]:
    """Regra pré-registrada: qualquer comparação que recuse o padrão leva o servido a 2017-2018."""
    recusou = any(v == "padrão refutado" for v in vereditos)
    return "2017-2018" if recusou else "base_inteira"


# ---------------------------------------------------------------------------------- o experimento


def _partes_da_receita(receita: Receita) -> dict[ParteModelo, ReceitaParte]:
    partes: dict[ParteModelo, ReceitaParte] = {
        "ocasiao": receita.ocasiao,
        "preferencia": receita.preferencia,
    }
    if receita.direto is not None:
        partes["direto"] = receita.direto
    return partes


def _com_pizza(ocasioes: pd.DataFrame) -> pd.DataFrame:
    return cast(pd.DataFrame, ocasioes[ocasioes["tem_pizza"] == 1])


def executar(
    celulas08: pd.DataFrame,
    ocasioes08: pd.DataFrame,
    celulas17: pd.DataFrame,
    ocasioes17: pd.DataFrame,
    receita: Receita,
    teste_final: dict[str, Any],
    b: int = B_PADRAO,
) -> dict[str, Any]:
    """Calcula o diagnóstico 6.8 e H9; devolve o dicionário que vai para `reports/pos_teste.json`.

    Não toca o disco. As bases de 2008-2009 só entram no treino; as de 2017-2018 trazem o alvo.
    """
    if receita.calibrar_score:
        raise NotImplementedError(
            "receita.calibrar_score = True: o diagnóstico do score não trata a calibração do "
            "score (a receita congelada não a usa)"
        )
    fora = set(ocasioes17["cod_upa"]) - set(celulas17["cod_upa"])
    if fora:
        raise ValueError(f"setores das ocasiões ausentes das células: {sorted(fora)[:10]}")
    alvos = {
        "y_pizza_refri": (celulas17, "y_pizza_refri"),
        "y_pizza_guarana": (celulas17, "y_pizza_guarana"),
        "y_guarana": (ocasioes17, "y_guarana"),
        "y_guarana_com_pizza": (_com_pizza(ocasioes17), "y_guarana"),
    }
    dobras, registro_dobras = escolher_dobras(tabela_de_upas(celulas17), alvos)
    fold_cel = dobra_das_linhas(celulas17, dobras)
    fold_oc = dobra_das_linhas(ocasioes17, dobras)
    partes = _partes_da_receita(receita)
    base08 = {"celulas": celulas08, "ocasioes": ocasioes08}
    base17 = {"celulas": celulas17, "ocasioes": ocasioes17}
    fold17 = {"celulas": fold_cel, "ocasioes": fold_oc}

    # previsões fora da dobra: braço -> nome -> vetor alinhado às linhas de 2017-2018
    oof: dict[str, dict[str, np.ndarray]] = {
        braco: {
            "ocasiao": np.full(len(celulas17), np.nan),
            "preferencia": np.full(len(ocasioes17), np.nan),
            "preferencia_nas_celulas": np.full(len(celulas17), np.nan),
            "direto": np.full(len(celulas17), np.nan),
        }
        for braco in BRACOS
    }
    for k in range(K_EXTERNO):
        interna = dobras_internas(dobras, k)
        for braco in BRACOS:
            modelos: dict[str, ModeloParte] = {}
            for parte, rp in partes.items():
                tipo = _BASE_DA_PARTE[parte]
                tem_treino = fold17[tipo] != k
                treino17 = cast(pd.DataFrame, base17[tipo][tem_treino])
                modelos[parte] = _ajustar_braco(
                    braco, parte, rp, ALVO_DA_PARTE[parte], base08[tipo], treino17, interna
                )
                avaliada = ~tem_treino
                X = cast(pd.DataFrame, base17[tipo].loc[avaliada, variaveis_da_parte(parte)])
                oof[braco][parte][avaliada] = modelos[parte].predict_proba(X)
            # preferência nas células de teste, com `tem_pizza` fixado em sim (como no score)
            cel_av = fold_cel == k
            X_cel = cast(pd.DataFrame, celulas17.loc[cel_av]).assign(tem_pizza=1)
            X_cel = cast(pd.DataFrame, X_cel[variaveis_da_parte("preferencia")])
            oof[braco]["preferencia_nas_celulas"][cel_av] = modelos["preferencia"].predict_proba(
                X_cel
            )
        print(f"pos-teste: dobra {k + 1}/{K_EXTERNO} pronta", flush=True)

    for braco in BRACOS:
        for nome, v in oof[braco].items():
            if nome == "direto" and "direto" not in partes:
                continue
            if np.isnan(v).any():
                raise ValueError(f"previsão fora da dobra com NaN em {braco}/{nome}")

    def score(braco: Braco) -> np.ndarray:
        if receita.score == "direto":
            return oof[braco]["direto"]
        return oof[braco]["ocasiao"] * oof[braco]["preferencia_nas_celulas"]

    # as quatro leituras: nome -> (base de 2017-2018, alvo, previsão por braço, linhas usadas)
    pizza = (ocasioes17["tem_pizza"] == 1).to_numpy()
    todas_linhas = slice(None)
    leituras: dict[str, tuple[pd.DataFrame, str, dict[str, np.ndarray], Any]] = {
        "ocasiao": (celulas17, "y_pizza_refri", {x: oof[x]["ocasiao"] for x in BRACOS}, todas_linhas),
        "preferencia": (ocasioes17, "y_guarana", {x: oof[x]["preferencia"] for x in BRACOS}, todas_linhas),
        "preferencia_com_pizza": (ocasioes17, "y_guarana", {x: oof[x]["preferencia"] for x in BRACOS}, pizza),
        "score": (celulas17, "y_pizza_guarana", {x: score(x) for x in BRACOS}, todas_linhas),
    }  # fmt: skip
    if "direto" in partes:
        leituras["direto"] = (
            celulas17, "y_pizza_guarana", {x: oof[x]["direto"] for x in BRACOS}, todas_linhas,
        )  # fmt: skip

    def partes_da_leitura(nome: str) -> tuple[np.ndarray, ...]:
        base, alvo, previsoes, linhas = leituras[nome]
        sub = cast(pd.DataFrame, base.loc[linhas] if isinstance(linhas, np.ndarray) else base)
        return (
            sub[alvo].to_numpy(),
            previsoes["recente"][linhas],
            previsoes["inteira"][linhas],
            sub["peso_final"].to_numpy(),
            sub["cod_upa"].to_numpy(),
            sub["uf"].to_numpy(),
        )

    # H9
    comparacoes: dict[str, Any] = {}
    nomes_h9 = ["ocasiao", "preferencia", "preferencia_com_pizza"] + (
        ["direto"] if "direto" in partes else []
    )
    for nome in nomes_h9:
        y, p_rec, p_int, w, upa, uf = partes_da_leitura(nome)
        ic = intervalo_ganho(log_loss_ponderado, False, y, p_rec, p_int, w, upa, uf, b, SEMENTE)
        comparacoes[nome] = {
            "ganho": ic,
            "log_loss_recente": log_loss_ponderado(y, p_rec, w),
            "log_loss_inteira": log_loss_ponderado(y, p_int, w),
            "n": len(y),
            "positivos": int(np.sum(y == 1)),
            "veredito": padrao(ic),
        }
        print(f"pos-teste: H9 {nome}: {comparacoes[nome]['veredito']}", flush=True)
    decisao = treino_servido([c["veredito"] for c in comparacoes.values()])

    # diagnóstico 6.8
    resumos_teste = teste_final["metricas"]["resumos"]
    queda = teste_final["metricas"]["queda"]
    diag: dict[str, Any] = {}
    for nome in ("ocasiao", "preferencia_todas", "preferencia_com_pizza", "score"):
        origem = "preferencia" if nome == "preferencia_todas" else nome
        y, p_rec, p_int, w, _, _ = partes_da_leitura(origem)
        recente = resumo(y, p_rec, w)
        teste = resumos_teste[nome]["receita"]
        diferenca = {
            m: (recente[m] - teste[m]) if recente.get(m) is not None and teste.get(m) is not None else None
            for m in _DIFERENCAS
        }  # fmt: skip
        diag[nome] = {
            "recente": recente,
            "base_inteira": resumo(y, p_int, w),
            "teste_final": {m: teste.get(m) for m in _DIFERENCAS},
            "validacao_2008": queda[nome]["receita"]["validacao"],
            "diferenca": diferenca,
        }
    return {
        "H9": {
            "natureza": "confirmatório",
            "validacao": VALIDACAO_DE_H9,
            "enunciado": "treinar com 2008-2009 somado não piora a previsão de 2017-2018 em "
            "relação a treinar só com 2017-2018",
            "padrao": "base inteira",
            "metrica": "ganho do braço recente sobre a base inteira no log loss ponderado "
            "(positivo favorece o braço recente)",
            "comparacoes": comparacoes,
            "regra": "se qualquer comparação recusar o padrão, treino_servido = 2017-2018 nas "
            "duas partes; senão, base_inteira",
            "treino_servido": decisao,
        },
        "diagnostico_6_8": {
            "natureza": "exploratório",
            "declaracao": _DECLARACAO_QUEDA,
            "partes": diag,
        },
        "_dobras": {"tabela": dobras, "registro": registro_dobras},
    }


def rodar_pos_teste(
    raiz: Path = RAIZ,
    pasta_bases: Path = DADOS_PROCESSADOS,
    pasta_relatorios: Path | None = None,
    b: int = B_PADRAO,
) -> Path:
    """Confere as travas, roda `executar` nas bases reais e grava `reports/pos_teste.json`."""
    pasta = raiz / "reports" if pasta_relatorios is None else pasta_relatorios
    # a trava confere o mesmo arquivo que é lido
    exigir_teste_commitado(raiz, (pasta / "teste_final.json").relative_to(raiz).as_posix())
    assinatura = exigir_receita_intacta(raiz)
    receita = carregar_receita(pasta / "receita.json")
    teste_final = json.loads((pasta / "teste_final.json").read_text(encoding="utf-8"))
    config: dict[str, Any] = {
        "hipotese": "H9 e diagnóstico 6.8",
        "receita": "congelada: parâmetros de reports/receita.json, sem nova busca",
        "assinatura_receita": assinatura,
        "bracos": {
            "recente": "só as dobras de treino de 2017-2018",
            "inteira": "as mesmas mais toda a 2008-2009, sempre no treino",
        },
        "dobras": "setores de 2017-2018, critério de 10% a 30% dos positivos dos quatro alvos; "
        "a chave (edicao, cod_upa) deixa 2008-2009 sempre no treino",
        "semente_dobras_internas": "SEMENTE + k para a dobra externa k (dobras_internas)",
        "bootstrap": {
            "semente": SEMENTE,
            "b": b,
            "unidade": "cod_upa de 2017-2018",
            "estrato": "uf",
            "nivel": 0.95,
        },
        "peso": _NOTA_PESO,
        "calibracao": {
            "ocasiao": receita.ocasiao.calibrar,
            "preferencia": receita.preferencia.calibrar,
            "regra": "sigmoide ajustada nas previsões fora da dobra interna só das linhas de "
            "2017-2018, com peso_final",
            "score": "calibrar_score da receita não entra: H9 compara as partes",
        },
        "score": receita.score,
        "nos_das_curvas": "sem peso (o sample_weight não chega ao SplineTransformer)",
        "linguagem": _DECLARACAO_QUEDA,
    }
    with experimento(NOME, "confirmatório", config, pasta) as r:
        inicio = time.time()
        celulas08 = carregar_base("celulas", EDICAO_TREINO, pasta=pasta_bases, raiz=raiz)
        ocasioes08 = carregar_base("ocasioes_refri", EDICAO_TREINO, pasta=pasta_bases, raiz=raiz)
        celulas17 = carregar_base("celulas", EDICAO_TESTE, pasta=pasta_bases, raiz=raiz)
        ocasioes17 = carregar_base("ocasioes_refri", EDICAO_TESTE, pasta=pasta_bases, raiz=raiz)
        r.dados.update(
            treino_2008=len(celulas08), ocasioes_2008=len(ocasioes08),
            celulas_2017=len(celulas17), ocasioes_2017=len(ocasioes17),
        )  # fmt: skip
        resultado = executar(celulas08, ocasioes08, celulas17, ocasioes17, receita, teste_final, b)
        dobras = resultado.pop("_dobras")
        r.semente = int(dobras["registro"]["semente"])
        r.configuracao["tentativas_de_dobras"] = len(dobras["registro"]["tentativas"])
        r.dados["hash_dobras"] = hash_das_dobras(dobras["tabela"])
        r.metricas.update(resultado)
        r.metricas["segundos"] = round(time.time() - inicio, 1)
    print(f"pos-teste: treino_servido = {resultado['H9']['treino_servido']}", flush=True)
    return pasta / f"{NOME}.json"
