"""Modelo servido: a receita congelada reajustada para a aplicação (seção 12 do design).

Só roda com `reports/teste_final.json` commitado. Os dados de treino vêm de H9
(`pos_teste.json`, `metricas["H9"]["treino_servido"]`): `"base_inteira"` junta 2008-2009 e
2017-2018; `"2017-2018"` usa só a edição recente. Não há indicador de edição nas variáveis.

Pesos: o `peso_final` de cada edição entra como está (as populações são diferentes, cada edição
soma a sua) e o modelo o normaliza para média 1 no conjunto de treino.

Calibração (das partes, se a receita a usa, e do score): sigmoide ajustada com `peso_final` só nas
previsões fora da dobra das linhas de 2017-2018, nas dobras externas de 2017-2018 do pós-teste
(`dobras_de_2017`). As linhas de 2008-2009 ficam sempre no treino e nunca são calibradas. A
população de referência e as UFs de mídia regional vêm só de 2017-2018.

Este modelo não tem teste temporal próprio: as métricas são as da receita treinada em 2008-2009
e testada uma vez em 2017-2018.
"""

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.bases.acesso import exigir_teste_commitado
from occasion.config import EDICAO_TESTE, EDICAO_TREINO, RAIZ
from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.experimentos.pos_teste import dobras_de_2017
from occasion.inference.artefato import ArtefatoModelo, _ufs_de_midia_regional
from occasion.inference.contract import VERSAO_CONTRATO, MetadadosModelo
from occasion.inference.treino import ALVO_DA_PARTE, ajustar_parte, descricao_do_degrau
from occasion.models.base import ModeloParte, logit, sigmoide
from occasion.models.calibracao import ModeloCalibrado
from occasion.receita import Receita, ReceitaParte
from occasion.score.score import construir_referencia
from occasion.validacao.dobras import dobra_das_linhas
from occasion.validacao.metricas import ajustar_sigmoide

__all__ = ["treinar_servido"]

_BASE_DA_PARTE: dict[ParteModelo, str] = {
    "ocasiao": "celulas",
    "direto": "celulas",
    "preferencia": "ocasioes_refri",
}
_SEM_CALIBRAR = np.zeros(0)  # só passa por `ajustar_parte` com `calibrar=False`


def _treino_servido(pos_teste: dict[str, Any]) -> str:
    valor = pos_teste["metricas"]["H9"]["treino_servido"]
    if valor not in ("base_inteira", "2017-2018"):
        raise ValueError(f"H9 com treino_servido inesperado: {valor!r}")
    return valor


def _ajustar(base: pd.DataFrame, parte: ParteModelo, rp: ReceitaParte) -> ModeloParte:
    """A parte treinada na base, sem calibração."""
    sem = rp.model_copy(update={"calibrar": False})
    return ajustar_parte(base, ALVO_DA_PARTE[parte], parte, sem, _SEM_CALIBRAR)


def _sigmoide_de(
    modelo: ModeloParte, p: np.ndarray, base17: pd.DataFrame, alvo: str
) -> ModeloParte:
    inclinacao, intercepto = ajustar_sigmoide(
        p, base17[alvo].to_numpy(dtype=float), base17["peso_final"].to_numpy(dtype=float)
    )
    return cast(ModeloParte, ModeloCalibrado(modelo, inclinacao, intercepto))


def _fora_da_dobra(
    partes: dict[ParteModelo, ReceitaParte],
    bases08: dict[str, pd.DataFrame],
    bases17: dict[str, pd.DataFrame],
    dobras: pd.DataFrame,
    usar_2008: bool,
) -> dict[str, np.ndarray]:
    """Previsões fora da dobra das linhas de 2017-2018 (e da preferência nas células)."""
    fold = {nome: dobra_das_linhas(b, dobras) for nome, b in bases17.items()}
    oof = {parte: np.full(len(bases17[_BASE_DA_PARTE[parte]]), np.nan) for parte in partes}
    oof["preferencia_nas_celulas"] = np.full(len(bases17["celulas"]), np.nan)
    for k in range(int(dobras["dobra"].to_numpy().max()) + 1):
        modelos: dict[ParteModelo, ModeloParte] = {}
        for parte, rp in partes.items():
            nome = _BASE_DA_PARTE[parte]
            fora = fold[nome] == k
            treino = bases17[nome][~fora]
            if usar_2008:
                treino = pd.concat([bases08[nome], treino], ignore_index=True)
            modelos[parte] = _ajustar(cast(pd.DataFrame, treino), parte, rp)
            X = cast(pd.DataFrame, bases17[nome].loc[fora, variaveis_da_parte(parte)])
            oof[parte][fora] = modelos[parte].predict_proba(X)
        cel = fold["celulas"] == k
        X_cel = cast(pd.DataFrame, bases17["celulas"].loc[cel]).assign(tem_pizza=1)
        X_cel = cast(pd.DataFrame, X_cel[variaveis_da_parte("preferencia")])
        oof["preferencia_nas_celulas"][cel] = modelos["preferencia"].predict_proba(X_cel)
    for nome, v in oof.items():
        if np.isnan(v).any():
            raise ValueError(f"previsão fora da dobra com NaN em {nome}")
    return oof


def _metricas_da_receita(teste_final: dict[str, Any]) -> dict[str, float]:
    m = teste_final["metricas"]
    score = m["resumos"]["score"]
    r = score["receita"]
    return {
        "pr_auc": float(r["pr_auc"]),
        "log_loss": float(r["log_loss"]),
        "roc_auc": float(r["roc_auc"]),
        "brier": float(r["brier"]),
        "pr_auc_degrau_a": float(score["degrau_a"]["pr_auc"]),
    }


def _descricao(treino: str) -> str:
    dados = "2008-2009 e 2017-2018" if treino == "base_inteira" else "2017-2018"
    return (
        "Métricas do score da receita treinada em 2008-2009 e testada uma vez em 2017-2018 "
        "(reports/teste_final.json), com a PR-AUC do degrau A (tabela de taxas) ao lado. O "
        "modelo servido não tem teste temporal próprio: foi reajustado com a mesma receita em "
        f"{dados}, escolha de H9 (validação por setores dentro de 2017-2018, em "
        "reports/pos_teste.json). Guaraná é o sabor registrado, não uma marca."
    )


def treinar_servido(
    receita: Receita,
    bases: dict[str, pd.DataFrame],
    pos_teste: dict[str, Any],
    teste_final: dict[str, Any],
    commit: str,
    raiz: Path = RAIZ,
) -> ArtefatoModelo:
    """Reajusta a receita para a aplicação.

    `bases` traz `celulas_<edicao>` e `ocasioes_refri_<edicao>` das duas edições; `pos_teste` e
    `teste_final` são os JSONs de `reports/` já lidos.
    """
    exigir_teste_commitado(raiz)
    treino = _treino_servido(pos_teste)
    usar_2008 = treino == "base_inteira"
    bases08 = {n: bases[f"{n}_{EDICAO_TREINO}"] for n in ("celulas", "ocasioes_refri")}
    bases17 = {n: bases[f"{n}_{EDICAO_TESTE}"] for n in ("celulas", "ocasioes_refri")}
    partes: dict[ParteModelo, ReceitaParte] = {
        "ocasiao": receita.ocasiao,
        "preferencia": receita.preferencia,
    }
    if receita.direto is not None:
        partes["direto"] = receita.direto

    treinos: dict[str, pd.DataFrame] = {
        n: pd.concat([bases08[n], bases17[n]], ignore_index=True) if usar_2008 else bases17[n]
        for n in bases17
    }
    modelos = {p: _ajustar(treinos[_BASE_DA_PARTE[p]], p, rp) for p, rp in partes.items()}

    precisa_oof = receita.calibrar_score or any(rp.calibrar for rp in partes.values())
    celulas17 = bases17["celulas"]
    oof: dict[str, np.ndarray] = {}
    if precisa_oof:
        dobras, _ = dobras_de_2017(celulas17, bases17["ocasioes_refri"])
        oof = _fora_da_dobra(partes, bases08, bases17, dobras, usar_2008)
        alvo = {"ocasiao": celulas17, "direto": celulas17, "preferencia": bases17["ocasioes_refri"]}
        for parte, rp in partes.items():
            if rp.calibrar:
                modelos[parte] = _sigmoide_de(
                    modelos[parte], oof[parte], alvo[parte], ALVO_DA_PARTE[parte]
                )

    calibracao_score: tuple[float, float] | None = None
    if receita.calibrar_score:

        def calibrada(parte: ParteModelo, p: np.ndarray) -> np.ndarray:
            m = modelos[parte]
            if isinstance(m, ModeloCalibrado):
                return sigmoide(m.inclinacao * logit(p) + m.intercepto)
            return p

        if "direto" in partes:
            produto = calibrada("direto", oof["direto"])
        else:
            produto = calibrada("ocasiao", oof["ocasiao"]) * calibrada(
                "preferencia", oof["preferencia_nas_celulas"]
            )
        calibracao_score = ajustar_sigmoide(
            produto,
            celulas17[ALVO_DA_PARTE["direto"]].to_numpy(dtype=float),
            celulas17["peso_final"].to_numpy(dtype=float),
        )
        if not calibracao_score[0] > 0:
            raise ValueError("inclinação da calibração do score não positiva")

    modelo_direto = modelos.get("direto")
    referencia = construir_referencia(
        celulas17, modelos["ocasiao"], modelos["preferencia"], calibracao_score, modelo_direto
    )
    metadados = MetadadosModelo(
        modelo="servido",
        degrau=descricao_do_degrau(receita),
        treinado_em=[EDICAO_TREINO, EDICAO_TESTE] if usar_2008 else [EDICAO_TESTE],
        receita_commit=commit,
        metricas_da_receita=_metricas_da_receita(teste_final),
        descricao_metricas=_descricao(treino),
        instante_de_previsao="planejamento, antes da ocasião",
        aviso="Guaraná é o sabor registrado na POF, não uma marca.",
        versao_contrato=VERSAO_CONTRATO,
    )
    return ArtefatoModelo(
        modelo_p1=modelos["ocasiao"],
        modelo_p2=modelos["preferencia"],
        referencia=referencia,
        ufs_midia_regional=_ufs_de_midia_regional(
            celulas17, modelos["ocasiao"], modelos["preferencia"]
        ),
        metadados=metadados,
        calibracao_score=calibracao_score,
        modelo_direto=modelo_direto,
    )
