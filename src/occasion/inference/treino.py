"""Artefato avaliado: a receita congelada aplicada à base inteira de 2008-2009.

`treinar_avaliado` refaz, na base inteira, o que a receita escolheu em validação cruzada por setor:
o degrau e os parâmetros de cada parte, o peso no treino (`usar_peso`) e a sigmoide de Platt
(`calibrar`). Os coeficientes das sigmoides saem das previsões fora da dobra gravadas pela etapa
`receita` (`data/experimentos/selecao_<parte>.joblib`), nunca de previsões dentro da amostra.
Só 2008-2009 entra aqui; a edição de teste só é lida depois do teste final.
"""

from typing import Any, cast

import numpy as np
import pandas as pd

from occasion.config import EDICAO_TREINO
from occasion.experimentos.especificacao import ParteModelo, variaveis_da_parte
from occasion.inference.contract import MetadadosModelo
from occasion.inference.predict import (
    VERSAO_CONTRATO,
    ArtefatoModelo,
    _exigir_so_treino,
    _ufs_de_midia_regional,
)
from occasion.models.base import ModeloParte, logit, sigmoide
from occasion.models.calibracao import ModeloCalibrado
from occasion.models.escada import construir
from occasion.receita import Receita, ReceitaParte
from occasion.score.score import construir_referencia
from occasion.validacao.metricas import ajustar_sigmoide

ALVO_DA_PARTE: dict[ParteModelo, str] = {
    "ocasiao": "y_pizza_refri",
    "preferencia": "y_guarana",
    "direto": "y_pizza_guarana",
}
_DESCRICAO_DO_DEGRAU: dict[str, str] = {
    "A": "A (tabela de taxas)",
    "B": "B (logística)",
    "C": "C (logística com curvas e interações)",
    "D": "D (LightGBM)",
}
_ROTULO_DA_PARTE: dict[ParteModelo, str] = {
    "ocasiao": "ocasião",
    "preferencia": "preferência",
    "direto": "score direto",
}


def _previsao_valida(previsao: np.ndarray, n: int, o_que: str) -> np.ndarray:
    p = np.asarray(previsao, dtype=float)
    if p.shape != (n,):
        raise ValueError(f"{o_que}: previsão fora da dobra com forma {p.shape}, esperado ({n},)")
    if not np.isfinite(p).all():
        raise ValueError(f"{o_que}: previsão fora da dobra com NaN ou infinito")
    return p


def ajustar_parte(
    base: pd.DataFrame,
    alvo: str,
    parte: ParteModelo,
    receita_parte: ReceitaParte,
    previsao_fora_da_dobra: np.ndarray,
) -> ModeloParte:
    """Treina a parte com os parâmetros da receita na base inteira.

    O peso entra no treino só se `usar_peso`. Com `calibrar`, a sigmoide é ajustada com
    `peso_final` (a calibração mira a população) na previsão fora da dobra da seleção, e o
    resultado é um `ModeloCalibrado`.
    """
    modelo = construir(receita_parte.degrau, parte, receita_parte.parametros)
    X = cast(pd.DataFrame, base[variaveis_da_parte(parte)])
    y = cast(pd.Series, base[alvo])
    peso = cast(pd.Series, base["peso_final"])
    modelo.fit(X, y, peso if receita_parte.usar_peso else None)
    if not receita_parte.calibrar:
        return modelo
    p = _previsao_valida(previsao_fora_da_dobra, len(base), f"parte {parte}")
    inclinacao, intercepto = ajustar_sigmoide(
        p, y.to_numpy(dtype=float), peso.to_numpy(dtype=float)
    )
    # o calibrado só prevê e explica (sem `fit`), o que basta para a inferência
    return cast(ModeloParte, ModeloCalibrado(modelo, inclinacao, intercepto))


def _aplicar_calibracao(modelo: ModeloParte, previsao: np.ndarray) -> np.ndarray:
    """A previsão fora da dobra passando pela sigmoide da parte, se ela tiver uma."""
    if isinstance(modelo, ModeloCalibrado):
        return sigmoide(modelo.inclinacao * logit(previsao) + modelo.intercepto)
    return previsao


def _selecao(selecoes: dict[str, Any], parte: ParteModelo, degrau: str) -> dict[str, Any]:
    if parte not in selecoes:
        raise ValueError(f"falta a seleção da parte {parte!r} (selecao_{parte}.joblib)")
    selecao = selecoes[parte]
    if selecao["degrau"] != degrau:
        raise ValueError(
            f"seleção de {parte!r} é do degrau {selecao['degrau']}, mas a receita manda {degrau}"
        )
    return selecao


def _conferir_dobras(selecoes: dict[str, Any]) -> None:
    """As seleções precisam vir das mesmas dobras: a calibração do score junta as previsões."""
    hashes = {nome: s.get("hash_dobras") for nome, s in selecoes.items()}
    if len(set(hashes.values())) > 1:
        raise ValueError(f"seleções de dobras diferentes: {hashes}")


def _descricao_do_degrau(receita: Receita) -> str:
    partes: list[tuple[ParteModelo, ReceitaParte]] = [
        ("ocasiao", receita.ocasiao),
        ("preferencia", receita.preferencia),
    ]
    if receita.direto is not None:
        partes.append(("direto", receita.direto))
    texto = "; ".join(f"{_ROTULO_DA_PARTE[p]}: {_DESCRICAO_DO_DEGRAU[r.degrau]}" for p, r in partes)
    if receita.calibrar_score:
        texto += "; score calibrado"
    return texto


def treinar_avaliado(
    receita: Receita,
    celulas: pd.DataFrame,
    ocasioes_refri: pd.DataFrame,
    selecoes: dict[str, Any],
    commit: str,
    assinatura: str | None = None,
) -> ArtefatoModelo:
    """Monta o artefato avaliado: partes treinadas, calibração do score e referência.

    `selecoes` é o dicionário `{parte: conteúdo de selecao_<parte>.joblib}`, com as chaves
    `degrau`, `parametros`, `oof` e `oof_extras` (na preferência, `oof_extras["celulas"]` é a
    previsão fora da dobra nas células). Só 2008-2009.
    """
    _exigir_so_treino(celulas, "celulas")
    _exigir_so_treino(ocasioes_refri, "ocasioes_refri")
    n = len(celulas)
    _conferir_dobras(selecoes)

    sel_p1 = _selecao(selecoes, "ocasiao", receita.ocasiao.degrau)
    modelo_p1 = ajustar_parte(
        celulas, ALVO_DA_PARTE["ocasiao"], "ocasiao", receita.ocasiao, sel_p1["oof"]
    )
    sel_p2 = _selecao(selecoes, "preferencia", receita.preferencia.degrau)
    # a sigmoide da preferência mira a base em que a parte é treinada: as ocasiões com refrigerante
    modelo_p2 = ajustar_parte(
        ocasioes_refri,
        ALVO_DA_PARTE["preferencia"],
        "preferencia",
        receita.preferencia,
        sel_p2["oof"],
    )

    modelo_direto: ModeloParte | None = None
    sel_direto: dict[str, Any] | None = None
    if receita.direto is not None:
        sel_direto = _selecao(selecoes, "direto", receita.direto.degrau)
        modelo_direto = ajustar_parte(
            celulas, ALVO_DA_PARTE["direto"], "direto", receita.direto, sel_direto["oof"]
        )

    calibracao_score: tuple[float, float] | None = None
    if receita.calibrar_score:
        if sel_direto is not None and modelo_direto is not None:
            produto = _aplicar_calibracao(
                modelo_direto, _previsao_valida(sel_direto["oof"], n, "direto")
            )
        else:
            p1 = _aplicar_calibracao(modelo_p1, _previsao_valida(sel_p1["oof"], n, "ocasião"))
            p2_celulas = _previsao_valida(
                sel_p2["oof_extras"]["celulas"], n, "preferência nas células"
            )
            produto = p1 * _aplicar_calibracao(modelo_p2, p2_celulas)
        calibracao_score = ajustar_sigmoide(
            produto,
            celulas[ALVO_DA_PARTE["direto"]].to_numpy(dtype=float),
            celulas["peso_final"].to_numpy(dtype=float),
        )

    referencia = construir_referencia(
        celulas, modelo_p1, modelo_p2, calibracao_score, modelo_direto
    )
    metadados = MetadadosModelo(
        modelo="avaliado",
        degrau=_descricao_do_degrau(receita),
        treinado_em=[EDICAO_TREINO],
        receita_commit=commit,
        metricas_da_receita=None,
        descricao_metricas=(
            "Ainda não medidas: o teste temporal acontece depois que a receita for congelada."
        ),
        instante_de_previsao="planejamento, antes da ocasião",
        aviso="Guaraná é o sabor registrado na POF, não uma marca.",
        versao_contrato=VERSAO_CONTRATO,
    )
    return ArtefatoModelo(
        modelo_p1=modelo_p1,
        modelo_p2=modelo_p2,
        referencia=referencia,
        ufs_midia_regional=_ufs_de_midia_regional(celulas, modelo_p1, modelo_p2),
        metadados=metadados,
        calibracao_score=calibracao_score,
        modelo_direto=modelo_direto,
        assinatura_receita=assinatura,
    )


__all__ = ["ALVO_DA_PARTE", "ajustar_parte", "treinar_avaliado"]
