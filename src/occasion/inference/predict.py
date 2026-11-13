"""Artefato do modelo e função única de inferência.

`treinar_baseline` ajusta o degrau A só em 2008-2009. `prever` é a função que a tela e o
processamento em lote chamam: ambos passam por `pontuar`, então dão o mesmo score.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd

from occasion.config import EDICAO_TREINO, MODELOS
from occasion.decision.engine import decidir, justificar
from occasion.features.variaveis import COLUNAS_CONTEXTO, regiao_da_uf
from occasion.inference.contract import (
    CODIGO_UF,
    PONTO_MEDIO_RENDA,
    UF,
    Contexto,
    Fator,
    MetadadosModelo,
    Resultado,
)
from occasion.models.taxas import DIMENSOES_PARTE_1, DIMENSOES_PARTE_2, TabelaDeTaxas
from occasion.score.score import (
    ReferenciaScore,
    classificar,
    construir_referencia,
    probabilidades,
)

N_UFS_MIDIA_REGIONAL = 7
MAXIMO_TOP_FATORES = 5
VERSAO_CONTRATO = "1.0"
NOME_ARQUIVO = "artefato.joblib"
_TOLERANCIA_EMPATE = 1e-12

_DIAS: dict[int, str] = {
    1: "segunda",
    2: "terca",
    3: "quarta",
    4: "quinta",
    5: "sexta",
    6: "sabado",
    7: "domingo",
}
_NUMERO_DO_DIA: dict[str, int] = {nome: numero for numero, nome in _DIAS.items()}
_SIGLA_DA_UF: dict[int, str] = {codigo: sigla for sigla, codigo in CODIGO_UF.items()}


class ArtefatoAusente(FileNotFoundError):
    """Nenhum artefato de modelo foi encontrado."""


@dataclass
class ArtefatoModelo:
    """Tudo o que a inferência precisa: as duas partes, a referência do score e os metadados."""

    modelo_p1: Any
    modelo_p2: Any
    referencia: ReferenciaScore
    ufs_midia_regional: frozenset[str]
    metadados: MetadadosModelo


def _exigir_so_treino(base: pd.DataFrame, nome: str) -> None:
    edicoes = set(cast(pd.Series, base["edicao"]).unique())
    if edicoes != {EDICAO_TREINO}:
        raise ValueError(
            f"{nome} deve conter só a edição {EDICAO_TREINO}; veio {sorted(map(str, edicoes))}"
        )


def _ufs_de_midia_regional(celulas: pd.DataFrame, modelo_p1: Any, modelo_p2: Any) -> frozenset[str]:
    """As UFs de maior preferência média ponderada, com as empatadas com a última incluídas."""
    p = probabilidades(celulas, modelo_p1, modelo_p2)
    pesos = celulas["peso_final"].to_numpy(dtype=float)
    pref = p["p_preferencia"].to_numpy(dtype=float)
    codigos = celulas["uf"].to_numpy()
    medias: dict[str, float] = {}
    for codigo in np.unique(codigos):
        da_uf = codigos == codigo
        medias[_SIGLA_DA_UF[int(codigo)]] = float(
            (pesos[da_uf] * pref[da_uf]).sum() / pesos[da_uf].sum()
        )
    ordenadas = sorted(medias.items(), key=lambda item: (-item[1], item[0]))
    corte = ordenadas[min(N_UFS_MIDIA_REGIONAL, len(ordenadas)) - 1][1]
    return frozenset(
        sigla for sigla, media in ordenadas if media >= corte * (1 - _TOLERANCIA_EMPATE)
    )


def treinar_baseline(
    celulas: pd.DataFrame, ocasioes_refri: pd.DataFrame, commit: str
) -> ArtefatoModelo:
    """Ajusta o degrau A (tabela de taxas) nas duas partes, só com dados de 2008-2009."""
    _exigir_so_treino(celulas, "celulas")
    _exigir_so_treino(ocasioes_refri, "ocasioes_refri")
    modelo_p1 = TabelaDeTaxas(DIMENSOES_PARTE_1).fit(
        celulas, cast(pd.Series, celulas["y_pizza_refri"]), cast(pd.Series, celulas["peso_final"])
    )
    modelo_p2 = TabelaDeTaxas(DIMENSOES_PARTE_2).fit(
        ocasioes_refri,
        cast(pd.Series, ocasioes_refri["y_guarana"]),
        cast(pd.Series, ocasioes_refri["peso_final"]),
    )
    referencia = construir_referencia(celulas, modelo_p1, modelo_p2)
    metadados = MetadadosModelo(
        modelo="avaliado",
        degrau="A (tabela de taxas)",
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
    )


def salvar_artefato(artefato: ArtefatoModelo, pasta: Path) -> Path:
    """Grava `pasta/artefato.joblib` e devolve o caminho."""
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / NOME_ARQUIVO
    joblib.dump(artefato, destino)
    return destino


def carregar_artefato(pasta: Path | None = None) -> ArtefatoModelo:
    """Carrega `servido/artefato.joblib` da raiz dos modelos; senão `avaliado/artefato.joblib`."""
    raiz = MODELOS if pasta is None else pasta
    for subpasta in ("servido", "avaliado"):
        caminho = raiz / subpasta / NOME_ARQUIVO
        if caminho.exists():
            return cast(ArtefatoModelo, joblib.load(caminho))
    raise ArtefatoAusente("Modelo não encontrado. Rode `make baseline`.")


def contexto_para_linha(contexto: Contexto) -> pd.DataFrame:
    """Uma linha com `COLUNAS_CONTEXTO`, no formato das bases de modelagem."""
    codigo = CODIGO_UF[contexto.uf]
    linha = {
        "dia_semana": _NUMERO_DO_DIA[contexto.dia_semana],
        "faixa_horario": contexto.faixa_horario,
        "local": contexto.local,
        "uf": codigo,
        "regiao": regiao_da_uf(codigo),
        "situacao": contexto.situacao,
        "idade": contexto.idade,
        "sexo": contexto.sexo,
        "renda_pc_sm": PONTO_MEDIO_RENDA[contexto.faixa_renda],
        "n_moradores_uc": contexto.numero_moradores,
    }
    return pd.DataFrame([linha], columns=COLUNAS_CONTEXTO)


def pontuar(X: pd.DataFrame, artefato: ArtefatoModelo) -> pd.DataFrame:
    """Probabilidades, posições e score (posição do produto) de cada linha de contexto."""
    p = probabilidades(X, artefato.modelo_p1, artefato.modelo_p2)
    if p.isna().any().any():
        raise ValueError("probabilidade NaN: o contexto não pôde ser pontuado")
    ref = artefato.referencia
    for qual, coluna in (
        ("ocasiao", "p_ocasiao"),
        ("preferencia", "p_preferencia"),
        ("produto", "p_produto"),
    ):
        p[f"posicao_{qual}"] = [ref.posicao(float(v), qual) for v in p[coluna]]  # type: ignore[arg-type]
    p["score"] = p["posicao_produto"]
    return p


def _texto_do_valor(variavel: str, valor: Any) -> str:
    if variavel == "dia_semana":
        return _DIAS[int(valor)]
    return str(valor)


def _top_fatores(linha: pd.DataFrame, artefato: ArtefatoModelo) -> list[Fator]:
    candidatos: list[Fator] = []
    partes = (
        ("ocasiao", artefato.modelo_p1, linha),
        ("preferencia", artefato.modelo_p2, linha.assign(tem_pizza=1)),
    )
    for parte, modelo, X in partes:
        contribuicoes = modelo.contribuicoes(X)
        for variavel in modelo.dimensoes:
            valor = float(contribuicoes[variavel].iloc[0])
            candidatos.append(
                Fator(
                    parte=parte,  # type: ignore[arg-type]
                    variavel=variavel,
                    valor=_texto_do_valor(variavel, linha[variavel].iloc[0]),
                    direcao="aumenta" if valor > 0 else "diminui",
                    contribuicao=valor,
                )
            )
    candidatos.sort(key=lambda f: abs(f.contribuicao), reverse=True)
    return candidatos[:MAXIMO_TOP_FATORES]


def prever(contexto: Contexto, artefato: ArtefatoModelo) -> Resultado:
    """Contexto em, `Resultado` do contrato fora: score, fatores e recomendação."""
    linha = contexto_para_linha(contexto)
    p = pontuar(linha, artefato).iloc[0]
    score = float(p["score"])
    fatores = _top_fatores(linha, artefato)
    uf: UF = contexto.uf
    decisao = decidir(
        float(p["posicao_ocasiao"]),
        float(p["posicao_preferencia"]),
        contexto.local,
        contexto.situacao,
        contexto.idade,
        uf,
        artefato.ufs_midia_regional,
    )
    return Resultado(
        occasion_score=score,
        classification=classificar(score),
        occasion_probability=float(p["p_produto"]),
        score_components={
            "ocasiao": {
                "posicao": float(p["posicao_ocasiao"]),
                "probabilidade": float(p["p_ocasiao"]),
            },
            "preferencia": {
                "posicao": float(p["posicao_preferencia"]),
                "probabilidade": float(p["p_preferencia"]),
            },
        },  # type: ignore[arg-type]
        top_factors=fatores,
        recommendation_category=decisao.categoria,
        recommendation_reason=justificar(decisao, fatores),
        model_metadata=artefato.metadados,
    )
