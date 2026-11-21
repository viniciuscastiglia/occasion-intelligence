"""Artefato do modelo e função única de inferência.

`treinar_baseline` ajusta o degrau A só em 2008-2009. `prever` é a função que a tela e o
processamento em lote chamam: ambos passam por `pontuar`, então dão o mesmo score.

`ArtefatoModelo`, `pontuar` e o que monta o artefato avaliado moram em
`occasion.inference.artefato`, que é assinado com a receita; este módulo (a resposta da tela e do
contrato) fica fora da assinatura e os reexporta, então `from occasion.inference.predict import
ArtefatoModelo, pontuar` continua valendo, inclusive para artefatos gravados antes da mudança.
"""

from pathlib import Path
from typing import Any, cast

import joblib
import pandas as pd

from occasion.config import EDICAO_TREINO, MODELOS
from occasion.decision.engine import decidir, justificar
from occasion.features.variaveis import COLUNAS_CONTEXTO, regiao_da_uf
from occasion.inference.artefato import (
    N_UFS_MIDIA_REGIONAL,
    NOME_ARQUIVO,
    ArtefatoModelo,
    _exigir_so_treino,
    _ufs_de_midia_regional,
    pontuar,
)
from occasion.inference.contract import (
    CODIGO_UF,
    PONTO_MEDIO_RENDA,
    UF,
    VERSAO_CONTRATO,
    Contexto,
    Fator,
    MetadadosModelo,
    Resultado,
)
from occasion.models.taxas import DIMENSOES_PARTE_1, DIMENSOES_PARTE_2, TabelaDeTaxas
from occasion.score.score import classificar, construir_referencia

MAXIMO_TOP_FATORES = 5

__all__ = [
    "MAXIMO_TOP_FATORES",
    "NOME_ARQUIVO",
    "N_UFS_MIDIA_REGIONAL",
    "VERSAO_CONTRATO",
    "ArtefatoAusente",
    "ArtefatoModelo",
    "carregar_artefato",
    "contexto_para_linha",
    "pontuar",
    "prever",
    "salvar_artefato",
    "treinar_baseline",
]

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


class ArtefatoAusente(FileNotFoundError):
    """Nenhum artefato de modelo foi encontrado."""


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
    """Carrega `servido/artefato.joblib` da raiz dos modelos; depois `avaliado` e `baseline`."""
    raiz = MODELOS if pasta is None else pasta
    for subpasta in ("servido", "avaliado", "baseline"):
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
    if set(linha) != set(COLUNAS_CONTEXTO):
        raise ValueError(
            "contexto_para_linha fora de paridade com COLUNAS_CONTEXTO: "
            f"faltam {sorted(set(COLUNAS_CONTEXTO) - set(linha))}, "
            f"sobram {sorted(set(linha) - set(COLUNAS_CONTEXTO))}"
        )
    return pd.DataFrame([linha], columns=COLUNAS_CONTEXTO)


def _texto_do_valor(variavel: str, valor: Any) -> str:
    if variavel == "dia_semana":
        return _DIAS[int(valor)]
    return str(valor)


def _top_fatores(linha: pd.DataFrame, artefato: ArtefatoModelo) -> list[Fator]:
    candidatos: list[Fator] = []
    if artefato.modelo_direto is not None:
        partes = (("direto", artefato.modelo_direto, linha),)
    else:
        partes = (
            ("ocasiao", artefato.modelo_p1, linha),
            ("preferencia", artefato.modelo_p2, linha.assign(tem_pizza=1)),
        )
    for parte, modelo, X in partes:
        contribuicoes = modelo.contribuicoes(X)
        for variavel in modelo.variaveis:
            if variavel == "tem_pizza":  # fixada em "sim" na inferência: não é fator do contexto
                continue
            valor = float(contribuicoes[variavel].iloc[0])
            # `fim_de_semana` deriva de `dia_semana`: a tela e os rótulos continuam com o dia
            nome = "dia_semana" if variavel == "fim_de_semana" else variavel
            candidatos.append(
                Fator(
                    parte=parte,  # type: ignore[arg-type]
                    variavel=nome,
                    valor=_texto_do_valor(nome, linha[nome].iloc[0]),
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
