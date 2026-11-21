"""Contrato de entrada (Contexto) e de saída (Resultado) da aplicação.

Definido em Pydantic. O Decision Engine, a função de inferência e a tela trabalham
contra estes tipos e contra o exemplo fixo em tests/exemplos/resultado_exemplo.json.
"""

from typing import Literal

from pydantic import BaseModel, Field

# `UF` e `CODIGO_UF` moram em `occasion.config` (assinado, porque definem o domínio da `uf`)
# e continuam expostos aqui para quem já os importa do contrato.
from occasion.config import CODIGO_UF, UF  # noqa: F401  (CODIGO_UF é reexportado)

DiaSemana = Literal["segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo"]
FaixaHorario = Literal["madrugada", "manha", "almoco", "tarde", "noite"]
Local = Literal["em_casa", "fora_de_casa"]
Situacao = Literal["urbano", "rural"]
Sexo = Literal["masculino", "feminino"]
FaixaRenda = Literal["ate_meio_sm", "meio_a_1_sm", "de_1_a_2_sm", "de_2_a_5_sm", "mais_de_5_sm"]
Classificacao = Literal["alta", "média", "baixa"]

# Versão do contrato de saída gravada nos metadados de cada artefato.
VERSAO_CONTRATO = "1.0"

# Ponto médio de cada faixa de renda, em salários mínimos por pessoa.
PONTO_MEDIO_RENDA: dict[FaixaRenda, float] = {
    "ate_meio_sm": 0.25,
    "meio_a_1_sm": 0.75,
    "de_1_a_2_sm": 1.5,
    "de_2_a_5_sm": 3.5,
    "mais_de_5_sm": 7.5,
}


class Contexto(BaseModel):
    """Entrada: o contexto em que a ocasião é planejada."""

    uf: UF
    situacao: Situacao
    dia_semana: DiaSemana
    faixa_horario: FaixaHorario
    local: Local
    idade: int = Field(ge=18, le=110)
    sexo: Sexo
    faixa_renda: FaixaRenda
    numero_moradores: int = Field(ge=1, le=20)
    versao_contrato: str = "1.0"


class ComponenteScore(BaseModel):
    """Uma das duas partes do score: posição na população de referência e probabilidade."""

    posicao: float
    probabilidade: float


class Fator(BaseModel):
    """Um fator que sustenta o score, com a direção e a contribuição do modelo."""

    parte: Literal["ocasiao", "preferencia", "direto"]
    variavel: str
    valor: str
    direcao: Literal["aumenta", "diminui"]
    contribuicao: float


class MetadadosModelo(BaseModel):
    """Em que dados o modelo foi treinado e os limites da previsão."""

    modelo: Literal["avaliado", "servido"]
    degrau: str
    treinado_em: list[str]
    receita_commit: str
    metricas_da_receita: dict[str, float] | None
    descricao_metricas: str
    instante_de_previsao: str
    aviso: str
    versao_contrato: str


class Resultado(BaseModel):
    """Saída da aplicação: o contrato do TAPI mais os componentes do score."""

    occasion_score: float = Field(ge=0, le=100)
    classification: Classificacao
    occasion_probability: float = Field(ge=0, le=1)
    score_components: dict[Literal["ocasiao", "preferencia"], ComponenteScore]
    top_factors: list[Fator]
    recommendation_category: str
    recommendation_reason: str
    model_metadata: MetadadosModelo
