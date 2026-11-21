"""A receita congelada do Plano 2: as escolhas feitas em 2008-2009, em JSON.

A receita guarda escolhas, não valores ajustados (seção 8.3 do design): variáveis, faixas de
horário, uso do peso no treino, degrau e hiperparâmetros de cada parte, se a sigmoide entra,
como o score é calculado e os cortes. Os coeficientes da sigmoide e os do modelo ficam no
artefato, porque o modelo servido os reajusta com a mesma receita.
"""

from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from occasion.config import RELATORIOS
from occasion.experimentos.especificacao import ParteModelo
from occasion.models.escada import Degrau

__all__ = ["CHAVES_DOS_CORTES", "Receita", "ReceitaParte", "carregar_receita", "salvar_receita"]

CHAVES_DOS_CORTES: frozenset[str] = frozenset({"alta", "media", "ocasiao", "preferencia"})


class ReceitaParte(BaseModel):
    """Escolhas de uma parte: degrau vencedor, hiperparâmetros, peso no treino e calibração."""

    model_config = ConfigDict(extra="forbid")

    degrau: Degrau
    parametros: dict[str, Any]
    usar_peso: bool
    calibrar: bool


class Receita(BaseModel):
    """Tudo o que o modelo avaliado e o servido precisam para serem refeitos do mesmo jeito."""

    model_config = ConfigDict(extra="forbid")

    faixas_horario: dict[str, list[int]]
    variaveis: dict[ParteModelo, list[str]]
    interacoes: dict[ParteModelo, list[tuple[str, str]]]
    ocasiao: ReceitaParte
    preferencia: ReceitaParte
    score: Literal["produto", "direto"]
    direto: ReceitaParte | None
    calibrar_score: bool
    referencia_a: dict[ParteModelo, dict[str, Any]]
    cortes: dict[str, float]
    experimentos: list[str]

    @model_validator(mode="after")
    def _coerente(self) -> Self:
        if self.score == "direto" and self.direto is None:
            raise ValueError("score 'direto' exige a receita do modelo direto")
        if self.score == "produto" and self.direto is not None:
            raise ValueError("o modelo direto só entra na receita quando o score é 'direto'")
        if set(self.cortes) != CHAVES_DOS_CORTES:
            raise ValueError(
                f"cortes precisam ter exatamente {sorted(CHAVES_DOS_CORTES)}: {sorted(self.cortes)}"
            )
        faltam = {"ocasiao", "preferencia"} - set(self.referencia_a)
        if self.score == "direto" and "direto" not in self.referencia_a:
            faltam.add("direto")
        if faltam:
            raise ValueError(f"referencia_a sem a configuração do degrau A de: {sorted(faltam)}")
        return self


def salvar_receita(receita: Receita, caminho: Path = RELATORIOS / "receita.json") -> Path:
    """Grava a receita em JSON legível e devolve o caminho."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(receita.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return caminho


def carregar_receita(caminho: Path = RELATORIOS / "receita.json") -> Receita:
    """Lê e valida a receita gravada por `salvar_receita`."""
    return Receita.model_validate_json(caminho.read_text(encoding="utf-8"))
