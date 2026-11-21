"""Compara dois artefatos servidos numa grade fixa de contextos.

Uso: `uv run python scripts/comparar_artefatos.py antigo.joblib novo.joblib`.
Imprime cada campo do `Resultado` que difere (floats por igualdade exata, ignorando
`model_metadata.receita_commit`) e sai com código 1 se houver alguma diferença.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import joblib

from occasion.inference.contract import Contexto
from occasion.inference.predict import ArtefatoModelo, prever

DIAS = ("segunda", "terca", "quarta", "quinta", "sexta", "sabado", "domingo")
FAIXAS = ("madrugada", "manha", "almoco", "tarde", "noite")

PERFIS: tuple[dict[str, Any], ...] = (
    {
        "uf": "SP",
        "situacao": "urbano",
        "local": "em_casa",
        "idade": 28,
        "sexo": "feminino",
        "faixa_renda": "de_2_a_5_sm",
        "numero_moradores": 3,
    },
    {
        "uf": "BA",
        "situacao": "rural",
        "local": "fora_de_casa",
        "idade": 45,
        "sexo": "masculino",
        "faixa_renda": "ate_meio_sm",
        "numero_moradores": 5,
    },
    {
        "uf": "RS",
        "situacao": "urbano",
        "local": "fora_de_casa",
        "idade": 22,
        "sexo": "feminino",
        "faixa_renda": "mais_de_5_sm",
        "numero_moradores": 1,
    },
    {
        "uf": "AM",
        "situacao": "urbano",
        "local": "em_casa",
        "idade": 60,
        "sexo": "masculino",
        "faixa_renda": "de_1_a_2_sm",
        "numero_moradores": 4,
    },
)


def grade() -> list[Contexto]:
    """Os 35 pares dia x faixa cruzados com os 4 perfis fixos (140 contextos)."""
    return [
        Contexto(dia_semana=dia, faixa_horario=faixa, **perfil)
        for perfil in PERFIS
        for dia in DIAS
        for faixa in FAIXAS
    ]


def _diferencas(caminho: str, a: Any, b: Any) -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        saida: list[str] = []
        for chave in sorted(set(a) | set(b)):
            if caminho == "model_metadata" and chave == "receita_commit":
                continue
            saida += _diferencas(f"{caminho}.{chave}", a.get(chave), b.get(chave))
        return saida
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        saida = []
        for i, (x, y) in enumerate(zip(a, b)):
            saida += _diferencas(f"{caminho}[{i}]", x, y)
        return saida
    return [] if a == b else [f"{caminho}: {a!r} != {b!r}"]


def comparar(antigo: ArtefatoModelo, novo: ArtefatoModelo) -> list[str]:
    """Descreve cada campo do `Resultado` que difere entre os artefatos, na grade fixa."""
    saida: list[str] = []
    for contexto in grade():
        rotulo = f"{contexto.uf}/{contexto.dia_semana}/{contexto.faixa_horario}"
        a = prever(contexto, antigo).model_dump()
        b = prever(contexto, novo).model_dump()
        saida += _diferencas(rotulo, a, b)
    return saida


def main(argv: list[str]) -> int:
    """Lê os dois caminhos, imprime as diferenças e devolve o código de saída."""
    if len(argv) != 2:
        print("uso: comparar_artefatos.py antigo.joblib novo.joblib")
        return 2
    antigo = cast(ArtefatoModelo, joblib.load(Path(argv[0])))
    novo = cast(ArtefatoModelo, joblib.load(Path(argv[1])))
    diferencas = comparar(antigo, novo)
    for linha in diferencas:
        print(linha)
    print(f"{len(diferencas)} diferença(s) em {len(grade())} contextos")
    return 1 if diferencas else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
