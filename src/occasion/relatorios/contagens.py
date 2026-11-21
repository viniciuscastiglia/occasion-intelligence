"""Contagens de referência das bases de modelagem, gravadas em `reports/contagens.json`.

O Data Card e o README leem este arquivo, e não a memória, para citar os números.
"""

import json
from typing import Any

import pandas as pd

from occasion.bases.acesso import carregar_base, exigir_teste_commitado
from occasion.config import EDICAO_TESTE, EDICAO_TREINO, RAIZ

CAMINHO_JSON = RAIZ / "reports" / "contagens.json"

NOTA = (
    "Contagens das bases de modelagem (células e ocasiões com refrigerante), depois das "
    "exclusões. Por isso 2017-2018 tem 796 dias com pizza e refrigerante no mesmo horário, "
    "e não os 797 da tabela de referência do parser, e 16.112 ocasiões com refrigerante, "
    "e não 16.122 (10 saem por local indeterminado). 'Dias com pizza' (sem refrigerante) "
    "não entra: exigiria ler a base pessoa x dia de 2017-2018 fora de `carregar_base`, a "
    "única porta de leitura do alvo da edição de teste."
)


def contar(celulas: pd.DataFrame, ocasioes_refri: pd.DataFrame) -> dict[str, Any]:
    """Conta dias-pessoa, células, dias positivos e ocasiões com refrigerante (brutas, sem peso)."""
    por_dia = celulas.drop_duplicates("id_dia")
    contagem_semana: dict[int, int] = {}
    for dia in por_dia["dia_semana"].astype(int).tolist():
        contagem_semana[dia] = contagem_semana.get(dia, 0) + 1
    return {
        "dias_pessoa": len(por_dia),
        "celulas": len(celulas),
        "dias_pizza_refri": len(set(celulas.loc[celulas["y_pizza_refri"] == 1, "id_dia"])),
        "dias_pizza_guarana": len(set(celulas.loc[celulas["y_pizza_guarana"] == 1, "id_dia"])),
        "ocasioes_refri": len(ocasioes_refri),
        "dias_por_dia_semana": {str(d): contagem_semana.get(d, 0) for d in range(1, 8)},
    }


def main() -> None:
    """Grava `reports/contagens.json` com as duas edições."""
    exigir_teste_commitado()
    saida: dict[str, Any] = {
        edicao: contar(carregar_base("celulas", edicao), carregar_base("ocasioes_refri", edicao))
        for edicao in (EDICAO_TREINO, EDICAO_TESTE)
    }
    saida["nota"] = NOTA
    CAMINHO_JSON.write_text(
        json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
