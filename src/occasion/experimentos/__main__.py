"""Linha de comando das etapas: `python -m occasion.experimentos <etapa>`."""

import argparse
from collections.abc import Callable
from typing import cast

from occasion.experimentos import etapas, teste_final


def _escada(parte: str) -> Callable[[], None]:
    def rodar() -> None:
        if parte == "ambas":
            # a preferência (curta) primeiro: uma falha na ocasião (longa) não a impede
            etapas.etapa_escada("preferencia")
            etapas.etapa_escada("ocasiao")
        else:
            etapas.etapa_escada(cast(etapas.Parte, parte))

    return rodar


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m occasion.experimentos")
    parser.add_argument(
        "etapa",
        choices=[
            "dobras", "peso", "escada", "h4", "embaralhado",
            "calibracao", "produto", "cortes", "receita", "avaliacao-2008", "teste-final",
        ],
    )  # fmt: skip
    parser.add_argument(
        "--parte",
        choices=["ocasiao", "preferencia", "ambas"],
        default="ambas",
        help="só para a etapa escada (padrão: as duas partes)",
    )
    args = parser.parse_args()
    acoes: dict[str, Callable[[], None]] = {
        "dobras": etapas.etapa_dobras,
        "peso": etapas.etapa_peso,
        "escada": _escada(args.parte),
        "h4": etapas.etapa_h4,
        "embaralhado": etapas.etapa_embaralhado,
        "calibracao": etapas.etapa_calibracao,
        "produto": etapas.etapa_produto,
        "cortes": etapas.etapa_cortes,
        "receita": etapas.etapa_receita,
        "avaliacao-2008": etapas.etapa_avaliacao_2008,
        # o teste final (H8) só roda depois da tag `receita-congelada` e de `make avaliado`
        "teste-final": lambda: print(f"gravado: {teste_final.rodar_teste_final()}"),
    }
    acoes[args.etapa]()


if __name__ == "__main__":
    main()
