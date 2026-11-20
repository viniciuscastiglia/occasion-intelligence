"""Linha de comando das etapas: `python -m occasion.experimentos <etapa>`."""

import argparse
from collections.abc import Callable
from typing import cast

from occasion.experimentos import etapas


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
    parser.add_argument("etapa", choices=["dobras", "peso", "escada", "h4", "embaralhado"])
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
    }
    acoes[args.etapa]()


if __name__ == "__main__":
    main()
