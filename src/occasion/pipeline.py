"""Linha de comando do pipeline: `uv run python -m occasion.pipeline <subcomando>`."""

import argparse
from collections.abc import Sequence

from occasion.config import DADOS_BRUTOS, DADOS_INTERIM
from occasion.ingest.parse import gravar_interim_2017

_INTERIM_POR_EDICAO = {"2017-2018": gravar_interim_2017}


def _interim(edicao: str) -> None:
    for destino in _INTERIM_POR_EDICAO[edicao](DADOS_BRUTOS / edicao, DADOS_INTERIM):
        print(f"gravado: {destino}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="occasion.pipeline", description=__doc__)
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    interim = subcomandos.add_parser(
        "interim", help="lê os brutos de uma edição e grava os parquets"
    )
    interim.add_argument("--edicao", required=True, choices=sorted(_INTERIM_POR_EDICAO))
    args = parser.parse_args(argv)
    if args.comando == "interim":
        _interim(args.edicao)


if __name__ == "__main__":
    main()
