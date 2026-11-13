"""Linha de comando do pipeline: `uv run python -m occasion.pipeline <subcomando>`."""

import argparse
import json
from collections.abc import Sequence

import pandas as pd

from occasion.bases.modelagem import montar_celulas, montar_ocasioes_refri
from occasion.bases.ocasioes import montar_ocasioes, montar_pessoa_dia
from occasion.config import (
    DADOS_BRUTOS,
    DADOS_INTERIM,
    DADOS_PROCESSADOS,
    EDICOES,
    RELATORIOS,
)
from occasion.ingest.parse import gravar_interim_2008, gravar_interim_2017

_INTERIM_POR_EDICAO = {
    "2008-2009": gravar_interim_2008,
    "2017-2018": gravar_interim_2017,
}


def _interim(edicao: str) -> None:
    for destino in _INTERIM_POR_EDICAO[edicao](DADOS_BRUTOS / edicao, DADOS_INTERIM):
        print(f"gravado: {destino}")


def _atualizar_relatorio(edicao: str, valores: dict[str, int]) -> None:
    """Grava `reports/bases.json`, preservando as chaves que já existirem."""
    caminho = RELATORIOS / "bases.json"
    relatorio: dict[str, dict[str, int]] = {}
    if caminho.exists():
        relatorio = json.loads(caminho.read_text(encoding="utf-8"))
    relatorio.setdefault(edicao, {}).update(valores)
    RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(relatorio, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _ocasioes(edicao: str) -> None:
    consumo = pd.read_parquet(DADOS_INTERIM / f"consumo_{edicao}.parquet")
    morador = pd.read_parquet(DADOS_INTERIM / f"morador_{edicao}.parquet")
    ocasioes = montar_ocasioes(consumo, edicao)
    pessoa_dia = montar_pessoa_dia(consumo, morador, ocasioes)
    DADOS_PROCESSADOS.mkdir(parents=True, exist_ok=True)
    for nome, quadro in (("ocasioes", ocasioes), ("pessoa_dia", pessoa_dia)):
        destino = DADOS_PROCESSADOS / f"{nome}_{edicao}.parquet"
        quadro.to_parquet(destino, index=False)
        print(f"gravado: {destino}")
    _atualizar_relatorio(edicao, {"itens_sem_hora": int(ocasioes.attrs["itens_sem_hora"])})


def _modelagem(edicao: str) -> None:
    pessoa_dia = pd.read_parquet(DADOS_PROCESSADOS / f"pessoa_dia_{edicao}.parquet")
    ocasioes = pd.read_parquet(DADOS_PROCESSADOS / f"ocasioes_{edicao}.parquet")
    celulas = montar_celulas(pessoa_dia, ocasioes)
    ocasioes_refri = montar_ocasioes_refri(pessoa_dia, ocasioes)
    for nome, quadro in (("celulas", celulas), ("ocasioes_refri", ocasioes_refri)):
        destino = DADOS_PROCESSADOS / f"{nome}_{edicao}.parquet"
        quadro.to_parquet(destino, index=False)
        print(f"gravado: {destino} ({len(quadro)} linhas)")
    _atualizar_relatorio(
        edicao,
        {
            "dias_positivos_excluidos": int(celulas.attrs["dias_positivos_excluidos"]),
            "ocasioes_excluidas": int(ocasioes_refri.attrs["ocasioes_excluidas"]),
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="occasion.pipeline", description=__doc__)
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    interim = subcomandos.add_parser(
        "interim", help="lê os brutos de uma edição e grava os parquets"
    )
    interim.add_argument("--edicao", required=True, choices=sorted(_INTERIM_POR_EDICAO))
    ocasioes = subcomandos.add_parser(
        "ocasioes", help="monta ocasiões e a base pessoa x dia a partir dos parquets interim"
    )
    ocasioes.add_argument("--edicao", choices=EDICOES, help="padrão: as duas edições")
    modelagem = subcomandos.add_parser(
        "modelagem", help="monta as bases de células (parte 1) e de ocasiões com refrigerante"
    )
    modelagem.add_argument("--edicao", choices=EDICOES, help="padrão: as duas edições")
    args = parser.parse_args(argv)
    if args.comando == "interim":
        _interim(args.edicao)
    elif args.comando == "ocasioes":
        for edicao in [args.edicao] if args.edicao else EDICOES:
            _ocasioes(edicao)
    elif args.comando == "modelagem":
        for edicao in [args.edicao] if args.edicao else EDICOES:
            _modelagem(edicao)


if __name__ == "__main__":
    main()
