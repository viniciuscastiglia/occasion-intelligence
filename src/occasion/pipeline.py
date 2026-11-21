"""Linha de comando do pipeline: `uv run python -m occasion.pipeline <subcomando>`."""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from occasion.audit.auditoria import rodar_auditoria
from occasion.bases.acesso import (
    TesteBloqueado,
    assinatura_da_receita,
    carregar_base,
    exigir_receita_intacta,
    exigir_teste_commitado,
)
from occasion.bases.modelagem import montar_celulas, montar_ocasioes_refri
from occasion.bases.ocasioes import montar_ocasioes, montar_pessoa_dia
from occasion.config import (
    DADOS_BRUTOS,
    DADOS_INTERIM,
    DADOS_PROCESSADOS,
    EDICAO_TESTE,
    EDICAO_TREINO,
    EDICOES,
    MODELOS,
    RAIZ,
    RELATORIOS,
)
from occasion.inference.predict import salvar_artefato, treinar_baseline
from occasion.inference.treino import treinar_avaliado
from occasion.ingest.parse import gravar_interim_2008, gravar_interim_2017
from occasion.receita import carregar_receita

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


def commit_da_receita(raiz: Path) -> str:
    """Saída de `git describe --always --dirty --abbrev=12`; `"desconhecido"` se o git falhar.

    O sufixo `-dirty` indica alterações não commitadas: o commit sozinho não contém a receita.
    """
    try:
        saida = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            cwd=raiz,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        saida = ""
    if not saida:
        print(
            "aviso: não foi possível obter o commit da receita (git ausente ou fora de um "
            "repositório); gravando 'desconhecido'",
            file=sys.stderr,
        )
        return "desconhecido"
    return saida


def _baseline(pasta_dados: Path = DADOS_PROCESSADOS, pasta_modelos: Path = MODELOS) -> None:
    """Ajusta o degrau A e grava em `<modelos>/baseline`, nunca no lugar do modelo avaliado."""
    celulas = pd.read_parquet(pasta_dados / f"celulas_{EDICAO_TREINO}.parquet")
    ocasioes_refri = pd.read_parquet(pasta_dados / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    commit = commit_da_receita(RAIZ)
    artefato = treinar_baseline(celulas, ocasioes_refri, commit)
    destino = salvar_artefato(artefato, pasta_modelos / "baseline")
    print(f"gravado: {destino}")
    print(f"taxa geral, parte 1: {artefato.modelo_p1.taxa_geral_:.6f}")
    print(f"taxa geral, parte 2: {artefato.modelo_p2.taxa_geral_:.6f}")
    print(f"UFs de mídia regional: {sorted(artefato.ufs_midia_regional)}")


def _avaliado(pasta_modelos: Path = MODELOS) -> None:
    """Monta o artefato avaliado a partir de `reports/receita.json` e grava em `<modelos>/avaliado`.

    O artefato guarda a assinatura da receita da árvore de trabalho, que o teste final exige ser
    a da tag `receita-congelada`. Por isso a árvore precisa estar limpa nos arquivos da receita
    (`ARQUIVOS_DA_RECEITA`, inclusive `reports/receita.json` commitado); senão, nada é gravado.
    """
    # importação local: `experimentos.registro` importa `commit_da_receita` deste módulo
    from occasion.experimentos.registro import carregar_resultado

    try:
        assinatura = assinatura_da_receita(RAIZ)
    except TesteBloqueado as erro:
        sys.exit(f"make avaliado recusado: {erro}")
    receita = carregar_receita(RELATORIOS / "receita.json")
    celulas = carregar_base("celulas", EDICAO_TREINO)
    ocasioes_refri = carregar_base("ocasioes_refri", EDICAO_TREINO)
    partes = ["ocasiao", "preferencia", *(["direto"] if receita.direto is not None else [])]
    selecoes = {parte: carregar_resultado(f"selecao_{parte}") for parte in partes}
    commit = commit_da_receita(RAIZ)
    artefato = treinar_avaliado(receita, celulas, ocasioes_refri, selecoes, commit, assinatura)
    destino = salvar_artefato(artefato, pasta_modelos / "avaliado")
    print(f"gravado: {destino}")
    print(f"degrau: {artefato.metadados.degrau}")
    print(f"calibração do score: {artefato.calibracao_score}")
    print(f"UFs de mídia regional: {sorted(artefato.ufs_midia_regional)}")
    print(f"assinatura da receita: {assinatura}")


def _servido(pasta_modelos: Path = MODELOS) -> None:
    """Reajusta a receita para a aplicação e grava em `<modelos>/servido` (seção 12 do design).

    Só com `reports/teste_final.json` commitado; os dados de treino vêm de H9 (`pos_teste.json`).
    """
    # importação local: `experimentos` importa `commit_da_receita` deste módulo
    from occasion.inference.servido import ARQUIVO_DO_POS_TESTE, treinar_servido

    try:
        exigir_receita_intacta(RAIZ)
        # as travas vêm antes de qualquer leitura do alvo de 2017-2018 e de `pos_teste.json`
        exigir_teste_commitado(RAIZ)
        exigir_teste_commitado(RAIZ, ARQUIVO_DO_POS_TESTE)
        receita = carregar_receita(RELATORIOS / "receita.json")
        pos_teste = json.loads((RELATORIOS / "pos_teste.json").read_text(encoding="utf-8"))
        teste_final = json.loads((RELATORIOS / "teste_final.json").read_text(encoding="utf-8"))
        bases = {
            f"{nome}_{edicao}": carregar_base(nome, edicao)
            for nome in ("celulas", "ocasioes_refri")
            for edicao in (EDICAO_TREINO, EDICAO_TESTE)
        }
        commit = commit_da_receita(RAIZ)
        artefato = treinar_servido(receita, bases, pos_teste, teste_final, commit)
    except TesteBloqueado as erro:
        sys.exit(f"make servido recusado: {erro}")
    destino = salvar_artefato(artefato, pasta_modelos / "servido")
    print(f"gravado: {destino}")
    print(f"treinado em: {artefato.metadados.treinado_em}")
    print(f"calibração do score: {artefato.calibracao_score}")
    print(f"UFs de mídia regional: {sorted(artefato.ufs_midia_regional)}")


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
    subcomandos.add_parser("auditoria", help="audita os dados e grava reports/auditoria.json")
    subcomandos.add_parser("baseline", help="treina o degrau A em 2008-2009 e grava o artefato")
    subcomandos.add_parser(
        "avaliado", help="monta o artefato avaliado a partir de reports/receita.json"
    )
    subcomandos.add_parser("servido", help="reajusta a receita para a aplicação (modelo servido)")
    args = parser.parse_args(argv)
    if args.comando == "interim":
        _interim(args.edicao)
    elif args.comando == "ocasioes":
        for edicao in [args.edicao] if args.edicao else EDICOES:
            _ocasioes(edicao)
    elif args.comando == "modelagem":
        for edicao in [args.edicao] if args.edicao else EDICOES:
            _modelagem(edicao)
    elif args.comando == "auditoria":
        rodar_auditoria()
        print(f"gravado: {RELATORIOS / 'auditoria.json'}")
    elif args.comando == "baseline":
        _baseline()
    elif args.comando == "avaliado":
        _avaliado()
    elif args.comando == "servido":
        _servido()


if __name__ == "__main__":
    main()
