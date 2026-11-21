"""Testes do módulo assinado do artefato e da compatibilidade com `predict` (nomes e pickles).

Os `joblib.load` daqui só leem arquivos gravados pelo próprio teste ou pelo pipeline do projeto
(`make baseline`) nesta máquina; nenhum vem de fora.
"""

from pathlib import Path

import joblib
import pytest
from tests.inference.test_predict import _artefato_unico

from occasion.config import MODELOS
from occasion.inference import artefato, contract, predict
from occasion.inference.artefato import NOME_ARQUIVO, ArtefatoModelo, pontuar


def test_predict_reexporta_os_nomes_do_artefato() -> None:
    for nome in (
        "ArtefatoModelo", "pontuar", "NOME_ARQUIVO", "N_UFS_MIDIA_REGIONAL",
        "_exigir_so_treino", "_ufs_de_midia_regional",
    ):  # fmt: skip
        assert getattr(predict, nome) is getattr(artefato, nome)
    assert predict.VERSAO_CONTRATO is contract.VERSAO_CONTRATO
    assert ArtefatoModelo.__module__ == "occasion.inference.artefato"


def test_pickle_gravado_com_o_caminho_antigo_carrega(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Um artefato gravado quando a classe morava em `predict` continua carregando."""
    original = _artefato_unico()
    caminho = tmp_path / NOME_ARQUIVO
    with monkeypatch.context() as m:
        m.setattr(ArtefatoModelo, "__module__", "occasion.inference.predict")
        joblib.dump(original, caminho)
    conteudo = caminho.read_bytes()
    assert b"occasion.inference.predict" in conteudo
    assert b"occasion.inference.artefato" not in conteudo
    carregado = joblib.load(caminho)
    assert isinstance(carregado, ArtefatoModelo)
    X = original.referencia  # a referência viaja inteira no pickle
    assert (carregado.referencia.produto == X.produto).all()


def test_pickle_novo_grava_o_caminho_do_artefato(tmp_path: Path) -> None:
    original = _artefato_unico()
    destino = predict.salvar_artefato(original, tmp_path)
    assert b"occasion.inference.artefato" in destino.read_bytes()
    assert isinstance(joblib.load(destino), predict.ArtefatoModelo)


@pytest.mark.dados
def test_artefato_baseline_do_plano_1_carrega() -> None:
    """`models/baseline/artefato.joblib` (gravado com a classe em `predict`) carrega e pontua."""
    caminho = MODELOS / "baseline" / NOME_ARQUIVO
    carregado = joblib.load(caminho)
    assert isinstance(carregado, ArtefatoModelo)
    linha = predict.contexto_para_linha(
        contract.Contexto(
            uf="SP", situacao="urbano", dia_semana="sexta", faixa_horario="noite",
            local="em_casa", idade=30, sexo="feminino", faixa_renda="de_1_a_2_sm",
            numero_moradores=3,
        )
    )  # fmt: skip
    assert 0.0 <= float(pontuar(linha, carregado)["score"].iloc[0]) <= 100.0
