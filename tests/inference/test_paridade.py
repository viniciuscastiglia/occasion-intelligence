"""Paridade entre treino e inferência: a linha de `contexto_para_linha` tem as mesmas colunas,
valores e tipos que as linhas de `montar_celulas`, exceto a renda, que vira o ponto médio da faixa.
"""

from typing import Any, cast, get_args

import pandas as pd
import pytest
from tests.bases.test_modelagem import EDICAO, ocasiao_lista, pessoa_dia_minimo
from tests.inference.test_predict import _artefato_unico

from occasion.bases.modelagem import montar_celulas
from occasion.eda.descritivas import faixa_de_renda
from occasion.features.variaveis import COLUNAS_CONTEXTO
from occasion.inference.contract import (
    CODIGO_UF,
    PONTO_MEDIO_RENDA,
    Contexto,
    DiaSemana,
    FaixaRenda,
)
from occasion.inference.predict import contexto_para_linha, pontuar

_DIAS: tuple[str, ...] = get_args(DiaSemana)
_SIGLA_DA_UF = {codigo: sigla for sigla, codigo in CODIGO_UF.items()}


def _faixa_de_renda(valor: float) -> FaixaRenda:
    return cast(FaixaRenda, faixa_de_renda(pd.Series([valor])).iloc[0])


def _celulas_variadas() -> pd.DataFrame:
    """Dias com UF, dia da semana, idade, sexo, situação, renda e moradores variados."""
    n = 5
    pessoa_dia = pessoa_dia_minimo(n_dias=n, pizza_refri=False)
    pessoa_dia["uf"] = [35, 29, 43, 13, 53]
    pessoa_dia["dia_semana"] = [1, 3, 5, 6, 7]
    pessoa_dia["idade"] = [18, 25, 40, 67, 110]
    pessoa_dia["sexo"] = pd.array(["feminino", "masculino"] * 2 + ["feminino"], dtype="string")
    pessoa_dia["situacao"] = pd.array(["urbano", "rural"] * 2 + ["urbano"], dtype="string")
    pessoa_dia["renda_total_uc"] = [300.0, 800.0, 1500.0, 4000.0, 20000.0]
    pessoa_dia["n_moradores_uc"] = [1, 2, 3, 4, 5]
    ocasioes = ocasiao_lista(
        [
            {
                "hora": 20,
                "local": "em_casa",
                "pizza": True,
                "refri": True,
                "id_dia": f"{EDICAO}|111|1|1|1|71",
            }
        ]
    )
    return montar_celulas(pessoa_dia, ocasioes)


def _contexto_da_linha(celula: pd.Series) -> Contexto:
    linha: dict[str, Any] = celula.to_dict()
    uf = _SIGLA_DA_UF[int(linha["uf"])]
    return Contexto(
        uf=uf,  # type: ignore[arg-type]
        situacao=linha["situacao"],
        dia_semana=_DIAS[int(linha["dia_semana"]) - 1],  # type: ignore[arg-type]
        faixa_horario=linha["faixa_horario"],
        local=linha["local"],
        idade=int(linha["idade"]),
        sexo=linha["sexo"],
        faixa_renda=_faixa_de_renda(float(linha["renda_pc_sm"])),
        numero_moradores=int(linha["n_moradores_uc"]),
    )


def _escalar(valor: Any) -> Any:
    return valor.item() if hasattr(valor, "item") else valor


def test_contexto_para_linha_reproduz_as_celulas_de_montar_celulas() -> None:
    celulas = _celulas_variadas()
    assert len(celulas) == 50
    for _, celula in celulas.iterrows():
        ctx = _contexto_da_linha(celula)
        linha = contexto_para_linha(ctx)
        assert list(linha.columns) == COLUNAS_CONTEXTO
        da_linha = linha.iloc[0]
        for coluna in COLUNAS_CONTEXTO:
            if coluna == "renda_pc_sm":
                assert da_linha[coluna] == PONTO_MEDIO_RENDA[ctx.faixa_renda]
                assert _faixa_de_renda(float(da_linha[coluna])) == ctx.faixa_renda
                continue
            esperado, obtido = _escalar(celula[coluna]), _escalar(da_linha[coluna])
            assert obtido == esperado, coluna
            assert type(obtido) is type(esperado), (coluna, type(obtido), type(esperado))


def test_pontuar_da_o_mesmo_resultado_para_celula_e_contexto() -> None:
    artefato = _artefato_unico()
    celulas = _celulas_variadas()
    for _, celula in celulas.iterrows():
        da_celula = cast(pd.DataFrame, celula.to_frame().T[COLUNAS_CONTEXTO]).infer_objects()
        da_inferencia = contexto_para_linha(_contexto_da_linha(celula))
        a = pontuar(da_celula.reset_index(drop=True), artefato)
        b = pontuar(da_inferencia, artefato)
        pd.testing.assert_frame_equal(a, b)


def test_contexto_para_linha_recusa_chaves_fora_de_colunas_contexto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from occasion.inference import predict

    monkeypatch.setattr(predict, "COLUNAS_CONTEXTO", [*COLUNAS_CONTEXTO, "outra_coluna"])
    ctx = Contexto(
        uf="SP",
        situacao="urbano",
        dia_semana="sabado",
        faixa_horario="noite",
        local="em_casa",
        idade=28,
        sexo="feminino",
        faixa_renda="de_2_a_5_sm",
        numero_moradores=3,
    )
    with pytest.raises(ValueError, match="paridade"):
        predict.contexto_para_linha(ctx)
