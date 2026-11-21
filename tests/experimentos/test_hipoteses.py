"""Relatório de hipóteses com JSONs de etapa falsos em `tmp_path` (não toca `reports/`)."""

import json
from pathlib import Path
from typing import Any

import pytest

from occasion.experimentos import hipoteses
from occasion.experimentos.hipoteses import NAO_EXECUTADA, consolidar


def _iv(est: float, inf: float, sup: float) -> dict[str, Any]:
    return {"b": 1000, "estimativa": est, "ic_inf": inf, "ic_sup": sup, "nivel": 0.95}


def _gravar(pasta: Path, relativo: str, metricas: dict[str, Any]) -> None:
    caminho = pasta / relativo
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps({"metricas": metricas}), encoding="utf-8")


def _comp(cand: str, ref: str, iv: dict[str, Any], veredito: str) -> dict[str, Any]:
    return {"candidato": cand, "referencia": ref, "intervalo": iv, "veredito": veredito}


def _tudo(pasta: Path) -> None:
    for parte in ("ocasiao", "preferencia"):
        _gravar(
            pasta,
            f"experimentos/escada_{parte}.json",
            {
                "vencedor": "C",
                "comparacoes": {
                    "H1": _comp("B", "A", _iv(0.01, 0.005, 0.02), "confirmada"),
                    "H2": _comp("C", "B", _iv(0.001, 0.0002, 0.002), "confirmada"),
                    "H3": _comp("D", "C", _iv(0.0, -0.001, 0.001), "refutada"),
                },
            },
        )
    _gravar(
        pasta,
        "experimentos/h4_tem_pizza.json",
        {"intervalo": _iv(-0.3, -0.6, -0.05), "veredito": "confirmada"},
    )
    _gravar(
        pasta,
        "experimentos/h5_peso.json",
        {
            parte: {
                "ganho_sem_peso": _iv(0.0, -0.1, 0.1),
                "veredito": "sem evidência contra o padrão",
                "usar_peso": True,
            }
            for parte in ("ocasiao", "preferencia")
        },
    )
    _gravar(
        pasta,
        "experimentos/h6_h7_produto.json",
        {
            "H6": {
                "ganho_do_direto": _iv(-0.0001, -0.0002, -0.00001),
                "veredito": "sem evidência contra o padrão",
            },
            "H7": {
                "ganho_corrigida": _iv(-0.000005, -0.00002, 0.000003),
                "veredito": "refutada",
                "calibrar_score": False,
                "inclinacao": _iv(1.0, 0.9, 1.1),
            },
        },
    )
    _gravar(
        pasta,
        "teste_final.json",
        {
            "H8": {
                "ganho": _iv(0.0, -0.001, 0.001),
                "pr_auc": {},
                "referencia": "x",
                "veredito": "refutada",
            }
        },
    )
    nomes = ("ocasiao", "preferencia", "preferencia_com_pizza")
    _gravar(
        pasta,
        "pos_teste.json",
        {
            "H9": {
                "comparacoes": {
                    n: {
                        "ganho": _iv(0.0, -0.01, 0.01),
                        "veredito": "sem evidência contra o padrão",
                    }
                    for n in nomes
                },
                "treino_servido": "base_inteira",
            }
        },
    )


def _por_id(res: dict[str, Any], id_: str) -> list[dict[str, Any]]:
    return [h for h in res["hipoteses"] if h["id"] == id_]


def test_cada_hipotese_aponta_para_a_fonte(tmp_path: Path) -> None:
    _tudo(tmp_path)
    res = consolidar(tmp_path)
    ids = {h["id"] for h in res["hipoteses"]}
    assert ids == {f"H{i}" for i in range(1, 11)}
    esperado = {
        "H4": "experimentos/h4_tem_pizza.json",
        "H5": "experimentos/h5_peso.json",
        "H6": "experimentos/h6_h7_produto.json",
        "H7": "experimentos/h6_h7_produto.json",
        "H8": "teste_final.json",
        "H9": "pos_teste.json",
    }
    for id_, fonte in esperado.items():
        assert all(h["fonte"] == fonte for h in _por_id(res, id_))
    for id_ in ("H1", "H2", "H3"):
        entradas = _por_id(res, id_)
        assert {h["parte"] for h in entradas} == {"ocasiao", "preferencia"}
        assert {h["fonte"] for h in entradas} == {
            "experimentos/escada_ocasiao.json",
            "experimentos/escada_preferencia.json",
        }
    assert len(_por_id(res, "H9")) == 3


def test_campos_e_intervalo_copiados(tmp_path: Path) -> None:
    _tudo(tmp_path)
    h = _por_id(consolidar(tmp_path), "H4")[0]
    assert (h["estimativa"], h["ic_inf"], h["ic_sup"]) == (-0.3, -0.6, -0.05)
    assert h["largura"] == pytest.approx(0.55)
    for campo in ("enunciado", "teste", "metrica", "veredito", "consequencia", "fonte"):
        assert h[campo]


def test_hipotese_ausente_sai_nao_executada(tmp_path: Path) -> None:
    _tudo(tmp_path)
    (tmp_path / "experimentos/h4_tem_pizza.json").unlink()
    (tmp_path / "teste_final.json").unlink()
    res = consolidar(tmp_path)
    for id_ in ("H4", "H8"):
        h = _por_id(res, id_)[0]
        assert h["veredito"] == NAO_EXECUTADA
        assert h["motivo"]
        assert h["estimativa"] is None
    assert _por_id(res, "H1")[0]["veredito"] == "confirmada"


def test_pasta_vazia_nao_executa_nada(tmp_path: Path) -> None:
    res = consolidar(tmp_path)
    assert {h["id"] for h in res["hipoteses"]} == {f"H{i}" for i in range(1, 11)}
    assert all(h["veredito"] == NAO_EXECUTADA for h in res["hipoteses"])


def test_h10_cortada_tem_motivo_mesmo_sem_json(tmp_path: Path) -> None:
    h = _por_id(consolidar(tmp_path), "H10")[0]
    assert h["veredito"] == NAO_EXECUTADA
    assert "Google Trends" in h["motivo"]
    assert "interesse de busca" in h["motivo"]


def test_h5_h6_h9_sem_evidencia_contra_o_padrao_nunca_confirmada(tmp_path: Path) -> None:
    _tudo(tmp_path)
    res = consolidar(tmp_path)
    for id_ in ("H5", "H6", "H9"):
        for h in _por_id(res, id_):
            assert h["veredito"] == "sem evidência contra o padrão"
            assert h["veredito"] != "confirmada"


def test_veredito_vem_do_json_nao_do_intervalo(tmp_path: Path) -> None:
    # intervalo inteiro acima de zero, mas o JSON diz "refutada": a cópia vale, não o recálculo
    _tudo(tmp_path)
    _gravar(
        tmp_path,
        "experimentos/escada_ocasiao.json",
        {
            "comparacoes": {
                "H1": _comp("B", "A", _iv(0.5, 0.4, 0.6), "refutada"),
                "H2": _comp("C", "B", _iv(0.5, 0.4, 0.6), "refutada"),
                "H3": _comp("D", "C", _iv(-0.5, -0.6, -0.4), "confirmada"),
            }
        },
    )
    res = consolidar(tmp_path)
    ocasiao = {h["id"]: h for h in res["hipoteses"] if h["parte"] == "ocasiao"}
    assert ocasiao["H1"]["veredito"] == "refutada"
    assert ocasiao["H3"]["veredito"] == "confirmada"


def test_consequencia_segue_o_json(tmp_path: Path) -> None:
    _tudo(tmp_path)
    res = consolidar(tmp_path)
    assert "(C) superam" in _por_id(res, "H2")[0]["consequencia"]
    assert "duas edições" in _por_id(res, "H9")[0]["consequencia"]
    assert "com peso" in _por_id(res, "H5")[0]["consequencia"]


def test_rodar_hipoteses_grava_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _tudo(tmp_path)
    monkeypatch.setattr(hipoteses, "RELATORIOS", tmp_path)
    caminho = hipoteses.rodar_hipoteses()
    assert caminho == tmp_path / "hipoteses.json"
    assert len(json.loads(caminho.read_text(encoding="utf-8"))["hipoteses"]) == 16


def test_escada_usa_o_vencedor_do_json(tmp_path: Path) -> None:
    _tudo(tmp_path)
    res = consolidar(tmp_path)
    for id_ in ("H1", "H2", "H3"):
        for h in _por_id(res, id_):
            assert h["vencedor"] == "C"
            assert "vencedor da escada" in h["consequencia"]
            assert h["consequencia"].endswith(": C.")


def test_h8_teste_e_avaliacao_temporal_e_consequencia_segue_o_veredito(tmp_path: Path) -> None:
    _tudo(tmp_path)
    h = _por_id(consolidar(tmp_path), "H8")[0]
    assert "temporal" in h["teste"] and "2008-2009" in h["teste"]
    assert "Se refutada" not in h["consequencia"]
    assert "não superou" in h["consequencia"]
    _gravar(
        tmp_path,
        "teste_final.json",
        {
            "H8": {
                "ganho": _iv(0.1, 0.05, 0.2),
                "pr_auc": {},
                "referencia": "x",
                "veredito": "confirmada",
            }
        },
    )
    h = _por_id(consolidar(tmp_path), "H8")[0]
    assert "superou" in h["consequencia"] and "não superou" not in h["consequencia"]
    assert "limitação" not in h["consequencia"]


@pytest.mark.parametrize(
    ("arquivo", "id_"),
    [
        ("experimentos/h4_tem_pizza.json", "H4"),
        ("teste_final.json", "H8"),
        ("pos_teste.json", "H9"),
    ],
)
def test_chave_ausente_em_json_existente_da_erro_claro(
    tmp_path: Path, arquivo: str, id_: str
) -> None:
    _tudo(tmp_path)
    _gravar(tmp_path, arquivo, {"H8": {}, "H9": {}})
    with pytest.raises(KeyError, match=arquivo.replace(".", r"\.")) as erro:
        consolidar(tmp_path)
    assert "chave ausente" in str(erro.value)
