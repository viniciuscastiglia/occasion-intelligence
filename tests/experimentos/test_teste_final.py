"""Teste final (H8) em repositório git sintético de `tmp_path`, sem tocar o repositório real.

Cada teste monta um repositório com o código da receita (arquivos fictícios), `reports/receita.json`,
os relatórios de validação, as bases sintéticas das duas edições e o artefato avaliado.
"""

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.avaliacao.segmentos import colunas_de_segmento, metricas_por_segmento
from occasion.bases import acesso
from occasion.bases.acesso import assinatura_da_receita
from occasion.config import EDICAO_TESTE, EDICAO_TREINO, FAIXAS_HORARIO, TAG_RECEITA
from occasion.decision.engine import CORTE_OCASIAO, CORTE_PREFERENCIA
from occasion.experimentos import teste_final
from occasion.experimentos.especificacao import variaveis_da_parte
from occasion.experimentos.registro import _para_json
from occasion.experimentos.teste_final import posicoes_na_referencia, rodar_teste_final
from occasion.inference.predict import ArtefatoModelo, salvar_artefato, treinar_baseline
from occasion.inference.treino import treinar_avaliado
from occasion.models.base import sigmoide
from occasion.receita import Receita, ReceitaParte, salvar_receita
from occasion.score.score import CORTE_ALTA, CORTE_MEDIA, construir_referencia
from occasion.validacao.metricas import posicoes_ponderadas, resumo

B_TESTE = 20  # réplicas do bootstrap nos testes (o teste final real usa B = 1000)

_GIT = [
    "git",
    "-c",
    "user.name=teste",
    "-c",
    "user.email=teste@exemplo.invalid",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "tag.gpgsign=false",
]

_CODIGO = {
    "src/occasion/config.py": "SEMENTE = 20261112\n",
    "src/occasion/models/taxas.py": "class TabelaDeTaxas: ...\n",
    "src/occasion/score/score.py": "CORTE_ALTA = 80.0\n",
    "src/occasion/trends/coleta.py": "def coletar(): ...\n",
    "pyproject.toml": "[project]\nname = 'sintetico'\n",
    "uv.lock": "version = 1\n",
    ".gitignore": "data/\n/models/\n",
}
_A = {
    "ocasiao": {"dimensoes": ["faixa_horario", "local"], "suavizacao": 100},
    "preferencia": {"dimensoes": ["regiao"], "suavizacao": 1000},
}


# ------------------------------------------------------------------------------ bases sintéticas


def _com_setores(n: int, semente: int, edicao: str) -> pd.DataFrame:
    X = contexto_sintetico(n, semente)
    rng = np.random.default_rng(semente + 5)
    X["cod_upa"] = [f"{uf}-{rng.integers(0, 5)}" for uf in X["uf"]]
    X["peso_final"] = rng.uniform(0.5, 3.0, size=n)
    X["edicao"] = edicao
    return X


def _celulas(edicao: str, semente: int) -> pd.DataFrame:
    X = _com_setores(1500, semente, edicao)
    rng = np.random.default_rng(semente + 1)
    z = -1.2 + 1.4 * (X["faixa_horario"] == "noite") - 0.02 * (X["idade"] - 40)
    X["y_pizza_refri"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    norte = (X["regiao"] == "norte").to_numpy()
    p_guarana = np.where(norte, 0.6, 0.3)
    X["y_pizza_guarana"] = ((rng.random(len(X)) < p_guarana) & (X["y_pizza_refri"] == 1)).astype(
        int
    )
    return X


def _ocasioes(edicao: str, semente: int) -> pd.DataFrame:
    X = _com_setores(1000, semente, edicao)
    rng = np.random.default_rng(semente + 1)
    X["tem_pizza"] = (rng.random(len(X)) < 0.4).astype(int)
    z = -1.0 - 0.4 * X["tem_pizza"] + 1.0 * (X["regiao"] == "norte")
    X["y_guarana"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    return X


def _bases() -> dict[tuple[str, str], pd.DataFrame]:
    return {
        ("celulas", EDICAO_TREINO): _celulas(EDICAO_TREINO, 10),
        ("ocasioes_refri", EDICAO_TREINO): _ocasioes(EDICAO_TREINO, 20),
        ("celulas", EDICAO_TESTE): _celulas(EDICAO_TESTE, 30),
        ("ocasioes_refri", EDICAO_TESTE): _ocasioes(EDICAO_TESTE, 40),
    }


# ------------------------------------------------------------------------------ receita e relatórios


def _parte(degrau: str) -> ReceitaParte:
    parametros: dict[str, Any] = {"C": 1.0} if degrau == "B" else {}
    return ReceitaParte(degrau=degrau, parametros=parametros, usar_peso=True, calibrar=False)  # type: ignore[arg-type]


def _receita(degrau: str = "B") -> Receita:
    def parte(nome: str) -> ReceitaParte:
        rp = _parte(degrau)
        return rp if degrau == "B" else rp.model_copy(update={"parametros": _A[nome]})

    return Receita(
        faixas_horario={nome: list(faixa) for nome, faixa in FAIXAS_HORARIO.items()},
        variaveis={p: variaveis_da_parte(p) for p in ("ocasiao", "preferencia", "direto")},
        interacoes={"ocasiao": [], "preferencia": [], "direto": []},
        ocasiao=parte("ocasiao"),
        preferencia=parte("preferencia"),
        score="produto",
        direto=None,
        calibrar_score=False,
        referencia_a=_A,  # type: ignore[arg-type]
        cortes={
            "alta": CORTE_ALTA,
            "media": CORTE_MEDIA,
            "ocasiao": CORTE_OCASIAO,
            "preferencia": CORTE_PREFERENCIA,
        },
        experimentos=[],
    )


def _gravar_json(caminho: Path, conteudo: dict[str, Any]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(json.dumps(_para_json(conteudo), indent=2, sort_keys=True), "utf-8")


def _avaliar(base: pd.DataFrame, alvo: str, p: np.ndarray, com_posicao: bool) -> dict[str, Any]:
    y, w = base[alvo].to_numpy(), base["peso_final"].to_numpy()
    posicao = posicoes_ponderadas(p, w) if com_posicao else None
    return {
        "resumo": resumo(y, p, w, posicao=posicao),
        "segmentos": metricas_por_segmento(y, p, w, colunas_de_segmento(base)),
    }


def _relatorios_de_validacao(repo: Path, receita: Receita, celulas: pd.DataFrame, oc: pd.DataFrame):
    """Relatórios de 2008-2009 com a estrutura das etapas, sobre previsões fictícias."""
    rng = np.random.default_rng(7)
    p_oc = np.clip(0.3 + rng.normal(0, 0.05, len(celulas)), 0.01, 0.99)
    p_pref = np.clip(0.3 + rng.normal(0, 0.05, len(oc)), 0.01, 0.99)
    p_score = np.clip(0.08 + rng.normal(0, 0.02, len(celulas)), 0.001, 0.99)
    com_pizza = (oc["tem_pizza"] == 1).to_numpy()
    pasta = repo / "reports" / "experimentos"
    ok = {"status": "ok"}
    a_oc = _avaliar(celulas, "y_pizza_refri", p_oc, False)
    a_pref = _avaliar(oc, "y_guarana", p_pref, False)
    a_pref_pizza = _avaliar(
        cast(pd.DataFrame, oc[com_pizza]), "y_guarana", p_pref[com_pizza], False
    )
    a_score = _avaliar(celulas, "y_pizza_guarana", p_score, True)
    for parte, degrau in (("ocasiao", receita.ocasiao.degrau), ("preferencia", "B")):
        extra = {"resumo_com_pizza": a_pref_pizza["resumo"]} if parte == "preferencia" else {}
        resumo_parte = a_oc["resumo"] if parte == "ocasiao" else a_pref["resumo"]
        _gravar_json(
            pasta / f"escada_{parte}.json",
            {
                "metricas": {
                    **ok,
                    "vencedor": degrau if parte == "ocasiao" else receita.preferencia.degrau,
                    "degraus": {d: {"resumo": resumo_parte, **extra} for d in "ABCD"},
                }
            },
        )
    _gravar_json(
        pasta / "h6_h7_produto.json",
        {
            "metricas": {
                **ok,
                "score": "produto",
                "H6": {
                    "log_loss": {"produto": 0.3, "direto": 0.31},
                    "pr_auc": {"produto": 0.2, "direto": 0.19},
                    "roc_auc": {"produto": 0.7, "direto": 0.69},
                },
                "H7": {"calibrar_score": False, "log_loss_antes": 0.3, "log_loss_depois": 0.3},
            }
        },
    )
    _gravar_json(
        pasta / "cortes.json", {"metricas": {**ok, "score": "produto", "resumo": a_score["resumo"]}}
    )
    _gravar_json(
        repo / "reports" / "avaliacao_2008.json",
        {
            "metricas": {
                **ok,
                "ocasiao": a_oc,
                "preferencia": {"todas": a_pref, "com_pizza": a_pref_pizza},
                "score": a_score,
            }
        },
    )


# ------------------------------------------------------------------------------ repositório git


def _git(repo: Path, *args: str) -> None:
    subprocess.run([*_GIT, *args], cwd=repo, check=True, capture_output=True)


def _escrever(repo: Path, caminho: str, conteudo: str) -> None:
    destino = repo / caminho
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(conteudo, encoding="utf-8")


def _commitar(repo: Path, mensagem: str) -> None:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", mensagem)


def _montar_repo(repo: Path, receita: Receita) -> Path:
    repo.mkdir()
    _git(repo, "init", "-q")
    for caminho, conteudo in _CODIGO.items():
        _escrever(repo, caminho, conteudo)
    salvar_receita(receita, repo / "reports" / "receita.json")
    bases = _bases()
    processados = repo / "data" / "processed"
    processados.mkdir(parents=True)
    for (nome, edicao), base in bases.items():
        base.to_parquet(processados / f"{nome}_{edicao}.parquet")
    _relatorios_de_validacao(
        repo,
        receita,
        bases[("celulas", EDICAO_TREINO)],
        bases[("ocasioes_refri", EDICAO_TREINO)],
    )
    _commitar(repo, "receita")
    return repo


def _selecoes(receita: Receita, n_celulas: int, n_ocasioes: int) -> dict[str, Any]:
    def selecao(rp: ReceitaParte, n: int, extras: dict[str, Any]) -> dict[str, Any]:
        oof = np.full(n, 0.3)
        return {"degrau": rp.degrau, "parametros": rp.parametros, "oof": oof,
                "oof_extras": extras, "hash_dobras": "h"}  # fmt: skip

    return {
        "ocasiao": selecao(receita.ocasiao, n_celulas, {}),
        "preferencia": selecao(
            receita.preferencia, n_ocasioes, {"celulas": np.full(n_celulas, 0.3)}
        ),
    }


def _artefato(repo: Path, receita: Receita, assinatura: str | None) -> ArtefatoModelo:
    celulas = pd.read_parquet(repo / "data" / "processed" / f"celulas_{EDICAO_TREINO}.parquet")
    oc = pd.read_parquet(repo / "data" / "processed" / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    selecoes = _selecoes(receita, len(celulas), len(oc))
    return treinar_avaliado(receita, celulas, oc, selecoes, "abc", assinatura)


def _salvar(repo: Path, artefato: ArtefatoModelo) -> None:
    salvar_artefato(artefato, repo / "models" / "avaliado")


@pytest.fixture
def repo_sem_tag(tmp_path: Path) -> Path:
    repo = _montar_repo(tmp_path / "repo", _receita())
    _salvar(repo, _artefato(repo, _receita(), assinatura_da_receita(repo)))
    return repo


@pytest.fixture
def repo_com_tag(tmp_path: Path) -> Path:
    repo = _montar_repo(tmp_path / "repo", _receita())
    _git(repo, "tag", TAG_RECEITA)
    return repo


@pytest.fixture
def artefato_da_tag(repo_com_tag: Path) -> Path:
    artefato = _artefato(repo_com_tag, _receita(), assinatura_da_receita(repo_com_tag, TAG_RECEITA))
    _salvar(repo_com_tag, artefato)
    return repo_com_tag


@pytest.fixture
def artefato_de_outra_receita(repo_com_tag: Path) -> Path:
    _salvar(repo_com_tag, _artefato(repo_com_tag, _receita(), "0" * 64))
    return repo_com_tag


@pytest.fixture
def artefato_baseline(repo_com_tag: Path) -> Path:
    """O degrau A do Plano 1 no lugar do avaliado, até com a assinatura certa."""
    processados = repo_com_tag / "data" / "processed"
    celulas = pd.read_parquet(processados / f"celulas_{EDICAO_TREINO}.parquet")
    oc = pd.read_parquet(processados / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    baseline = dataclasses.replace(
        treinar_baseline(celulas, oc, "abc"),
        assinatura_receita=assinatura_da_receita(repo_com_tag, TAG_RECEITA),
    )
    _salvar(repo_com_tag, baseline)
    return repo_com_tag


@pytest.fixture
def repo_com_tag_e_alteracao_em_models(artefato_da_tag: Path) -> Path:
    _escrever(artefato_da_tag, "src/occasion/models/taxas.py", "class TabelaDeTaxas: pass\n")
    _commitar(artefato_da_tag, "muda o modelo depois da tag")
    return artefato_da_tag


@pytest.fixture
def repo_com_tag_e_alteracao_no_trends(artefato_da_tag: Path) -> Path:
    _escrever(artefato_da_tag, "src/occasion/trends/coleta.py", "def coletar(): return 1\n")
    _commitar(artefato_da_tag, "muda o trends depois da tag")
    return artefato_da_tag


@pytest.fixture
def repo_com_tag_e_arquivo_novo_em_models(artefato_da_tag: Path) -> Path:
    _escrever(artefato_da_tag, "src/occasion/models/novo.py", "X = 1\n")
    return artefato_da_tag


@pytest.fixture
def sem_leitura_de_bases(monkeypatch: pytest.MonkeyPatch) -> None:
    """Falha o teste se o teste final chegar a ler qualquer base."""

    def proibido(*args: Any, **kwargs: Any) -> pd.DataFrame:
        raise AssertionError(f"base lida antes das conferências: {args}")

    monkeypatch.setattr(teste_final, "carregar_base", proibido)


def _json(caminho: Path) -> dict[str, Any]:
    return json.loads(caminho.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------ testes


def test_bloqueia_sem_tag(repo_sem_tag: Path, sem_leitura_de_bases: None) -> None:
    with pytest.raises(acesso.TesteBloqueado, match=TAG_RECEITA):
        rodar_teste_final(repo_sem_tag, b=B_TESTE)
    assert not (repo_sem_tag / "reports" / "teste_final.json").exists()


def test_bloqueia_com_receita_alterada_depois_da_tag(
    repo_com_tag_e_alteracao_em_models: Path, sem_leitura_de_bases: None
) -> None:
    with pytest.raises(acesso.TesteBloqueado, match="mudou"):
        rodar_teste_final(repo_com_tag_e_alteracao_em_models, b=B_TESTE)


def test_bloqueia_alteracao_nao_commitada_na_receita(
    artefato_da_tag: Path, sem_leitura_de_bases: None
) -> None:
    _escrever(artefato_da_tag, "reports/receita.json", "{}\n")
    with pytest.raises(acesso.TesteBloqueado):
        rodar_teste_final(artefato_da_tag, b=B_TESTE)


def test_aceita_alteracao_fora_da_receita_depois_da_tag(
    repo_com_tag_e_alteracao_no_trends: Path,
) -> None:
    caminho = rodar_teste_final(repo_com_tag_e_alteracao_no_trends, b=B_TESTE)
    assert caminho == repo_com_tag_e_alteracao_no_trends / "reports" / "teste_final.json"
    assert caminho.exists()


def test_bloqueia_arquivo_nao_rastreado_na_receita(
    repo_com_tag_e_arquivo_novo_em_models: Path, sem_leitura_de_bases: None
) -> None:
    with pytest.raises(acesso.TesteBloqueado, match="novo.py"):
        rodar_teste_final(repo_com_tag_e_arquivo_novo_em_models, b=B_TESTE)


def test_bloqueia_artefato_com_outra_assinatura(
    repo_com_tag: Path, artefato_de_outra_receita: Path, sem_leitura_de_bases: None
) -> None:
    with pytest.raises(acesso.TesteBloqueado, match="assinatura"):
        rodar_teste_final(repo_com_tag, b=B_TESTE)


def test_bloqueia_artefato_do_degrau_a_no_lugar_do_avaliado(
    repo_com_tag: Path, artefato_baseline: Path, sem_leitura_de_bases: None
) -> None:
    with pytest.raises(acesso.TesteBloqueado, match="degrau"):
        rodar_teste_final(repo_com_tag, b=B_TESTE)


def test_bloqueia_sem_o_artefato_avaliado(repo_com_tag: Path, sem_leitura_de_bases: None) -> None:
    with pytest.raises(acesso.TesteBloqueado, match="make avaliado"):
        rodar_teste_final(repo_com_tag, b=B_TESTE)


def test_segunda_execucao_nao_sobrescreve_e_confere(
    repo_com_tag: Path, artefato_da_tag: Path
) -> None:
    relatorios = repo_com_tag / "reports"
    primeiro = rodar_teste_final(repo_com_tag, b=B_TESTE)
    original = primeiro.read_text(encoding="utf-8")
    segundo = rodar_teste_final(repo_com_tag, b=B_TESTE)
    assert segundo == relatorios / "teste_final_reexecucao.json"
    assert primeiro.read_text(encoding="utf-8") == original
    reexec = json.loads((relatorios / "teste_final_reexecucao.json").read_text())
    assert reexec["identico_ao_original"] is True
    assert reexec["chaves_diferentes"] == []


def test_relatorio_do_teste_final(repo_com_tag: Path, artefato_da_tag: Path) -> None:
    d = _json(rodar_teste_final(repo_com_tag, b=B_TESTE))
    assert d["nome"] == "teste_final"
    assert d["natureza"] == "confirmatório"
    assert d["assinatura_receita"] == assinatura_da_receita(repo_com_tag, TAG_RECEITA)
    assert d["configuracao"]["b"] == B_TESTE
    m = d["metricas"]
    h8 = m["H8"]
    assert h8["referencia"] == "produto das tabelas do degrau A"
    assert h8["veredito"] in {"confirmada", "refutada"}
    assert h8["ganho"]["b"] <= B_TESTE
    assert h8["ganho"]["estimativa"] == pytest.approx(
        h8["pr_auc"]["receita"] - h8["pr_auc"]["referencia"]
    )
    assert set(m["partes"]) == {"ocasiao", "preferencia_todas", "preferencia_com_pizza"}
    for nome in ("score", "ocasiao", "preferencia_todas", "preferencia_com_pizza"):
        assert set(m["resumos"][nome]) == {"receita", "degrau_a", "constante"}
        assert m["segmentos"][nome]
        assert set(m["queda_por_segmento"][nome][0]) >= {"segmento", "valor", "validacao", "teste"}
    assert m["resumos"]["score"]["receita"]["pr_auc"] == pytest.approx(h8["pr_auc"]["receita"])
    assert set(m["concentracao_topo"]["score"]) == {"receita", "degrau_a", "constante"}
    queda = m["queda"]["score"]["receita"]
    assert queda["diferenca"]["pr_auc"] == pytest.approx(
        queda["teste"]["pr_auc"] - queda["validacao"]["pr_auc"]
    )
    assert set(m["erros"]) == {"score", "ocasiao", "preferencia_todas"}
    assert d["dados"]["celulas_teste"]["n"] == 1500


def test_receita_so_de_tabelas_compara_com_a_constante(tmp_path: Path) -> None:
    receita = _receita("A")
    repo = _montar_repo(tmp_path / "repo", receita)
    _git(repo, "tag", TAG_RECEITA)
    _salvar(repo, _artefato(repo, receita, assinatura_da_receita(repo, TAG_RECEITA)))
    d = _json(rodar_teste_final(repo, b=B_TESTE))
    h8 = d["metricas"]["H8"]
    assert h8["referencia"] == "constante"
    celulas = pd.read_parquet(repo / "data" / "processed" / f"celulas_{EDICAO_TREINO}.parquet")
    w = celulas["peso_final"].to_numpy()
    prevalencia = float(np.sum(w * celulas["y_pizza_guarana"]) / np.sum(w))
    assert d["configuracao"]["constante"]["score"] == pytest.approx(prevalencia)


def test_posicoes_na_referencia_iguais_a_posicao_da_referencia() -> None:
    celulas = _celulas(EDICAO_TREINO, 10)
    artefato = treinar_baseline(celulas, _ocasioes(EDICAO_TREINO, 20), "abc")
    ref = construir_referencia(celulas, artefato.modelo_p1, artefato.modelo_p2)
    valores = np.concatenate([ref.produto[::37], [-1.0, 0.0, 2.0], ref.produto[:3] + 1e-12])
    esperado = [ref.posicao(float(v), "produto") for v in valores]
    assert posicoes_na_referencia(ref, valores, "produto") == pytest.approx(esperado)


def test_teste_final_nao_muda_receita_nem_artefato(repo_com_tag: Path, artefato_da_tag: Path):
    receita = (repo_com_tag / "reports" / "receita.json").read_bytes()
    artefato = repo_com_tag / "models" / "avaliado" / "artefato.joblib"
    antes = artefato.read_bytes()
    rodar_teste_final(repo_com_tag, b=B_TESTE)
    assert (repo_com_tag / "reports" / "receita.json").read_bytes() == receita
    assert artefato.read_bytes() == antes
    assert isinstance(joblib.load(artefato), ArtefatoModelo)
