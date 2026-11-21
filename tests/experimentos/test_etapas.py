"""Teste de integração das etapas de experimento em base sintética, sem tocar `data/` nem `reports/`."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from tests.models.bases_sinteticas import contexto_sintetico

from occasion.config import EDICAO_TREINO, SEMENTE
from occasion.decision.engine import CORTE_OCASIAO, CORTE_PREFERENCIA
from occasion.experimentos import etapas, registro
from occasion.experimentos.aninhada import PrevisaoInterna, ResultadoAninhado
from occasion.experimentos.especificacao import INTERACOES, espaco_lightgbm
from occasion.experimentos.regras import melhor_logistico, padrao, sinal_negativo, superioridade
from occasion.models import escada
from occasion.models.base import logit, sigmoide
from occasion.receita import carregar_receita
from occasion.score.score import CORTE_ALTA, CORTE_MEDIA
from occasion.validacao.bootstrap import Intervalo
from occasion.validacao.dobras import dobra_das_linhas
from occasion.validacao.metricas import log_loss_ponderado, posicoes_ponderadas

CHAVES = ("configuracao", "metricas", "commit")


def _com_setores(n: int, semente: int) -> pd.DataFrame:
    X = contexto_sintetico(n, semente)
    rng = np.random.default_rng(semente + 5)
    X["cod_upa"] = [f"{uf}-{rng.integers(0, 5)}" for uf in X["uf"]]
    X["estrato"] = X["cod_upa"]
    X["peso_final"] = rng.uniform(0.5, 3.0, size=n)
    return X


def _celulas() -> pd.DataFrame:
    X = _com_setores(8000, 0)
    rng = np.random.default_rng(1)
    z = -2.2 + 1.2 * (X["faixa_horario"] == "noite") - 0.02 * (X["idade"] - 40)
    X["y_pizza_refri"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    X["y_pizza_guarana"] = ((rng.random(len(X)) < 0.35) & (X["y_pizza_refri"] == 1)).astype(int)
    return X


def _ocasioes() -> pd.DataFrame:
    X = _com_setores(3000, 2)
    rng = np.random.default_rng(3)
    X["tem_pizza"] = (rng.random(len(X)) < 0.4).astype(int)
    z = -1.0 - 0.6 * X["tem_pizza"] + 0.5 * (X["local"] == "fora_de_casa")
    X["y_guarana"] = (rng.random(len(X)) < sigmoide(np.asarray(z, float))).astype(int)
    return X


@pytest.fixture
def pastas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Aponta as três pastas das etapas para `tmp_path`."""
    processados = tmp_path / "processed"
    processados.mkdir()
    _celulas().to_parquet(processados / f"celulas_{EDICAO_TREINO}.parquet")
    _ocasioes().to_parquet(processados / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    for modulo in (etapas, registro):
        monkeypatch.setattr(modulo, "DADOS_EXPERIMENTOS", tmp_path / "experimentos")
        monkeypatch.setattr(modulo, "RELATORIOS", tmp_path / "reports")
    monkeypatch.setattr(etapas, "DADOS_PROCESSADOS", processados)
    # o espaço do D, só neste teste, tem duas configurações
    monkeypatch.setattr(escada, "espaco_lightgbm", lambda: espaco_lightgbm()[:2])
    return tmp_path


def _json(caminho: Path) -> dict:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _ic(d: dict) -> Intervalo:
    return Intervalo(**d)


def test_dobras_peso_e_escada_da_preferencia(pastas: Path) -> None:
    # as pastas lidas e gravadas pelas etapas são as de `tmp_path`, nunca as do repositório
    assert etapas._pasta_relatorios().is_relative_to(pastas)
    assert etapas.DADOS_EXPERIMENTOS.is_relative_to(pastas)
    assert registro.DADOS_EXPERIMENTOS.is_relative_to(pastas)

    etapas.etapa_dobras()
    dobras = _json(pastas / "reports" / "dobras.json")
    assert all(c in dobras for c in CHAVES)
    assert (pastas / "experimentos" / "dobras.parquet").exists()
    assert dobras["semente"] == dobras["metricas"]["semente_usada"]
    assert dobras["metricas"]["status"] == "ok"
    hash_dobras = dobras["dados"]["hash_dobras"]

    etapas.etapa_peso()
    h5 = _json(pastas / "reports" / "experimentos" / "h5_peso.json")
    assert all(c in h5 for c in CHAVES)
    assert h5["metricas"]["status"] == "ok"
    assert {"ocasiao", "preferencia"} <= set(h5["metricas"])
    assert h5["dados"]["dobras"] == {"hash": hash_dobras, "semente": dobras["semente"]}
    for parte in ("ocasiao", "preferencia"):
        m = h5["metricas"][parte]
        # o veredito segue a regra aplicada ao intervalo gravado
        assert m["usar_peso"] == (padrao(_ic(m["ganho_sem_peso"])) != "padrão refutado")

    etapas.etapa_escada("preferencia")
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert all(c in e for c in CHAVES)
    assert e["dados"]["dobras"]["hash"] == hash_dobras
    m = e["metricas"]
    assert m["status"] == "ok"
    assert set(m["degraus"]) == {"A", "B", "C", "D"}
    for d in m["degraus"].values():
        assert d["tempo_s"] >= 0
        assert len(d["parametros_por_dobra"]) == 5
        assert d["resumo"]["log_loss"] > 0
        assert d["resumo_com_pizza"]["n"] > 0
    for h in m["comparacoes"].values():
        assert h["veredito"] == superioridade(_ic(h["intervalo"]))
    assert m["comparacoes"]["H1"]["candidato"] == "B"
    assert m["comparacoes"]["H1"]["referencia"] == "A"
    assert m["comparacoes"]["H3"]["referencia"] == melhor_logistico(
        _ic(m["comparacoes"]["H2"]["intervalo"])
    )
    confirmados = [
        c["candidato"] for c in m["comparacoes_sequenciais"] if c["veredito"] == "confirmada"
    ]
    assert m["vencedor"] == (confirmados[-1] if confirmados else "A")
    assert (pastas / "experimentos" / "aninhada_preferencia_D.joblib").exists()
    # o B da escada é o B de H5 (o resultado salvo por H5, com o hash das dobras)
    assert (pastas / "experimentos" / "aninhada_preferencia_B.joblib").exists()


def test_escada_falha_registra_o_parcial_e_relanca(
    pastas: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    real = etapas.validacao_aninhada

    def com_falha(base, alvo, degrau, *args, **kwargs):  # type: ignore[no-untyped-def]
        if degrau == "C":
            raise RuntimeError("falha forçada no degrau C")
        return real(base, alvo, degrau, *args, **kwargs)

    monkeypatch.setattr(etapas, "validacao_aninhada", com_falha)
    with pytest.raises(RuntimeError, match="forçada"):
        etapas.etapa_escada("preferencia")
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert all(c in e for c in CHAVES)
    assert e["metricas"]["status"] == "falhou"
    assert e["metricas"]["erro"] == {
        "tipo": "RuntimeError",
        "mensagem": "falha forçada no degrau C",
    }
    assert set(e["metricas"]["degraus_concluidos"]) == {"A", "B"}
    assert e["configuracao"]["parte"] == "preferencia"


def test_h5_com_previsao_nan_registra_a_falha(
    pastas: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    etapas.etapa_dobras()

    def nan(p: np.ndarray, o_que: str) -> np.ndarray:
        raise ValueError(f"previsão fora da dobra com NaN em {o_que}")

    monkeypatch.setattr(etapas, "_sem_nan", nan)
    with pytest.raises(ValueError, match="NaN"):
        etapas.etapa_peso()
    h5 = _json(pastas / "reports" / "experimentos" / "h5_peso.json")
    assert h5["metricas"]["status"] == "falhou"
    assert h5["metricas"]["erro"]["tipo"] == "ValueError"
    # com H5 falho, as etapas seguintes recusam
    with pytest.raises(ValueError, match="falhou"):
        etapas.etapa_escada("preferencia")


def test_escada_recusa_dobras_diferentes_das_de_h5(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    caminho = pastas / "experimentos" / "dobras.parquet"
    dobras = pd.read_parquet(caminho)
    dobras["dobra"] = (dobras["dobra"] + 1) % 5
    dobras.to_parquet(caminho)
    with pytest.raises(ValueError, match="dobras"):
        etapas.etapa_escada("preferencia")
    with pytest.raises(ValueError, match="dobras"):
        etapas.etapa_h4(b=5)
    e = _json(pastas / "reports" / "experimentos" / "escada_preferencia.json")
    assert e["metricas"]["status"] == "falhou"


def test_h4_da_preferencia(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_h4(b=20)
    h4 = _json(pastas / "reports" / "experimentos" / "h4_tem_pizza.json")
    assert all(c in h4 for c in CHAVES)
    m = h4["metricas"]
    assert m["status"] == "ok"
    assert m["veredito"] == sinal_negativo(_ic(m["intervalo"]))
    assert m["razao_de_chances"] == pytest.approx(np.exp(m["coeficiente_tem_pizza"]))
    assert m["intervalo"]["ic_inf"] <= m["intervalo"]["ic_sup"]
    assert m["intervalo"]["b"] > 0
    assert h4["configuracao"]["C"] > 0
    assert "regulariza" in h4["configuracao"]["aviso"]
    assert (
        h4["dados"]["dobras"]["hash"]
        == _json(pastas / "reports" / "dobras.json")["dados"]["hash_dobras"]
    )


def test_embaralhado_em_base_sintetica(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_embaralhado(partes=("preferencia",), usar_peso={"preferencia": True})
    r = _json(pastas / "reports" / "experimentos" / "rotulos_embaralhados.json")
    assert all(c in r for c in CHAVES)
    m = r["metricas"]["preferencia"]
    assert r["metricas"]["status"] == "ok"
    assert m["veredito"] == superioridade(_ic(m["ganho_do_modelo_sobre_a_constante"]))
    # a constante é a prevalência ponderada do treino de cada dobra, conferida à parte
    base = pd.read_parquet(pastas / "processed" / f"ocasioes_refri_{EDICAO_TREINO}.parquet")
    y = np.random.default_rng(SEMENTE).permutation(base["y_guarana"].to_numpy())
    w = base["peso_final"].to_numpy()
    por_linha = dobra_das_linhas(base, pd.read_parquet(pastas / "experimentos" / "dobras.parquet"))
    p = np.empty(len(base))
    for k in range(5):
        dentro = por_linha != k
        p[~dentro] = np.sum(w[dentro] * y[dentro]) / np.sum(w[dentro])
    assert m["log_loss_constante"] == pytest.approx(log_loss_ponderado(y, p, w))


@pytest.mark.dados
def test_rotulos_embaralhados_na_preferencia_nao_vencem_a_constante(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for modulo in (etapas, registro):
        monkeypatch.setattr(modulo, "DADOS_EXPERIMENTOS", tmp_path / "experimentos")
        monkeypatch.setattr(modulo, "RELATORIOS", tmp_path / "reports")
    etapas.etapa_dobras()
    etapas.etapa_embaralhado(partes=("preferencia",), usar_peso={"preferencia": True})
    r = _json(tmp_path / "reports" / "experimentos" / "rotulos_embaralhados.json")
    assert r["metricas"]["preferencia"]["veredito"] == "refutada"


# ------------------------------------------------- calibração, produto (H6 e H7), cortes, receita


def _interno(indices: list[int], p: list[float]) -> PrevisaoInterna:
    return PrevisaoInterna(np.array(indices), np.array(p))


def test_sigmoide_da_dobra_usa_so_as_previsoes_internas() -> None:
    rng = np.random.default_rng(0)
    n = 600
    por_linha = np.repeat([0, 1], n // 2)
    p = rng.uniform(0.05, 0.6, n)
    y = (rng.random(n) < p).astype(float)
    w = rng.uniform(0.5, 2.0, n)
    fora_0, fora_1 = np.flatnonzero(por_linha != 0), np.flatnonzero(por_linha != 1)
    res = ResultadoAninhado(
        oof=p, oof_extras={}, parametros=[{}, {}], modelos=[],
        interno=[PrevisaoInterna(fora_0, p[fora_0]), PrevisaoInterna(fora_1, p[fora_1])],
    )  # fmt: skip
    sigmoides = etapas._sigmoides_por_dobra(res, y, w, por_linha)
    # mudar o alvo das linhas da dobra 0 não muda a sigmoide que a dobra 0 recebe
    y_mudado = y.copy()
    y_mudado[por_linha == 0] = 1 - y_mudado[por_linha == 0]
    assert etapas._sigmoides_por_dobra(res, y_mudado, w, por_linha)[0] == sigmoides[0]
    assert etapas._sigmoides_por_dobra(res, y_mudado, w, por_linha)[1] != sigmoides[1]
    calibrado = etapas._calibrar_por_dobra(p, por_linha, sigmoides)
    a, b = sigmoides[1]
    assert calibrado[por_linha == 1] == pytest.approx(sigmoide(a * logit(p[por_linha == 1]) + b))


def test_sigmoide_recusa_previsao_interna_da_propria_dobra() -> None:
    por_linha = np.array([0, 0, 1, 1])
    res = ResultadoAninhado(
        oof=np.full(4, 0.5), oof_extras={}, parametros=[{}], modelos=[],
        interno=[_interno([0, 2, 3], [0.2, 0.3, 0.4])],  # a linha 0 é da dobra externa 0
    )  # fmt: skip
    with pytest.raises(ValueError, match="dobra externa"):
        etapas._sigmoides_por_dobra(res, np.array([0, 1, 0, 1]), np.ones(4), por_linha)


def test_produto_interno_exige_as_mesmas_linhas_nas_duas_partes() -> None:
    oc = etapas._PrevisoesCelulas(
        np.array([0.1, 0.2, 0.3]), [(np.array([1, 2]), np.array([0.2, 0.3]))]
    )
    pref = etapas._PrevisoesCelulas(
        np.array([0.5, 0.5, 0.4]), [(np.array([1, 2]), np.array([0.5, 0.4]))]
    )
    produto = etapas._produto(oc, pref)
    assert produto.fora == pytest.approx([0.05, 0.1, 0.12])
    assert produto.interno[0][1] == pytest.approx([0.1, 0.12])
    torto = etapas._PrevisoesCelulas(pref.fora, [(np.array([0, 2]), np.array([0.5, 0.4]))])
    with pytest.raises(ValueError, match="mesmas linhas"):
        etapas._produto(oc, torto)


def _parcela_que_muda(
    oc: np.ndarray, pref: np.ndarray, w: np.ndarray, qual: str, d: float
) -> float:
    po, pp = posicoes_ponderadas(oc, w), posicoes_ponderadas(pref, w)
    antes = (po >= CORTE_OCASIAO, pp >= CORTE_PREFERENCIA)
    if qual == "ocasiao":
        depois = (po >= CORTE_OCASIAO + d, antes[1])
    else:
        depois = (antes[0], pp >= CORTE_PREFERENCIA + d)
    muda = (antes[0] != depois[0]) | (antes[1] != depois[1])
    return float(np.sum(w[muda]) / np.sum(w))


def _conferir_cadeia(pastas: Path, esperado_score: str, forcado: bool = False) -> None:
    rel = pastas / "reports" / "experimentos"
    hash_dobras = _json(pastas / "reports" / "dobras.json")["dados"]["hash_dobras"]
    h5 = _json(rel / "h5_peso.json")["metricas"]
    cal = _json(rel / "calibracao.json")
    prod = _json(rel / "h6_h7_produto.json")
    cortes = _json(rel / "cortes.json")
    for r in (cal, prod, cortes):
        assert all(c in r for c in CHAVES)
        assert r["metricas"]["status"] == "ok"
        assert r["dados"]["dobras"]["hash"] == hash_dobras

    for parte in ("ocasiao", "preferencia"):
        m = cal["metricas"][parte]
        assert m["vencedor"] == _json(rel / f"escada_{parte}.json")["metricas"]["vencedor"]
        assert m["veredito"] == superioridade(_ic(m["ganho_calibrada"]))
        assert m["calibrar"] == (m["veredito"] == "confirmada")
        for versao in ("antes", "depois"):
            assert {"o_e", "inclinacao", "intercepto_cil", "curva"} <= set(m[versao])
        assert len(m["sigmoide_por_dobra"]) == 5

    h6, h7 = prod["metricas"]["H6"], prod["metricas"]["H7"]
    if forcado:  # `padrao` foi trocado por um que sempre refuta: o veredito não vem do IC
        assert h6["veredito"] == "padrão refutado"
    else:
        assert h6["veredito"] == padrao(_ic(h6["ganho_do_direto"]))
    assert prod["metricas"]["score"] == esperado_score
    assert h6["pr_auc"]["produto"] is not None and h6["pr_auc"]["direto"] is not None
    degrau_direto = h6["degrau"]
    assert degrau_direto == cal["metricas"]["ocasiao"]["vencedor"]
    assert (pastas / "experimentos" / f"aninhada_direto_{degrau_direto}.joblib").exists()
    assert h7["sobre"] == esperado_score
    assert h7["veredito"] == superioridade(_ic(h7["ganho_corrigida"]))
    assert h7["calibrar_score"] == (h7["veredito"] == "confirmada")
    assert h7["inclinacao"]["b"] > 0
    assert h7["inclinacao"]["ic_inf"] <= h7["inclinacao"]["ic_sup"]

    # a sensibilidade confere com a conta feita à parte sobre as previsões salvas
    salvo = joblib.load(pastas / "experimentos" / "previsoes_score.joblib")
    assert salvo["hash_dobras"] == hash_dobras
    base = pd.read_parquet(pastas / "processed" / f"celulas_{EDICAO_TREINO}.parquet")
    w = base["peso_final"].to_numpy()
    sens = cortes["metricas"]["sensibilidade"]
    for qual in ("ocasiao", "preferencia"):
        for chave, d in (("menos_10", -10.0), ("mais_10", 10.0)):
            assert sens[qual][chave] == pytest.approx(
                _parcela_que_muda(salvo["ocasiao"], salvo["preferencia"], w, qual, d)
            )
    assert "corte_alta" in cortes["metricas"]["resumo"]
    assert cortes["metricas"]["score"] == esperado_score

    receita = carregar_receita(pastas / "reports" / "receita.json")
    assert receita.score == esperado_score
    assert receita.calibrar_score == h7["calibrar_score"]
    for parte in ("ocasiao", "preferencia"):
        rp = getattr(receita, parte)
        assert rp.degrau == cal["metricas"][parte]["vencedor"]
        assert rp.usar_peso == h5[parte]["usar_peso"]
        assert rp.calibrar == cal["metricas"][parte]["calibrar"]
        selecao = joblib.load(pastas / "experimentos" / f"selecao_{parte}.joblib")
        assert selecao["parametros"] == rp.parametros
        assert selecao["hash_dobras"] == hash_dobras
        assert receita.interacoes[parte] == (
            [tuple(par) for par in INTERACOES[parte][rp.parametros["interacoes"]]]
            if rp.degrau == "C"
            else []
        )
    assert receita.cortes == {
        "alta": CORTE_ALTA, "media": CORTE_MEDIA,
        "ocasiao": CORTE_OCASIAO, "preferencia": CORTE_PREFERENCIA,
    }  # fmt: skip
    for caminho in receita.experimentos:
        assert (pastas / caminho).exists(), caminho
    assert "reports/experimentos/h6_h7_produto.json" in receita.experimentos
    if esperado_score == "direto":
        assert receita.direto is not None
        assert receita.direto.degrau == receita.ocasiao.degrau
        assert receita.direto.usar_peso == receita.ocasiao.usar_peso
        assert receita.direto.calibrar == receita.ocasiao.calibrar
        assert "direto" in receita.referencia_a
        assert (pastas / "experimentos" / "selecao_direto.joblib").exists()
    else:
        assert receita.direto is None
        assert "direto" not in receita.referencia_a


def test_calibracao_produto_cortes_e_receita(pastas: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_escada("preferencia")
    etapas.etapa_escada("ocasiao")
    etapas.etapa_calibracao()
    etapas.etapa_produto(b=20)
    etapas.etapa_cortes()
    etapas.etapa_receita()
    h6 = _json(pastas / "reports" / "experimentos" / "h6_h7_produto.json")["metricas"]["H6"]
    esperado = "direto" if h6["veredito"] == "padrão refutado" else "produto"
    _conferir_cadeia(pastas, esperado)

    # o outro caminho: H6 forçado a promover o direto
    monkeypatch.setattr(etapas, "padrao", lambda ic: "padrão refutado")
    etapas.etapa_produto(b=20)
    etapas.etapa_cortes()
    etapas.etapa_receita()
    _conferir_cadeia(pastas, "direto", forcado=True)
    # com o score direto, a sensibilidade continua sobre as partes, e o JSON avisa
    cortes = _json(pastas / "reports" / "experimentos" / "cortes.json")
    assert "partes do produto" in cortes["metricas"]["aviso_direto"]


def test_produto_recusa_calibracao_que_falhou(pastas: Path) -> None:
    etapas.etapa_dobras()
    rel = pastas / "reports" / "experimentos"
    rel.mkdir(parents=True)
    (rel / "calibracao.json").write_text(
        json.dumps({"metricas": {"status": "falhou"}, "dados": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="calibracao.json"):
        etapas.etapa_produto(b=5)
    assert _json(rel / "h6_h7_produto.json")["metricas"]["status"] == "falhou"


def test_produto_recusa_calibracao_de_outro_vencedor(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_escada("preferencia")
    etapas.etapa_escada("ocasiao")
    etapas.etapa_calibracao()
    rel = pastas / "reports" / "experimentos"
    caminho = rel / "calibracao.json"
    cal = _json(caminho)
    cal["metricas"]["ocasiao"]["vencedor"] = "outro_degrau"
    caminho.write_text(json.dumps(cal), encoding="utf-8")
    with pytest.raises(ValueError, match="discordam do vencedor"):
        etapas.etapa_produto(b=5)
    assert _json(rel / "h6_h7_produto.json")["metricas"]["status"] == "falhou"


def test_receita_recusa_cortes_de_outro_score(pastas: Path) -> None:
    etapas.etapa_dobras()
    etapas.etapa_peso()
    etapas.etapa_escada("preferencia")
    etapas.etapa_escada("ocasiao")
    etapas.etapa_calibracao()
    etapas.etapa_produto(b=20)
    etapas.etapa_cortes()
    caminho = pastas / "reports" / "experimentos" / "cortes.json"
    cortes = _json(caminho)
    cortes["metricas"]["score"] = (
        "direto" if cortes["metricas"]["score"] == "produto" else "produto"
    )
    caminho.write_text(json.dumps(cortes), encoding="utf-8")
    with pytest.raises(ValueError, match="cortes.json é de outro score"):
        etapas.etapa_receita()
