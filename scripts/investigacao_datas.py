"""Investigação: é possível casar a POF com clima e feriados?

Reconstrói a data de calendário dos registros e mede até onde ela chega em cada edição.
Grava `reports/investigacao_datas.json`. Rodar da raiz com `make investigacao-datas`
(precisa de `make dados-brutos` e `make bases`).

Não calcula nenhuma taxa do target contra data, clima ou feriado: só contagens de amostra,
para não contaminar hipóteses que venham a ser registradas. Nada de 2017-2018 além de
datas de nascimento e idade é lido.

Decisão registrada na seção 26 de `docs/registro-de-decisoes.md`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS_2008 = RAIZ / "data/raw/2008-2009"
BRUTOS_2017 = RAIZ / "data/raw/2017-2018"
PESSOA_DIA_2008 = RAIZ / "data/processed/pessoa_dia_2008-2009.parquet"
SAIDA = RAIZ / "reports/investigacao_datas.json"

# Meia largura (dias) da janela de coleta de um setor. Em 2008-2009 as datas dos domicílios
# de um mesmo setor se espalham por cerca de 75 a 90 dias (ver "amplitude_no_setor").
MEIA_JANELA_SETOR = 45

FERIADOS_NACIONAIS_2008_2009 = [
    "2008-05-22",  # Corpus Christi
    "2008-09-07",
    "2008-10-12",
    "2008-11-02",
    "2008-11-15",
    "2008-12-25",
    "2009-01-01",
    "2009-02-23",  # Carnaval
    "2009-02-24",  # Carnaval
    "2009-04-10",  # Sexta-feira Santa
    "2009-04-21",
    "2009-05-01",
]
DATAS_COMERCIAIS_2008_2009 = {
    "dia_dos_namorados": "2008-06-12",
    "dia_dos_pais": "2008-08-10",
    "dia_das_criancas": "2008-10-12",
    "vespera_de_natal": "2008-12-24",
    "reveillon": "2008-12-31",
    "dia_das_maes": "2009-05-10",
}


def _s(obj: object) -> pd.Series:
    """Coluna como Series, para o Pyright (mesmo padrão de `occasion.audit.auditoria`)."""
    return cast(pd.Series, obj)


def _frac(mascara: object, casas: int) -> float:
    return round(float(_s(mascara).mean()), casas)


def _ler(caminho: Path, campos: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Lê campos de largura fixa. `campos` usa posição inicial em base 1 e tamanho."""
    specs = [(ini - 1, ini - 1 + tam) for ini, tam in campos.values()]
    return pd.read_fwf(
        caminho, colspecs=specs, names=list(campos), header=None, dtype=str, encoding="latin1"
    )


def _nascimento(df: pd.DataFrame, ano_max: int) -> pd.DataFrame:
    """Mantém moradores com data de nascimento válida e acrescenta a coluna `nasc`."""
    for c in ["dia", "mes", "ano"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    ok = df["dia"].between(1, 31) & df["mes"].between(1, 12) & df["ano"].between(1890, ano_max)
    df = df.loc[ok].copy()
    partes = pd.DataFrame({"year": df["ano"], "month": df["mes"], "day": df["dia"]})
    df["nasc"] = pd.to_datetime(partes, errors="coerce")
    return df.dropna(subset=["nasc"])


def _somar_anos(datas: pd.Series, anos: pd.Series) -> pd.Series:
    dia = np.where((datas.dt.month == 2) & (datas.dt.day == 29), 28, datas.dt.day)
    partes = pd.DataFrame(
        {"year": datas.dt.year + anos, "month": datas.dt.month, "day": dia}, index=datas.index
    )
    return pd.to_datetime(partes)


def _janela_por_idade(df: pd.DataFrame) -> pd.DataFrame:
    """Datas de referência compatíveis com a idade em anos completos: [lo, hi]."""
    idade = pd.Series(pd.to_numeric(df["idade"], errors="coerce"), index=df.index)
    df = df.loc[idade.notna()].copy()
    idade = idade.loc[idade.notna()].astype(int)
    df["lo"] = _somar_anos(_s(df["nasc"]), idade)
    df["hi"] = _somar_anos(_s(df["nasc"]), idade + 1) - pd.Timedelta(days=1)
    return df


def _centro_do_setor(df: pd.DataFrame, inicio: str, fim: str) -> pd.Series:
    """Para cada setor, a data t que maximiza moradores compatíveis com t ± MEIA_JANELA_SETOR."""
    dias = pd.date_range(inicio, fim)
    t = dias.values.astype("datetime64[D]").astype(np.int64)
    centros: dict[str, pd.Timestamp] = {}
    for setor, g in df.groupby("setor"):
        lo = g["lo"].values.astype("datetime64[D]").astype(np.int64)[:, None]
        hi = g["hi"].values.astype("datetime64[D]").astype(np.int64)[:, None]
        comp = ((lo <= t + MEIA_JANELA_SETOR) & (hi >= t - MEIA_JANELA_SETOR)).sum(0)
        melhores = np.flatnonzero(comp == comp.max())
        centros[str(setor)] = cast(pd.Timestamp, dias[int(np.median(melhores))])
    return pd.Series(centros, name="centro")


def _quantis(s: pd.Series, qs: list[float]) -> dict[str, float]:
    valores = _s(s.quantile(qs)).tolist()
    return {f"q{int(q * 100)}": float(v) for q, v in zip(qs, valores, strict=True)}


def edicao_2008() -> tuple[dict[str, Any], pd.DataFrame]:
    """Data de início da caderneta (POF3) por domicílio: nascimento + IDADE_DIA."""
    chave = {"COD_UF": (3, 2), "NUM_SEQ": (5, 3), "NUM_DV": (8, 1), "NUM_DOM": (9, 2)}
    mor = _ler(
        BRUTOS_2008 / "T_MORADOR_S.txt",
        chave
        | {"dia": (52, 2), "mes": (54, 2), "ano": (56, 4), "idade": (60, 3), "IDADE_DIA": (69, 7)},
    )
    dom = _ler(BRUTOS_2008 / "T_DOMICILIO_S.txt", chave | {"PERIODO_REAL": (41, 4)})
    for d in (mor, dom):
        d["setor"] = (
            d["COD_UF"].str.strip().str.zfill(2)
            + d["NUM_SEQ"].str.strip().str.zfill(3)
            + d["NUM_DV"].str.strip()
        )
        d["num_dom"] = d["NUM_DOM"].astype(int)
    n_moradores = len(mor)
    mor = _nascimento(mor, ano_max=2009)
    idade_dia = _s(pd.to_numeric(mor["IDADE_DIA"], errors="coerce"))
    mor["data_ini"] = _s(mor["nasc"]) + pd.to_timedelta(idade_dia, unit="D")
    mor = mor.dropna(subset=["data_ini"])
    por_dom = cast(pd.DataFrame, mor.groupby(["setor", "num_dom"])["data_ini"].agg(["min", "max"]))
    divergencia = (_s(por_dom["max"]) - _s(por_dom["min"])).dt.days
    datas = _s(por_dom["min"]).rename("data_ini").reset_index()
    datas = datas.merge(dom[["setor", "num_dom", "PERIODO_REAL"]], on=["setor", "num_dom"])
    datas["PERIODO_REAL"] = pd.to_numeric(datas["PERIODO_REAL"])
    semana = _s(datas.groupby("PERIODO_REAL")["data_ini"].agg(lambda s: (s.max() - s.min()).days))
    amplitude_setor = _s(datas.groupby("setor")["data_ini"].agg(lambda s: (s.max() - s.min()).days))

    # Os dois dias de consumo caem na semana da caderneta? Se sim, o dia da semana fixa a data.
    pdia = pd.read_parquet(PESSOA_DIA_2008)
    x = pdia.merge(
        datas[["setor", "num_dom", "data_ini"]],
        left_on=["cod_upa", "num_dom"],
        right_on=["setor", "num_dom"],
        how="left",
    )
    casados = int(_s(x["data_ini"]).notna().sum())
    x = x.dropna(subset=["data_ini"]).copy()
    x["desloc"] = (x["dia_semana"] - (x["data_ini"].dt.day_of_week + 1)) % 7
    x["data"] = x["data_ini"] + pd.to_timedelta(_s(x["desloc"]), unit="D")
    par = x.pivot_table(index="id_pessoa", columns="quadro", values="desloc").dropna()
    diff = par[72] - par[71]

    # Mesmo método de 2017-2018 (só idade em anos), validado contra a data verdadeira.
    mor_v = _janela_por_idade(mor)
    centro = _centro_do_setor(mor_v, "2008-04-01", "2009-06-30")
    v = datas.merge(centro, left_on="setor", right_index=True)
    erro = (v["centro"] - v["data_ini"]).dt.days.abs()

    res: dict[str, Any] = {
        "moradores": n_moradores,
        "moradores_com_nascimento_valido": len(mor),
        "domicilios": len(por_dom),
        "domicilios_com_moradores_divergentes": int((divergencia > 0).sum()),
        "data_ini_minima": str(datas["data_ini"].min().date()),
        "data_ini_maxima": str(datas["data_ini"].max().date()),
        "amplitude_da_semana_PERIODO_REAL_dias": _quantis(semana, [0.5, 0.9, 1.0]),
        "amplitude_no_setor_dias": _quantis(amplitude_setor, [0.5, 0.9]),
        "dias_pessoa": len(pdia),
        "dias_pessoa_com_data": casados,
        "data_consumo_minima": str(_s(x["data"]).min().date()),
        "data_consumo_maxima": str(_s(x["data"]).max().date()),
        "pessoas_com_dois_dias": len(par),
        "fracao_quadro71_antes_do_72": _frac(diff > 0, 4),
        "fracao_dias_consecutivos": _frac(diff.abs() == 1, 4),
        "deslocamento_mais_comum": {
            "quadro_71": int(x.loc[x["quadro"] == 71, "desloc"].mode().iloc[0]),
            "quadro_72": int(x.loc[x["quadro"] == 72, "desloc"].mode().iloc[0]),
        },
        "dia_da_semana_do_inicio_da_caderneta": {
            str(k): int(n)
            for k, n in (datas["data_ini"].dt.day_of_week + 1).value_counts().sort_index().items()
        },
        "validacao_do_metodo_de_2017": {
            "descricao": "centro do setor estimado só com nascimento + idade em anos",
            "erro_absoluto_dias": _quantis(erro, [0.5, 0.75, 0.9]),
            "fracao_mesmo_trimestre": round(
                float((v["centro"].dt.to_period("Q") == v["data_ini"].dt.to_period("Q")).mean()),
                3,
            ),
            "fracao_mesmo_mes": round(
                float((v["centro"].dt.to_period("M") == v["data_ini"].dt.to_period("M")).mean()),
                3,
            ),
        },
    }
    return res, x


def amostra_2008(x: pd.DataFrame) -> dict[str, Any]:
    """Contagens de dias-pessoa e positivos em feriados e datas comerciais (sem taxas)."""
    grupos = {
        "feriado_nacional": x["data"].isin(pd.to_datetime(FERIADOS_NACIONAIS_2008_2009).tolist()),
        "datas_comerciais": x["data"].isin(
            pd.to_datetime(list(DATAS_COMERCIAIS_2008_2009.values())).tolist()
        ),
        "24_dez_a_1_jan": x["data"].between("2008-12-24", "2009-01-01"),
        "carnaval_sabado_a_terca": x["data"].between("2009-02-21", "2009-02-24"),
        "total": pd.Series(True, index=x.index),
    }
    return {
        nome: {
            "dias_pessoa": int(m.sum()),
            "datas": int(x.loc[m, "data"].nunique()),
            "pizza_refri": int(x.loc[m, "tem_pizza_refri"].sum()),
            "pizza_guarana": int(x.loc[m, "tem_pizza_guarana"].sum()),
        }
        for nome, m in grupos.items()
    }


def edicao_2017() -> dict[str, Any]:
    """Sem data no registro. Mede quanto nascimento + idade em anos (V0403) estreitam a data."""
    mor = _ler(
        BRUTOS_2017 / "MORADOR.txt",
        {
            "COD_UPA": (8, 9),
            "NUM_DOM": (17, 2),
            "dia": (25, 2),
            "mes": (27, 2),
            "ano": (29, 4),
            "idade": (33, 3),
        },
    )
    n = len(mor)
    mor = _janela_por_idade(_nascimento(mor, ano_max=2018))
    mor["setor"] = mor["COD_UPA"]
    fixas = {
        d: _frac((mor["lo"] <= pd.Timestamp(d)) & (pd.Timestamp(d) <= mor["hi"]), 4)
        for d in ["2017-07-01", "2018-01-15", "2018-07-01"]
    }
    dom = mor.groupby(["COD_UPA", "NUM_DOM"]).agg(lo=("lo", "max"), hi=("hi", "min"))
    largura = (dom["hi"] - dom["lo"]).dt.days + 1
    centro = _centro_do_setor(mor, "2017-05-01", "2018-09-30")
    return {
        "moradores": n,
        "moradores_com_nascimento_valido": len(mor),
        "fracao_compativel_com_data_fixa": fixas,
        "domicilios": len(dom),
        "fracao_domicilios_sem_data_comum": round(float((largura <= 0).mean()), 4),
        "largura_da_janela_por_domicilio_dias": _quantis(largura[largura > 0], [0.25, 0.5, 0.75]),
        "setores": len(centro),
        "centro_do_setor_por_trimestre": {
            str(k): int(v) for k, v in centro.dt.to_period("Q").value_counts().sort_index().items()
        },
    }


def main() -> None:
    res_2008, x = edicao_2008()
    saida = {
        "pergunta": "a POF pode ser casada com clima e feriados no nível do dia?",
        "2008-2009": res_2008 | {"amostra_feriados": amostra_2008(x)},
        "2017-2018": edicao_2017(),
        "observacao": "só contagens; nenhuma taxa do target contra data, clima ou feriado.",
    }
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gravado {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
