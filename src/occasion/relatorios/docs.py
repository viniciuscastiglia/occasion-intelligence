"""Tabelas dos documentos (README e Data Card) geradas dos JSON de `reports/`.

Os números dos textos só vêm daqui: cada tabela fica entre marcadores
`<!-- BEGIN:nome -->` e `<!-- END:nome -->`, e um teste confere que os arquivos estão
atualizados e que a prosa não cita decimal nem porcentagem que não esteja num bloco.

Regra de "número" (`numeros_fora_dos_blocos`): casa-se `\\d+(\\.\\d+)?%` (porcentagem) ou
`\\d+\\.\\d+` (decimal, ponto como separador), por igualdade exata de texto, sem o sinal.
Inteiros ficam fora da regra. Ignoram-se trechos entre crases, blocos de código cercados e URLs
(inclusive o alvo de um link Markdown).
"""

import json
import math
import re
from pathlib import Path
from typing import Any

from occasion.config import RAIZ

ARQUIVOS_COM_BLOCOS = ("README.md", "docs/DATA_CARD.md")

VEREDITOS = {
    "confirmada": "supported",
    "refutada": "refuted",
    "sem evidência contra o padrão": "no evidence against the default",
    "não executada": "not run",
}

PARTES = {
    "ocasiao": "occasion",
    "preferencia": "preference",
    "preferencia_com_pizza": "preference (pizza only)",
    "preferencia_todas": "preference (all soda occasions)",
}

# Grandeza de cada hipótese, traduzida do campo `metrica` de `reports/hipoteses.json`.
GRANDEZA_HIPOTESE = {
    "H1": "log-loss gain of logistic (B) over rate table (A)",
    "H2": "log-loss gain of curves and interactions (C) over logistic (B)",
    "H3": "log-loss gain of LightGBM (D) over best logistic",
    "H4": "pizza coefficient, log-odds",
    "H5": "log-loss gain of unweighted over weighted training",
    "H6": "log-loss gain of direct model over product",
    "H7": "log-loss gain of corrected over raw product",
    "H8": "PR-AUC gain of recipe over rate-table product, 2017-2018",
    "H9": "log-loss gain of recent-only over combined training",
    "H10": "rank correlation, search interest vs. POF share",
}

DIAS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

# Dimensões de `segmentos_teste`: as três com positivos suficientes em todos os níveis na
# previsão do score em 2017-2018 (faixa de horário, dia, faixa etária e renda têm níveis sem
# positivos suficientes ou muitos níveis).
SEGMENTOS_ESCOLHIDOS = ("regiao", "local", "sexo")
ROTULO_SEGMENTO = {"regiao": "region", "local": "place", "sexo": "sex"}
ROTULO_VALOR = {
    "centro_oeste": "Center-West",
    "nordeste": "Northeast",
    "norte": "North",
    "sudeste": "Southeast",
    "sul": "South",
    "em_casa": "at home",
    "fora_de_casa": "away from home",
    "feminino": "female",
    "masculino": "male",
}

NOTA_CONTAGENS = (
    "Counts of the modelling bases (cells and soda occasions) after exclusions, unweighted. "
    "In 2017-2018 a few soda occasions are dropped for undetermined place, so the totals differ "
    "slightly from the parser reference table. Days with pizza and no soda are not listed: "
    "counting them would require reading the 2017-2018 person-day base outside the gated loader."
)


def traduzir_veredito(veredito: str) -> str:
    """Traduz o veredito do JSON; veredito fora da tabela levanta `KeyError`."""
    return VEREDITOS[veredito]


def _fmt(x: float | None) -> str:
    """4 algarismos significativos, sem notação científica."""
    if x is None:
        return "-"
    if x == 0:
        return "0"
    casas = max(0, 3 - math.floor(math.log10(abs(x))))
    return f"{x:.{casas}f}"


def _ic(est: float | None, inf: float | None, sup: float | None) -> str:
    if est is None:
        return "-"
    return f"{_fmt(est)} [{_fmt(inf)}, {_fmt(sup)}]"


def _int(n: int) -> str:
    return f"{n:,}"


def _tabela(cabecalho: list[str], linhas: list[list[str]]) -> str:
    out = ["| " + " | ".join(cabecalho) + " |", "|" + "|".join("---" for _ in cabecalho) + "|"]
    out += ["| " + " | ".join(linha) + " |" for linha in linhas]
    return "\n".join(out)


def _ler(relatorios: Path, nome: str) -> dict[str, Any]:
    return json.loads((relatorios / nome).read_text(encoding="utf-8"))


def _bloco_contagens(c: dict[str, Any]) -> str:
    a, b = c["2008-2009"], c["2017-2018"]
    rotulos = [
        ("Person-days", "dias_pessoa"),
        ("Cells (person-day x time band x place)", "celulas"),
        ("Days with pizza and soda in the same hour", "dias_pizza_refri"),
        ("Days with pizza and guaraná soda in the same hour", "dias_pizza_guarana"),
        ("Soda occasions", "ocasioes_refri"),
    ]
    t1 = _tabela(
        ["Count", "2008-2009", "2017-2018"],
        [[r, _int(a[k]), _int(b[k])] for r, k in rotulos],
    )
    t2 = _tabela(
        ["Person-days by weekday", "2008-2009", "2017-2018"],
        [
            [
                DIAS[i],
                _int(a["dias_por_dia_semana"][str(i + 1)]),
                _int(b["dias_por_dia_semana"][str(i + 1)]),
            ]
            for i in range(7)
        ],
    )
    return f"{t1}\n\n{t2}\n\n{NOTA_CONTAGENS}"


def _bloco_hipoteses(h: dict[str, Any]) -> str:
    linhas = []
    for x in h["hipoteses"]:
        linhas.append(
            [
                x["id"],
                PARTES[x["parte"]] if x["parte"] else "-",
                GRANDEZA_HIPOTESE[x["id"]],
                _ic(x["estimativa"], x["ic_inf"], x["ic_sup"]),
                traduzir_veredito(x["veredito"]),
            ]
        )
    return _tabela(["ID", "Part", "Quantity", "Estimate [95% CI]", "Verdict"], linhas)


def _bloco_teste_final(t: dict[str, Any]) -> str:
    m = t["metricas"]
    linhas = []
    h8 = m["H8"]
    linhas.append(["H8: score (guaraná soda with pizza)", h8, "product of step-A rate tables"])
    for chave in ("ocasiao", "preferencia_todas", "preferencia_com_pizza"):
        linhas.append([PARTES[chave], m["partes"][chave], "step-A rate table"])
    corpo = [
        [
            nome,
            _fmt(r["pr_auc"]["receita"]),
            _fmt(r["pr_auc"]["referencia"]),
            _ic(r["ganho"]["estimativa"], r["ganho"]["ic_inf"], r["ganho"]["ic_sup"]),
            traduzir_veredito(r["veredito"]),
            ref,
        ]
        for nome, r, ref in linhas
    ]
    regra = (
        "A part is 'supported' only if the 95% interval of the PR-AUC gain excludes zero in "
        "favor of the recipe; otherwise it is 'refuted'. Evaluated on 2017-2018 cells, weighted."
    )
    tabela = _tabela(
        ["Target", "PR-AUC recipe", "PR-AUC reference", "Gain [95% CI]", "Verdict", "Reference"],
        corpo,
    )
    return f"{tabela}\n\n{regra}"


def _bloco_metricas_2008(a: dict[str, Any]) -> str:
    m = a["metricas"]
    fontes = [
        ("Occasion (cells)", m["ocasiao"]["resumo"]),
        ("Preference (all soda occasions)", m["preferencia"]["todas"]["resumo"]),
        ("Preference (pizza only)", m["preferencia"]["com_pizza"]["resumo"]),
        ("Score (product)", m["score"]["resumo"]),
    ]
    linhas = [
        [
            nome,
            _int(r["n"]),
            _int(r["positivos"]),
            _fmt(r["prevalencia"]),
            _fmt(r["pr_auc"]),
            _fmt(r["roc_auc"]),
            _fmt(r["log_loss"]),
            _fmt(r["brier"]),
        ]
        for nome, r in fontes
    ]
    tabela = _tabela(
        ["Model", "Rows", "Positives", "Prevalence", "PR-AUC", "ROC-AUC", "Log loss", "Brier"],
        linhas,
    )
    return (
        f"{tabela}\n\nOut-of-fold predictions on 2008-2009; prevalence, metrics and "
        "the score are weighted by the survey weight, rows and positives are unweighted."
    )


def _bloco_segmentos_teste(t: dict[str, Any]) -> str:
    linhas = []
    for r in t["metricas"]["segmentos"]["score"]:
        if r["segmento"] not in SEGMENTOS_ESCOLHIDOS:
            continue
        if r["pr_auc"] is None:
            medidas = ["too few positives", "-"]
        else:
            medidas = [_fmt(r["pr_auc"]), _fmt(r["o_e"])]
        linhas.append(
            [
                ROTULO_SEGMENTO[r["segmento"]],
                ROTULO_VALOR[r["valor"]],
                _int(r["n"]),
                _int(r["positivos"]),
            ]
            + medidas
        )
    linhas.sort(key=lambda x: (SEGMENTOS_ESCOLHIDOS.index(_inverso(x[0])), x[1]))
    tabela = _tabela(
        ["Dimension", "Level", "Cells", "Positives", "PR-AUC", "Observed / expected"], linhas
    )
    return f"{tabela}\n\nScore of the recipe on 2017-2018, weighted; observed / expected is the ratio of mean outcome to mean score."


def _inverso(rotulo: str) -> str:
    return next(k for k, v in ROTULO_SEGMENTO.items() if v == rotulo)


def gerar_blocos(relatorios: Path) -> dict[str, str]:
    """Gera o texto Markdown de cada bloco a partir dos JSON de `relatorios`."""
    teste = _ler(relatorios, "teste_final.json")
    return {
        "contagens": _bloco_contagens(_ler(relatorios, "contagens.json")),
        "hipoteses": _bloco_hipoteses(_ler(relatorios, "hipoteses.json")),
        "teste_final": _bloco_teste_final(teste),
        "metricas_2008": _bloco_metricas_2008(_ler(relatorios, "avaliacao_2008.json")),
        "segmentos_teste": _bloco_segmentos_teste(teste),
    }


_MARCADOR = re.compile(r"<!-- BEGIN:(\w+) -->\n(.*?)\n?<!-- END:\1 -->", re.DOTALL)


def aplicar_blocos(texto: str, blocos: dict[str, str]) -> str:
    """Substitui o conteúdo entre os marcadores; marcador sem bloco levanta `ValueError`."""

    def trocar(m: re.Match[str]) -> str:
        nome = m.group(1)
        if nome not in blocos:
            raise ValueError(f"marcador sem bloco correspondente: {nome}")
        return f"<!-- BEGIN:{nome} -->\n{blocos[nome]}\n<!-- END:{nome} -->"

    return _MARCADOR.sub(trocar, texto)


_NUMERO = re.compile(r"\d+(?:\.\d+)?%|\d+\.\d+")
_CODIGO_CERCADO = re.compile(r"```.*?```", re.DOTALL)
_CRASES = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")


def numeros_fora_dos_blocos(texto: str) -> list[str]:
    """Decimais e porcentagens da prosa que não aparecem em nenhum bloco do mesmo texto."""
    nos_blocos = {n for m in _MARCADOR.finditer(texto) for n in _NUMERO.findall(m.group(2))}
    prosa = _MARCADOR.sub("", texto)
    for padrao in (_CODIGO_CERCADO, _CRASES, _LINK, _URL):
        prosa = padrao.sub(" ", prosa)
    return [n for n in _NUMERO.findall(prosa) if n not in nos_blocos]


def main() -> None:
    """Reescreve os blocos de README.md e docs/DATA_CARD.md."""
    blocos = gerar_blocos(RAIZ / "reports")
    for arquivo in ARQUIVOS_COM_BLOCOS:
        caminho = RAIZ / arquivo
        caminho.write_text(
            aplicar_blocos(caminho.read_text(encoding="utf-8"), blocos), encoding="utf-8"
        )
        print(f"atualizado: {arquivo}")


if __name__ == "__main__":
    main()
