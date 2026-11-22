"""Compara provedores de LLM (Groq e NVIDIA NIM) no briefing automático.

Uso: `set -a; . ./.env; set +a; uv run python scripts/comparar_provedores.py`.
Lê `GROQ_API_KEY` e `NVIDIA_API_KEY` do ambiente, roda 10 contextos fixos em cada par
provedor x modelo (uma chamada por vez) e grava `reports/briefing_provedores.json`.
Nenhuma chave vai para o JSON, e nenhum microdado entra: só `Contexto` montado à mão.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from datetime import date
from pathlib import Path
from typing import Any

from occasion.briefing.cliente import ClienteHTTP, ConfiguracaoLLM, FalhaDoLLM
from occasion.briefing.conferencia import conferir
from occasion.briefing.prompt import entrada_do_briefing, montar_mensagens
from occasion.inference.contract import Contexto, Resultado
from occasion.inference.predict import carregar_artefato, prever

RAIZ = Path(__file__).resolve().parent.parent
SAIDA = RAIZ / "reports" / "briefing_provedores.json"
PAUSA_ENTRE_CHAMADAS = 3.0  # segundos; folga sob os limites por minuto dos planos gratuitos
TEMPO_LIMITE = 30.0

# (provedor, variável da chave, base_url, modelo)
PARES: tuple[tuple[str, str, str, str], ...] = (
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "qwen/qwen3.8-27b"),
    ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1", "openai/gpt-oss-120b"),
    (
        "nvidia",
        "NVIDIA_API_KEY",
        "https://integrate.api.nvidia.com/v1",
        "minimaxai/minimax-m3.1-flash",
    ),
    ("nvidia", "NVIDIA_API_KEY", "https://integrate.api.nvidia.com/v1", "poolside/laguna-xs-2.1"),
)

_CAMPOS = [
    "uf",
    "situacao",
    "dia_semana",
    "faixa_horario",
    "local",
    "idade",
    "sexo",
    "faixa_renda",
    "numero_moradores",
]
_VALORES: tuple[tuple[Any, ...], ...] = (
    ("RO", "urbano", "sabado", "noite", "em_casa", 45, "feminino", "ate_meio_sm", 4),
    ("RO", "rural", "sabado", "noite", "em_casa", 19, "masculino", "ate_meio_sm", 4),
    ("SP", "urbano", "sabado", "noite", "em_casa", 28, "feminino", "de_2_a_5_sm", 3),
    ("SC", "urbano", "terca", "noite", "em_casa", 55, "feminino", "de_2_a_5_sm", 2),
    ("RS", "urbano", "domingo", "almoco", "fora_de_casa", 45, "masculino", "mais_de_5_sm", 2),
    ("BA", "rural", "terca", "tarde", "em_casa", 60, "feminino", "ate_meio_sm", 5),
    ("PE", "urbano", "sexta", "tarde", "fora_de_casa", 40, "masculino", "meio_a_1_sm", 6),
    ("RJ", "urbano", "quinta", "manha", "em_casa", 35, "feminino", "meio_a_1_sm", 3),
    ("MG", "urbano", "segunda", "madrugada", "fora_de_casa", 50, "masculino", "de_1_a_2_sm", 1),
    ("CE", "rural", "quarta", "almoco", "em_casa", 70, "feminino", "ate_meio_sm", 2),
)
OBJETIVOS = ("Sustentar", "Converter", "Criar ocasião", "Não priorizar")


def contextos_fixos() -> list[Contexto]:
    """Os 10 contextos da comparação, montados à mão."""
    return [Contexto(**dict(zip(_CAMPOS, v, strict=True))) for v in _VALORES]


def objetivo_de(resultado: Resultado) -> str:
    """Objetivo do Decision Engine: o prefixo da categoria de recomendação."""
    for objetivo in OBJETIVOS:
        if resultado.recommendation_category.startswith(objetivo):
            return objetivo
    raise ValueError(f"categoria sem objetivo conhecido: {resultado.recommendation_category}")


def rodar_par(
    base_url: str, modelo: str, chave: str, casos: list[tuple[Contexto, Resultado]]
) -> dict[str, Any]:
    """Uma chamada por contexto, medindo a latência; falhas são registradas por motivo."""
    cliente = ClienteHTTP(ConfiguracaoLLM(base_url, modelo, chave), tempo_limite=TEMPO_LIMITE)
    chamadas: list[dict[str, Any]] = []
    for indice, (contexto, resultado) in enumerate(casos):
        entrada = entrada_do_briefing(contexto, resultado)
        inicio = time.perf_counter()
        item: dict[str, Any] = {"contexto": indice}
        try:
            texto = cliente.completar(montar_mensagens(contexto, resultado))
            item["latencia_s"] = round(time.perf_counter() - inicio, 3)
            conf = conferir(texto, entrada)
            item.update(texto=texto, aprovado=conf.aprovado, motivos=list(conf.motivos))
        except FalhaDoLLM as falha:
            item["latencia_s"] = round(time.perf_counter() - inicio, 3)
            item["falha"] = falha.motivo
        chamadas.append(item)
        print(f"    {indice}: {item['latencia_s']} s {item.get('falha', '')}", flush=True)
        time.sleep(PAUSA_ENTRE_CHAMADAS)
    latencias = [c["latencia_s"] for c in chamadas if "falha" not in c]
    falhas: dict[str, int] = {}
    reprovacoes: dict[str, int] = {}
    for c in chamadas:
        if "falha" in c:
            falhas[c["falha"]] = falhas.get(c["falha"], 0) + 1
        for m in c.get("motivos", []):
            reprovacoes[m] = reprovacoes.get(m, 0) + 1
    return {
        "modelo": modelo,
        "latencia_mediana_s": round(statistics.median(latencias), 3) if latencias else None,
        "latencia_maxima_s": max(latencias) if latencias else None,
        "aprovados": sum(1 for c in chamadas if c.get("aprovado")),
        "total": len(chamadas),
        "falhas_por_motivo": falhas,
        "reprovacoes_por_motivo": reprovacoes,
        "chamadas": chamadas,
    }


def main() -> None:
    artefato = carregar_artefato()
    contextos = contextos_fixos()
    casos = [(c, prever(c, artefato)) for c in contextos]
    objetivos = [objetivo_de(r) for _, r in casos]
    faltam = set(OBJETIVOS) - set(objetivos)
    if faltam:
        raise SystemExit(f"Objetivos ausentes nos contextos: {sorted(faltam)}")
    provedores: list[dict[str, Any]] = []
    for nome, var, base_url, modelo in PARES:
        chave = os.environ.get(var, "").strip()
        if not chave:
            print(f"{var} ausente; pulando {nome}/{modelo}")
            continue
        print(f"{nome}/{modelo} ...", flush=True)
        par = rodar_par(base_url, modelo, chave, casos)
        par.update(provedor=nome, base_url=base_url)
        provedores.append(par)
        print(
            f"  mediana {par['latencia_mediana_s']} s, máx {par['latencia_maxima_s']} s, "
            f"aprovados {par['aprovados']}/{par['total']}, falhas {par['falhas_por_motivo']}"
        )
    saida = {
        "data": date.today().isoformat(),
        "contextos": [
            {
                "indice": i,
                "contexto": c.model_dump(),
                "objetivo": o,
                "categoria": r.recommendation_category,
            }
            for i, ((c, r), o) in enumerate(zip(casos, objetivos, strict=True))
        ],
        "parametros": {"temperatura": 0.2, "max_tokens": 400, "tempo_limite_s": TEMPO_LIMITE},
        "pares": provedores,
    }
    SAIDA.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"gravado {SAIDA.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
