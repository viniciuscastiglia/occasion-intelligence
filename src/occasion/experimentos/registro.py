"""Registro dos experimentos: JSON legível em `reports/experimentos/` e objetos em `data/`.

Natureza (seção 6.5 do design): confirmatórios são H1 a H10 e as comparações da escada; dobras,
avaliação por segmento, erros, explicabilidade, região fora e diagnóstico são exploratórios.
"""

import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np

from occasion.config import DADOS_EXPERIMENTOS, RAIZ, RELATORIOS, SEMENTE
from occasion.pipeline import commit_da_receita

Natureza = Literal["confirmatório", "exploratório"]


def _para_json(objeto: Any) -> Any:
    """Converte `Intervalo`, tipos do NumPy, tuplas e NaN em valores que o JSON aceita."""
    if is_dataclass(objeto) and not isinstance(objeto, type):
        return _para_json(asdict(objeto))
    if isinstance(objeto, dict):
        return {str(k): _para_json(v) for k, v in objeto.items()}
    if isinstance(objeto, list | tuple | set):
        return [_para_json(v) for v in objeto]
    if isinstance(objeto, np.ndarray):
        return _para_json(objeto.tolist())
    if isinstance(objeto, np.bool_):
        return bool(objeto)
    if isinstance(objeto, np.integer):
        return int(objeto)
    if isinstance(objeto, np.floating | float):
        valor = float(objeto)
        return valor if np.isfinite(valor) else None
    if isinstance(objeto, Path):
        return str(objeto)
    return objeto


def registrar(
    nome: str,
    natureza: Natureza,
    configuracao: dict[str, Any],
    metricas: dict[str, Any],
    dados: dict[str, Any],
    pasta: Path | None = None,
    semente: int = SEMENTE,
) -> Path:
    """Grava `<pasta>/<nome>.json` com os campos recebidos, o commit, a semente e as versões.

    `semente` é a realmente usada pelo experimento; sem `pasta`, vale `RELATORIOS/experimentos`
    (lido na hora da chamada).
    """
    pasta = RELATORIOS / "experimentos" if pasta is None else pasta
    registro = {
        "nome": nome,
        "natureza": natureza,
        "configuracao": configuracao,
        "metricas": metricas,
        "dados": dados,
        "commit": commit_da_receita(RAIZ),
        "semente": semente,
        "data_hora": datetime.now().astimezone().isoformat(timespec="seconds"),
        "versoes": {
            "python": sys.version.split()[0],
            "pandas": version("pandas"),
            "scikit-learn": version("scikit-learn"),
            "lightgbm": version("lightgbm"),
        },
    }
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.json"
    caminho.write_text(
        json.dumps(_para_json(registro), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return caminho


def salvar_resultado(nome: str, objeto: Any) -> Path:
    """Grava o objeto em `DADOS_EXPERIMENTOS/<nome>.joblib` (só arquivos do próprio pipeline)."""
    DADOS_EXPERIMENTOS.mkdir(parents=True, exist_ok=True)
    caminho = DADOS_EXPERIMENTOS / f"{nome}.joblib"
    joblib.dump(objeto, caminho)
    return caminho


def carregar_resultado(nome: str) -> Any:
    """Lê um objeto gravado por `salvar_resultado`."""
    return joblib.load(DADOS_EXPERIMENTOS / f"{nome}.joblib")
