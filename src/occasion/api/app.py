"""API FastAPI fina sobre `prever`: `POST /predict` e `GET /health`. Uso local, não publicada."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from occasion.config import MODELOS
from occasion.inference.contract import Contexto, Resultado
from occasion.inference.predict import ArtefatoModelo, carregar_artefato, prever


def criar_app(artefato: ArtefatoModelo | None = None) -> FastAPI:
    """Monta a aplicação. Sem artefato, carrega um na subida, da pasta `OCCASION_MODELOS`."""

    @asynccontextmanager
    async def ciclo_de_vida(app: FastAPI) -> AsyncIterator[None]:
        if getattr(app.state, "artefato", None) is None:
            pasta = Path(os.environ.get("OCCASION_MODELOS", MODELOS))
            app.state.artefato = carregar_artefato(pasta)
        yield

    app = FastAPI(title="Occasion Intelligence", lifespan=ciclo_de_vida)
    app.state.artefato = artefato

    @app.post("/predict", response_model=Resultado)
    def predict(contexto: Contexto, request: Request) -> Resultado:
        return prever(contexto, request.app.state.artefato)

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        meta = request.app.state.artefato.metadados
        return {
            "status": "ok",
            "modelo": meta.modelo,
            "degrau": meta.degrau,
            "treinado_em": meta.treinado_em,
            "receita_commit": meta.receita_commit,
            "versao_contrato": meta.versao_contrato,
        }

    return app


app = criar_app()
