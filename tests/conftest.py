"""Gancho que pula os testes marcados `dados` quando os microdados não foram baixados."""

import pytest

from occasion import config as projeto


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if projeto.DADOS_BRUTOS.exists():
        return
    pular = pytest.mark.skip(
        reason=f"microdados ausentes em {projeto.DADOS_BRUTOS}; rode o download para executar"
    )
    for item in items:
        if item.get_closest_marker("dados"):
            item.add_marker(pular)
