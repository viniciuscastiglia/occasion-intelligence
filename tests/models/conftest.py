"""Fixtures das bases sintéticas dos modelos de parte."""

import pandas as pd
import pytest
from tests.models.bases_sinteticas import base_sintetica


@pytest.fixture
def base_p1() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    return base_sintetica(3000, semente=0)


@pytest.fixture
def base_p2() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    return base_sintetica(3000, semente=10, com_pizza=True)
