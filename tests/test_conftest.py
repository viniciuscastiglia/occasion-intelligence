from pathlib import Path

import pytest

from occasion import config

CONFTEST = Path(__file__).with_name("conftest.py").read_text(encoding="utf-8")

TESTES = """
import pytest

@pytest.mark.dados
def test_precisa_de_dados():
    assert True

def test_sem_dados():
    assert True
"""


def test_marca_dados_e_pulada_sem_data_raw(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "DADOS_BRUTOS", tmp_path / "nao_existe")
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(TESTES)
    resultado = pytester.runpytest_inprocess("-rs")
    resultado.assert_outcomes(passed=1, skipped=1)
    resultado.stdout.fnmatch_lines(["*microdados ausentes*"])


def test_marca_dados_roda_com_data_raw(
    pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(config, "DADOS_BRUTOS", tmp_path)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(TESTES)
    resultado = pytester.runpytest_inprocess()
    resultado.assert_outcomes(passed=2)
