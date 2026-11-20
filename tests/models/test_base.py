"""Testes das funções numéricas e da interface comum dos modelos de parte."""

import numpy as np

from occasion.models.base import EPS, ModeloParte, logit, sigmoide
from occasion.models.logistica import ModeloLogistico
from occasion.models.taxas import TabelaDeTaxas


def test_logit_e_sigmoide_sao_inversas() -> None:
    p = np.array([0.001, 0.2, 0.5, 0.97])
    np.testing.assert_allclose(sigmoide(logit(p)), p, atol=1e-12)


def test_logit_corta_em_eps_em_vez_de_dar_infinito() -> None:
    z = logit(np.array([0.0, 1.0]))
    assert np.isfinite(z).all()
    np.testing.assert_allclose(z[0], np.log(EPS / (1 - EPS)))


def test_sigmoide_e_estavel_para_valores_grandes() -> None:
    p = sigmoide(np.array([-800.0, 800.0]))
    assert p[0] == 0.0 and p[1] == 1.0


# O LightGBM entra na tarefa 8; a conformidade ao Protocol é checada pelo pyright.
def test_tabela_e_logistica_expoem_variaveis_pela_interface() -> None:
    def aceita(modelo: ModeloParte) -> list[str]:
        return modelo.variaveis

    assert aceita(TabelaDeTaxas(["local"])) == ["local"]
    assert "idade" in aceita(ModeloLogistico("ocasiao", C=1.0))
