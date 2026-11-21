from occasion import config


def test_faixas_cobrem_as_24_horas_sem_sobreposicao():
    horas = [h for faixa in config.FAIXAS_HORARIO.values() for h in faixa]
    assert sorted(horas) == list(range(24))


def test_faixa_da_hora():
    assert config.faixa_da_hora(0) == "madrugada"
    assert config.faixa_da_hora(20) == "noite"
    assert config.faixa_da_hora(11) == "almoco"


def test_codigo_uf_mora_na_config_e_o_contrato_o_reexporta():
    from typing import get_args

    from occasion.features import variaveis
    from occasion.inference import contract

    assert contract.CODIGO_UF is config.CODIGO_UF
    assert contract.UF is config.UF
    assert list(config.CODIGO_UF) == list(get_args(config.UF))
    assert variaveis.DOMINIOS["uf"] == list(config.CODIGO_UF.values())
