from occasion import config


def test_faixas_cobrem_as_24_horas_sem_sobreposicao():
    horas = [h for faixa in config.FAIXAS_HORARIO.values() for h in faixa]
    assert sorted(horas) == list(range(24))


def test_faixa_da_hora():
    assert config.faixa_da_hora(0) == "madrugada"
    assert config.faixa_da_hora(20) == "noite"
    assert config.faixa_da_hora(11) == "almoco"
