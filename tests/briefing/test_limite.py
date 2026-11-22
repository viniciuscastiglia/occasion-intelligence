"""Limite global de chamadas ao provedor, com relógio injetado."""

import threading

from occasion.briefing.limite import JANELA_EM_SEGUNDOS, LIMITE_GLOBAL_POR_HORA, LimiteGlobal


class Relogio:
    def __init__(self) -> None:
        self.agora = 0.0

    def __call__(self) -> float:
        return self.agora


def test_valores_padrao() -> None:
    assert LIMITE_GLOBAL_POR_HORA == 60
    assert JANELA_EM_SEGUNDOS == 3600.0


def test_permite_n_e_bloqueia_a_seguinte() -> None:
    limite = LimiteGlobal(maximo=3, relogio=Relogio())
    assert [limite.reservar() for _ in range(4)] == [True, True, True, False]
    assert limite.usadas() == 3


def test_libera_depois_da_janela() -> None:
    relogio = Relogio()
    limite = LimiteGlobal(maximo=2, janela_em_segundos=100.0, relogio=relogio)
    assert limite.reservar()
    relogio.agora = 10.0
    assert limite.reservar()
    assert not limite.reservar()
    relogio.agora = 99.9
    assert not limite.reservar()
    relogio.agora = 100.0  # a primeira chamada saiu da janela
    assert limite.reservar()
    assert not limite.reservar()
    relogio.agora = 110.0  # a segunda também
    assert limite.reservar()
    assert limite.usadas() == 2


def test_bloqueio_nao_consome() -> None:
    relogio = Relogio()
    limite = LimiteGlobal(maximo=1, janela_em_segundos=10.0, relogio=relogio)
    assert limite.reservar()
    relogio.agora = 5.0
    assert not limite.reservar()
    relogio.agora = 10.0
    assert limite.reservar()


def test_e_seguro_entre_threads() -> None:
    limite = LimiteGlobal(maximo=50, relogio=Relogio())
    aprovadas: list[bool] = []
    trava = threading.Lock()

    def tentar() -> None:
        for _ in range(20):
            ok = limite.reservar()
            with trava:
                aprovadas.append(ok)

    threads = [threading.Thread(target=tentar) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(aprovadas) == 50
    assert limite.usadas() == 50
