"""Limite global de chamadas ao provedor por processo, em janela deslizante.

O limite por sessão sozinho não protege a demonstração pública: abrir outra sessão zera a
contagem. Este contador vale para o processo inteiro (a tela o guarda em `st.cache_resource`)
e é seguro entre as threads das sessões.
"""

import threading
import time
from collections import deque
from collections.abc import Callable

LIMITE_GLOBAL_POR_HORA = 60
JANELA_EM_SEGUNDOS = 3600.0


class LimiteGlobal:
    """Conta as chamadas da última janela; `reservar` só deixa passar até `maximo`."""

    def __init__(
        self,
        maximo: int = LIMITE_GLOBAL_POR_HORA,
        janela_em_segundos: float = JANELA_EM_SEGUNDOS,
        relogio: Callable[[], float] = time.monotonic,
    ) -> None:
        self._maximo = maximo
        self._janela = janela_em_segundos
        self._relogio = relogio
        self._instantes: deque[float] = deque()
        self._trava = threading.Lock()

    def reservar(self) -> bool:
        """Registra uma chamada e devolve True se ainda cabe na janela; senão, False."""
        with self._trava:
            agora = self._relogio()
            while self._instantes and agora - self._instantes[0] >= self._janela:
                self._instantes.popleft()
            if len(self._instantes) >= self._maximo:
                return False
            self._instantes.append(agora)
            return True

    def usadas(self) -> int:
        """Chamadas registradas na janela atual."""
        with self._trava:
            agora = self._relogio()
            return sum(1 for t in self._instantes if agora - t < self._janela)
