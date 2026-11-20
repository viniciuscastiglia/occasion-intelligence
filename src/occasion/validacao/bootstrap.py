"""Bootstrap de setores inteiros, sorteados dentro de cada UF, para os intervalos de 95%.

O setor (`cod_upa`) é a unidade de sorteio porque as pessoas do mesmo setor não são
independentes. A UF é o estrato: com tão poucos setores por estrato da POF (mediana de 5), o
bootstrap dentro do estrato subestimaria a variância; por UF são dezenas a centenas de setores.
O intervalo é o percentil de 2,5 e 97,5 das réplicas (método percentil).
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from occasion.config import SEMENTE

__all__ = [
    "B_PADRAO",
    "Intervalo",
    "intervalo_estatistica",
    "intervalo_ganho",
    "intervalo_por_unidades",
    "multiplicadores",
]

B_PADRAO = 1000
NIVEL = 0.95


@dataclass(frozen=True)
class Intervalo:
    """Estimativa pontual e intervalo percentil; `b` é o número de réplicas válidas usadas."""

    estimativa: float
    ic_inf: float
    ic_sup: float
    b: int
    nivel: float = NIVEL

    @property
    def largura(self) -> float:
        return self.ic_sup - self.ic_inf


def multiplicadores(
    cod_upa: np.ndarray, grupo: np.ndarray, b: int, semente: int
) -> Iterator[np.ndarray]:
    """Gera `b` vetores de inteiros, um valor por linha: quantas vezes o setor da linha saiu.

    Em cada réplica, cada grupo sorteia com reposição tantos setores quantos ele tem. Um setor
    que aparece em dois grupos levanta `ValueError`.
    """
    cod_upa = np.asarray(cod_upa)
    grupo = np.asarray(grupo)
    setores, setor_da_linha = np.unique(cod_upa, return_inverse=True)
    n_setores = len(setores)
    grupo_do_setor = np.empty(n_setores, dtype=grupo.dtype)
    grupo_do_setor[setor_da_linha] = grupo
    if not np.array_equal(grupo_do_setor[setor_da_linha], grupo):
        raise ValueError("um setor (cod_upa) aparece em mais de um grupo")

    # Setores reordenados por grupo: cada grupo ocupa uma faixa contínua [inicio, inicio + tam).
    ordem = np.argsort(grupo_do_setor, kind="stable")
    posicao = np.empty(n_setores, dtype=np.int64)
    posicao[ordem] = np.arange(n_setores)
    _, inicio_do_grupo, tamanho_do_grupo = np.unique(
        grupo_do_setor[ordem], return_index=True, return_counts=True
    )
    rotulo = np.repeat(np.arange(len(tamanho_do_grupo)), tamanho_do_grupo)
    inicio = inicio_do_grupo[rotulo]
    tamanho = tamanho_do_grupo[rotulo]
    posicao_da_linha = posicao[setor_da_linha]

    rng = np.random.default_rng(semente)
    for _ in range(b):
        sorteado = inicio + np.minimum(
            (rng.random(n_setores) * tamanho).astype(np.int64), tamanho - 1
        )
        vezes = np.bincount(sorteado, minlength=n_setores)
        yield vezes[posicao_da_linha]


def _intervalo(estimativa: float, replicas: list[float]) -> Intervalo:
    if not replicas:
        raise ValueError("nenhuma réplica válida do bootstrap")
    baixo, alto = np.percentile(replicas, [100 * (1 - NIVEL) / 2, 100 * (1 + NIVEL) / 2])
    return Intervalo(float(estimativa), float(baixo), float(alto), len(replicas))


def intervalo_estatistica(
    funcao: Callable[[np.ndarray], float | None],
    cod_upa: np.ndarray,
    grupo: np.ndarray,
    b: int = B_PADRAO,
    semente: int = SEMENTE,
) -> Intervalo:
    """Intervalo de `funcao(multiplicadores)`; réplicas em que ela devolve `None` são descartadas."""
    estimativa = funcao(np.ones(len(cod_upa)))
    if estimativa is None:
        raise ValueError("a estatística não está definida na amostra original")
    replicas = [
        v for m in multiplicadores(cod_upa, grupo, b, semente) if (v := funcao(m)) is not None
    ]
    return _intervalo(estimativa, replicas)


def intervalo_ganho(
    metrica: Callable[[np.ndarray, np.ndarray, np.ndarray], float | None],
    maior_e_melhor: bool,
    y: np.ndarray,
    p_candidato: np.ndarray,
    p_referencia: np.ndarray,
    w: np.ndarray,
    cod_upa: np.ndarray,
    grupo: np.ndarray,
    b: int = B_PADRAO,
    semente: int = SEMENTE,
) -> Intervalo:
    """Intervalo do ganho do candidato sobre a referência; positivo favorece o candidato.

    Bootstrap pareado: as duas previsões usam o mesmo `w · multiplicador` em cada réplica.
    """
    w = np.asarray(w, dtype=float)

    def ganho(m: np.ndarray) -> float | None:
        w_rep = w * m
        cand = metrica(y, p_candidato, w_rep)
        ref = metrica(y, p_referencia, w_rep)
        if cand is None or ref is None:
            return None
        return cand - ref if maior_e_melhor else ref - cand

    return intervalo_estatistica(ganho, cod_upa, grupo, b, semente)


def intervalo_por_unidades(
    funcao: Callable[[np.ndarray], float | None],
    n_unidades: int,
    b: int = B_PADRAO,
    semente: int = SEMENTE,
) -> Intervalo:
    """Bootstrap simples de unidades (as 27 UFs em H10): `funcao` recebe os índices sorteados."""
    estimativa = funcao(np.arange(n_unidades))
    if estimativa is None:
        raise ValueError("a estatística não está definida na amostra original")
    rng = np.random.default_rng(semente)
    replicas = [
        v for _ in range(b) if (v := funcao(rng.integers(0, n_unidades, n_unidades))) is not None
    ]
    return _intervalo(estimativa, replicas)
