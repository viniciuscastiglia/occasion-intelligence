"""Rótulos legíveis de variáveis, valores, partes e regras, usados pela tela e pelo Decision Engine."""

from occasion.config import CODIGO_UF

ROTULOS_DIA: dict[str, str] = {
    "segunda": "Segunda", "terca": "Terça", "quarta": "Quarta", "quinta": "Quinta",
    "sexta": "Sexta", "sabado": "Sábado", "domingo": "Domingo",
}  # fmt: skip
ROTULOS_HORARIO: dict[str, str] = {
    "madrugada": "Madrugada", "manha": "Manhã", "almoco": "Almoço", "tarde": "Tarde",
    "noite": "Noite",
}  # fmt: skip
ROTULOS_LOCAL: dict[str, str] = {"em_casa": "Em casa", "fora_de_casa": "Fora de casa"}
ROTULOS_SITUACAO: dict[str, str] = {"urbano": "Urbano", "rural": "Rural"}
ROTULOS_SEXO: dict[str, str] = {"masculino": "Masculino", "feminino": "Feminino"}
ROTULOS_RENDA: dict[str, str] = {
    "ate_meio_sm": "Até 1/2 salário mínimo por pessoa",
    "meio_a_1_sm": "De 1/2 a 1 salário mínimo",
    "de_1_a_2_sm": "De 1 a 2 salários mínimos",
    "de_2_a_5_sm": "De 2 a 5 salários mínimos",
    "mais_de_5_sm": "Mais de 5 salários mínimos",
}
ROTULOS_REGIAO: dict[str, str] = {
    "norte": "Norte", "nordeste": "Nordeste", "sudeste": "Sudeste", "sul": "Sul",
    "centro_oeste": "Centro-Oeste",
}  # fmt: skip
ROTULOS_PARTE: dict[str, str] = {
    "ocasiao": "ocasião", "preferencia": "preferência", "direto": "direto",
}  # fmt: skip
ROTULOS_VARIAVEL: dict[str, str] = {
    "dia_semana": "dia da semana", "faixa_horario": "faixa de horário", "local": "local",
    "uf": "UF", "regiao": "região", "situacao": "situação", "idade": "idade", "sexo": "sexo",
    "renda_pc_sm": "renda", "n_moradores_uc": "moradores", "tem_pizza": "tem pizza",
}  # fmt: skip

# Condição em palavras de cada regra de canal do Decision Engine (o identificador fica no JSON).
ROTULOS_REGRA: dict[str, str] = {
    "canal-1": "ocasião e preferência abaixo dos cortes",
    "canal-2": "ocasião alta, preferência baixa e idade abaixo de 30 anos",
    "canal-3": "fora de casa",
    "canal-4": "em casa e área urbana",
    "canal-5": "em casa e área rural",
}

_ROTULOS_DO_VALOR: dict[str, dict[str, str]] = {
    "dia_semana": ROTULOS_DIA,
    "faixa_horario": {c: r.lower() for c, r in ROTULOS_HORARIO.items()},
    "local": {c: r.lower() for c, r in ROTULOS_LOCAL.items()},
    "situacao": {c: r.lower() for c, r in ROTULOS_SITUACAO.items()},
    "sexo": {c: r.lower() for c, r in ROTULOS_SEXO.items()},
    "regiao": ROTULOS_REGIAO,
    # O fator da UF carrega o código do IBGE; a tela e o texto mostram a sigla.
    "uf": {str(codigo): sigla for sigla, codigo in CODIGO_UF.items()},
}


def rotulo_da_variavel(variavel: str) -> str:
    """Nome legível da variável; a que não é conhecida volta com o sublinhado trocado por espaço."""
    return ROTULOS_VARIAVEL.get(variavel, variavel.replace("_", " "))


def rotulo_da_parte(parte: str) -> str:
    return ROTULOS_PARTE.get(parte, parte)


def rotulo_da_regra(regra: str) -> str:
    """Condição da regra de canal em palavras; identificador desconhecido volta sem sublinhado."""
    return ROTULOS_REGRA.get(regra, regra.replace("_", " "))


def valor_legivel(variavel: str, valor: str) -> str:
    """Valor do fator em texto legível: sem códigos com sublinhado."""
    if variavel == "renda_pc_sm":
        try:
            return f"{float(valor):.1f} salário mínimo por pessoa".replace(".", ",")
        except ValueError:
            pass
    rotulos = _ROTULOS_DO_VALOR.get(variavel, {})
    return rotulos.get(valor, valor.replace("_", " "))
