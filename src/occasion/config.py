"""Constantes do projeto: caminhos, edições da POF, faixas de horário e semente."""

from pathlib import Path

RAIZ: Path = Path(__file__).resolve().parents[2]
DADOS_BRUTOS: Path = RAIZ / "data" / "raw"
DADOS_INTERIM: Path = RAIZ / "data" / "interim"
DADOS_PROCESSADOS: Path = RAIZ / "data" / "processed"
MODELOS: Path = RAIZ / "models"
RELATORIOS: Path = RAIZ / "reports"

EDICOES: tuple[str, str] = ("2008-2009", "2017-2018")
EDICAO_TREINO: str = "2008-2009"
EDICAO_TESTE: str = "2017-2018"

SEMENTE: int = 20261112

# Faixas de horário (hora cheia, 0 a 23). Juntas cobrem as 24 horas sem sobreposição.
FAIXAS_HORARIO: dict[str, range] = {
    "madrugada": range(6),
    "manha": range(6, 11),
    "almoco": range(11, 15),
    "tarde": range(15, 18),
    "noite": range(18, 24),
}

LOCAIS: tuple[str, str] = ("em_casa", "fora_de_casa")

# Salário mínimo nominal (R$) vigente na data de referência de cada edição da POF
# (15/01/2009 e 15/01/2018). Conferido nas publicações do IBGE, que citam os valores:
#   - POF 2008-2009: R$ 415,00 em 15/01/2009 (Lei 11.709/2008, vigente de 01/03/2008 a
#     31/01/2009; o reajuste para R$ 465,00 só veio em fevereiro de 2009).
#     https://biblioteca.ibge.gov.br/visualizacao/livros/liv50063.pdf
#   - POF 2017-2018: R$ 954,00 em 15/01/2018 (Decreto 9.255/2017).
#     https://biblioteca.ibge.gov.br/visualizacao/livros/liv101844.pdf
SALARIO_MINIMO: dict[str, float] = {"2008-2009": 415.0, "2017-2018": 954.0}


def faixa_da_hora(hora: int) -> str:
    """Devolve o nome da faixa de horário a que a hora cheia (0 a 23) pertence."""
    for nome, faixa in FAIXAS_HORARIO.items():
        if hora in faixa:
            return nome
    raise ValueError(f"hora fora do intervalo de 0 a 23: {hora}")


def _faixa(inicio: int, fim: int) -> frozenset[int]:
    """Códigos de inicio a fim, inclusive."""
    return frozenset(range(inicio, fim + 1))


# Códigos de alimento por edição. `refri` é a união de guarana, cola e outros_refri.
# `8200308` e `8500608` (2017-2018) são refresco de guaraná natural e ficam fora.
#
# 2017-2018: lista do CLAUDE.md, fixa.
#
# 2008-2009: montada a partir do `Cadastro de Produtos do Consumo Alimentar.xls` da edição.
# A lista é a que reproduz as três contagens de referência (1.149 dias com pizza, 635 com
# pizza e refrigerante na mesma hora, 101 com pizza e guaraná na mesma hora).
#   - Pizza: pizza e pizza light (8500903, 8500914-17, 8508003, 8508014-17). Os dois
#     códigos em dúvida, `6502403` e `6506003` (mini pizza semipronta), ficaram FORA:
#     com eles os dias com pizza passam de 1.149 para 1.249.
#   - Outros refrigerantes: só os seis códigos do cadastro cuja descrição é refrigerante
#     (8200407, 8203501, 8205402, 8207601, 8209102, 8212906). Os sabores sem a palavra
#     "refrigerante" no nome (Fanta, Crush, Sukita, Bidu, Jaó, Tubaína etc.) ficaram FORA:
#     com eles os dias com pizza e refrigerante passam de 635 para 651.
CODIGOS: dict[str, dict[str, frozenset[int]]] = {
    "2017-2018": {
        "pizza": frozenset({8500903, 8500924, 8500932, 8508017})
        | _faixa(8500914, 8500917)
        | _faixa(8500927, 8500930)
        | _faixa(8500934, 8500939),
        "guarana": frozenset({8200301, 8200302, 8201601, 8201602, 8207401, 8207402}),
        "cola": frozenset({8200101, 8200102, 8200701, 8200702, 8200801}),
        "outros_refri": frozenset(
            {
                8200211,
                8200407,
                8200411,
                8200507,
                8201511,
                8203501,
                8205402,
                8207501,
                8207601,
                8207701,
                8209102,
                8212906,
            }
        ),
    },
    "2008-2009": {
        "pizza": frozenset({8500903, 8508003})
        | _faixa(8500914, 8500917)
        | _faixa(8508014, 8508017),
        "guarana": frozenset({8200301, 8200302, 8201601, 8201602, 8207401, 8207402}),
        "cola": frozenset({8200101, 8200102, 8200701, 8200702, 8200801}),
        "outros_refri": frozenset({8200407, 8203501, 8205402, 8207601, 8209102, 8212906}),
    },
}
