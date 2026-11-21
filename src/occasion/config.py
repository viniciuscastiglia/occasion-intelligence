"""Constantes do projeto: caminhos, edições da POF, faixas de horário e semente."""

from pathlib import Path
from typing import Literal

RAIZ: Path = Path(__file__).resolve().parents[2]
DADOS_BRUTOS: Path = RAIZ / "data" / "raw"
DADOS_INTERIM: Path = RAIZ / "data" / "interim"
DADOS_PROCESSADOS: Path = RAIZ / "data" / "processed"
DADOS_EXPERIMENTOS: Path = RAIZ / "data" / "experimentos"
MODELOS: Path = RAIZ / "models"
RELATORIOS: Path = RAIZ / "reports"

EDICOES: tuple[str, str] = ("2008-2009", "2017-2018")
EDICAO_TREINO: str = "2008-2009"
EDICAO_TESTE: str = "2017-2018"

SEMENTE: int = 20261112

# Tag git que marca a receita congelada; só depois dela o alvo da edição de teste pode ser lido.
TAG_RECEITA: str = "receita-congelada"

# Faixas de horário (hora cheia, 0 a 23). Juntas cobrem as 24 horas sem sobreposição.
FAIXAS_HORARIO: dict[str, range] = {
    "madrugada": range(6),
    "manha": range(6, 11),
    "almoco": range(11, 15),
    "tarde": range(15, 18),
    "noite": range(18, 24),
}

LOCAIS: tuple[str, str] = ("em_casa", "fora_de_casa")

# Siglas das unidades da federação e seus códigos IBGE. Moram aqui (e não no contrato da
# aplicação) porque definem o domínio fechado da variável `uf` dos modelos (`DOMINIOS`), que faz
# parte da receita congelada; `occasion.inference.contract` os reexporta.
UF = Literal[
    "RO", "AC", "AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA",
    "MG", "ES", "RJ", "SP", "PR", "SC", "RS", "MS", "MT", "GO", "DF",
]  # fmt: skip
CODIGO_UF: dict[UF, int] = {
    "RO": 11, "AC": 12, "AM": 13, "RR": 14, "PA": 15, "AP": 16, "TO": 17,
    "MA": 21, "PI": 22, "CE": 23, "RN": 24, "PB": 25, "PE": 26, "AL": 27, "SE": 28, "BA": 29,
    "MG": 31, "ES": 32, "RJ": 33, "SP": 35,
    "PR": 41, "SC": 42, "RS": 43,
    "MS": 50, "MT": 51, "GO": 52, "DF": 53,
}  # fmt: skip

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
#
# Regra (decisão R13 do autor): "refrigerante" é definido pelo produto, com a mesma regra nas
# duas edições. `guarana` e `cola` são os refrigerantes desses sabores, genéricos ou de marca.
# `outros_refri` é todo refrigerante que não é cola nem guaraná, genérico ou de marca (Fanta,
# Crush, Sukita, Sprite, Minuano, Bidu, Jaó, Aloa, Xodó da Bahia, Tubaína, Paraguai, laranja,
# uva, limão, não especificado), mais água tônica e refrigerante de quinino (todas as versões)
# e a "bebida refrigerante tipo H2O" (só em 2017-2018). Ficam fora nas duas edições: mate (é
# chá), Q-Refresko, energéticos, isotônicos, águas, chocomilk, refrescos (`8200308`, `8500608`)
# e chás. Bidu (possivelmente cola) e Jaó, Aloa e Xodó (sabor não informado no cadastro)
# ficam em `outros_refri` porque o cadastro não dá o sabor.
# Pizza: a mini pizza semipronta (`6502403`, `6506003`) só existe no cadastro de 2008-2009 e
# fica fora, para manter as duas edições coerentes.
#
# 2017-2018, `outros_refri` (código -> nome do cadastro):
#   8200202 FANTA LARANJA TRADICIONAL; 8200208 SUKITA TRADICIONAL; 8200211 REFRIGERANTE DE
#   LARANJA TRADICIONAL; 8200407 SPRITE REFRIGERANTE TRADICIONAL; 8200411 REFRIGERANTE DE LIMAO
#   TRADICIONAL; 8200505 FANTA UVA TRADICIONAL; 8200507 REFRIGERANTE DE UVA TRADICIONAL;
#   8200902 MINUANO TRADICIONAL; 8201017 BEBIDA REFRIGERANTE, TIPO H2O; 8201202 BIDU
#   TRADICIONAL; 8201502 FANTA LARANJA LIGHT; 8201511 REFRIGERANTE DE LARANJA DIET; 8201802
#   TUBAINA TRADICIONAL; 8201902 TUBAINA LIGHT; 8203501 REFRIGERANTE NAO ESPECIFICADO; 8204902
#   AGUA TONICA TRADICIONAL; 8205402 PARAGUAI REFRIGERANTE TRADICIONAL; 8207501 REFRIGERANTE DE
#   LIMAO LIGHT; 8207601 REFRIGERANTE DE LIMAO DIET; 8207701 REFRIGERANTE DE UVA DIET / LIGHT;
#   8207705 FANTA UVA LIGHT; 8209002 AGUA TONICA LIGHT; 8209102 PARAGUAI REFRIGERANTE LIGHT;
#   8212902 AGUA TONICA DIET; 8212906 REFRIGERANTE DE QUININO DIETETICO.
#   Fora (limite): 8201104 MATE TRADICIONAL; 8208104 MATE LIGHT; 8212802 MATE DIET;
#   8202403 Q-REFRESKO; 8211003 Q-REFRESKO LIGHT; 8200308 REFRESCO DE GUARANA NATURAL.
# 2008-2009, `outros_refri` (código -> nome do cadastro):
#   8200202 FANTA LARANJA TRADICIONAL; 8200204 CRUSH TRADICIONAL; 8200208 SUKITA TRADICIONAL;
#   8200407 SPRIT REFRIGERANTE TRADICIONAL; 8200505 FANTA UVA TRADICIONAL; 8200902 MINUANO
#   TRADICIONAL; 8201202 BIDU TRADICIONAL; 8201302 JAO TRADICIONAL; 8201402 ALOA TRADICIONAL;
#   8201502 FANTA LARANJA LIGHT; 8201504 CRUSH LIGHT; 8201508 SUKITA LIGHT; 8201702 XODO DA
#   BAHIA TRADICIONAL; 8201802 TUBAINA TRADICIONAL; 8201902 TUBAINA LIGHT; 8203501 REFRIGERANTE
#   NAO ESPECIFICADO; 8204902 AGUA TONICA TRADICIONAL; 8205402 PARAGUAI REFRIGERANTE
#   TRADICIONAL; 8207601 REFRIGERANTE DE LIMAO DIET; 8207705 FANTA UVA LIGHT; 8207902 MINUANO
#   LIGHT; 8208202 BIDU LIGHT; 8208302 JAO LIGHT; 8208402 ALOA LIGHT; 8208502 XODO DA BAHIA
#   LIGHT; 8209002 AGUA TONICA LIGHT; 8209102 PARAGUAI REFRIGERANTE LIGHT; 8212902 AGUA TONICA
#   DIET; 8212906 REFRIGERANTE DE QUININO DIETETICO.
#   Fora (limite): 8201104 MATE TRADICIONAL; 8208104 MATE LIGHT; 8202403 Q-REFRESKO;
#   8211003 Q-REFRESKO LIGHT; 8211103 Q-REFRESKO DIET.
#
# Pizza 2017-2018 e as listas de guaraná e cola: do CLAUDE.md / cadastro, sem mudança.
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
                8200202,
                8200208,
                8200211,
                8200407,
                8200411,
                8200505,
                8200507,
                8200902,
                8201017,
                8201202,
                8201502,
                8201511,
                8201802,
                8201902,
                8203501,
                8204902,
                8205402,
                8207501,
                8207601,
                8207701,
                8207705,
                8209002,
                8209102,
                8212902,
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
        "outros_refri": frozenset(
            {
                8200202,
                8200204,
                8200208,
                8200407,
                8200505,
                8200902,
                8201202,
                8201302,
                8201402,
                8201502,
                8201504,
                8201508,
                8201702,
                8201802,
                8201902,
                8203501,
                8204902,
                8205402,
                8207601,
                8207705,
                8207902,
                8208202,
                8208302,
                8208402,
                8208502,
                8209002,
                8209102,
                8212902,
                8212906,
            }
        ),
    },
}
