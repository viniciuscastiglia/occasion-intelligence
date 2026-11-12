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
