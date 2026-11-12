"""Posições dos campos dos registros da POF, como dados.

Cada campo é `(posição inicial em base 1, tamanho)`, como nos dicionários de variáveis do IBGE.
A chave externa é a edição; a interna, o registro (`consumo` ou `morador`).
"""

Campos = dict[str, tuple[int, int]]

LAYOUTS: dict[str, dict[str, Campos]] = {
    "2017-2018": {
        # CONSUMO_ALIMENTAR.txt (conferido no dicionário da edição).
        "consumo": {
            "UF": (1, 2),
            "ESTRATO_POF": (3, 4),
            "TIPO_SITUACAO_REG": (7, 1),
            "COD_UPA": (8, 9),
            "NUM_DOM": (17, 2),
            "NUM_UC": (19, 1),
            "COD_INFORMANTE": (20, 2),
            "QUADRO": (22, 2),
            "V9001": (32, 7),
            "V9015": (39, 3),
            "V9017": (44, 1),
            "V9018": (45, 1),
            "PESO_FINAL": (504, 15),
            "RENDA_TOTAL": (519, 10),
            "DIA_SEMANA": (529, 15),
        },
        # MORADOR.txt.
        "morador": {
            "UF": (1, 2),
            "ESTRATO_POF": (3, 4),
            "TIPO_SITUACAO_REG": (7, 1),
            "COD_UPA": (8, 9),
            "NUM_DOM": (17, 2),
            "NUM_UC": (19, 1),
            "COD_INFORMANTE": (20, 2),
            "V0403": (33, 3),
            "V0404": (36, 1),
        },
    },
}
