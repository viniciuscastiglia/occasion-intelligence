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
    "2008-2009": {
        # T_CONSUMO_S.txt (conferido no dicionário da edição).
        # Atenção: em 2008-2009 o campo V9018 é "fonte do alimento" (1 dentro, 2 fora do
        # domicílio), e não "local de refeição" como em 2017-2018 (1 em casa, 2 a 7 fora).
        # A diferença vai para a auditoria e para o Data Card.
        "consumo": {
            "COD_UF": (3, 2),
            "NUM_SEQ": (5, 3),
            "NUM_DV": (8, 1),
            "NUM_DOM": (9, 2),
            "NUM_UC": (11, 1),
            "COD_INFORMANTE": (12, 2),
            "ESTRATO_POF": (14, 2),
            "PESO_FINAL": (31, 15),  # 8 decimais, com ponto no arquivo
            "NUM_QUADRO": (46, 2),
            "V9018": (48, 1),
            "V9015": (49, 2),
            "V9001": (64, 7),
            "RENDA_TOTAL": (115, 16),  # 2 decimais, com ponto no arquivo
            "DIA_SEMANA": (152, 1),  # 1 segunda a 7 domingo
        },
        # T_MORADOR_S.txt. O PESO_FINAL deste registro começa na posição 30 (14), não é usado.
        "morador": {
            "COD_UF": (3, 2),
            "NUM_SEQ": (5, 3),
            "NUM_DV": (8, 1),
            "NUM_DOM": (9, 2),
            "NUM_UC": (11, 1),
            "COD_INFORMANTE": (12, 2),
            "ESTRATO_POF": (14, 2),
            "IDADE_ANOS": (60, 3),
            "V0405": (76, 2),  # 1 masculino, 2 feminino
            "COD_UPA": (247, 6),
            "TIPO_SITUACAO_REG": (253, 1),  # 1 urbano, 2 rural
        },
    },
}
