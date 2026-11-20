# Google Trends: interesse de busca por estado

Interesse de busca, não consumo.

## O que foi extraído

- **Data da extração:** 20/11/2026.
- **Tópicos:** "Guaraná Antarctica (Bebida)" e "Coca-Cola (Refrigerante)". Eram os únicos tópicos com volume para cada marca. Os rótulos de categoria do Google diferem, mas os dois são a entidade principal da marca.
- **Filtros:** Brasil, Todas as categorias, Pesquisa na Web, comparação dos dois tópicos por sub-região (estado).
- **Arquivos:**
  - `trends_2008-2009.csv`: 20/05/2008 a 22/05/2009 (dias de consumo da POF 2008-2009, `reports/investigacao_datas.json`)
  - `trends_2017-2018.csv`: 01/06/2017 a 31/07/2018 (período de coleta da POF 2017-2018, IBGE)
  - `trends_recente.csv`: 20/11/2025 a 20/11/2026

## Por que a comparação com a POF (hipótese H10) não foi feita

A hipótese H10 (seção 7 do design) compararia, por UF, a participação do guaraná nas buscas com a participação do guaraná nas ocasiões com refrigerante da POF. A extração mostrou três problemas, e o autor decidiu cortar o bloco em 20/11/2026, antes de qualquer cálculo com a POF. A decisão está na seção 27 do `docs/registro-de-decisoes.md`.

1. **Resolução.** Na comparação por estado, o site dá a participação de cada tópico em porcentagem inteira. A do guaraná fica entre 1% e 11% em 2008-2009 e quase toda em 2% em 2017-2018, o que não permite ordenar os estados.
2. **Ruído de marca.** Os tópicos juntam buscas que não são sobre a bebida, como bonés e coleções de maquiagem licenciadas. Essas buscas continuam entre as consultas mais frequentes mesmo com o filtro de categoria "Alimentos e bebidas". Como o público delas tem geografia própria, o ruído pode mudar a ordem dos estados.
3. **Volume em 2008-2009.** Na comparação, 15 dos 27 estados vêm sem valor para o guaraná.

Os arquivos ficam no repositório como registro do que foi tentado. Nenhum número deles entra no score, na recomendação ou nos resultados do projeto.
