# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado do repositório

Ainda não há código, stack definida, nem comandos de build, lint ou teste. O único arquivo é o TAPI (`Occasion_Intelligence_Pizza_Guarana.md`), o enunciado do case. Quando o código existir, registrar aqui os comandos reais; não inventar comandos antes disso.

O projeto é individual, com entrega por volta de 25/11/2026 (duas semanas a partir de 11/11/2026). O escopo precisa caber nesse prazo: priorizar os entregáveis obrigatórios e reservar tempo para o vídeo, que é eliminatório.

Conversar e documentar em português.

## O que é o projeto

Case da Liga de IA com a Ambev: aumentar a frequência da associação Pizza + Guaraná Antarctica. O sistema estima em quais contextos a ocasião é mais provável (Occasion Score) e transforma isso em uma categoria de ativação justificada (Decision Engine).

O case é aberto. O TAPI é um guia, não uma ordem: a estratégia, as fontes, a unidade de análise e a stack são escolha nossa. Só os itens abaixo são fixos.

## O que é fixo (vem do TAPI)

Entregáveis obrigatórios:

1. Dataset, Data Card e pipeline de dados reprodutível.
2. EDA e feature engineering.
3. Modelo: baseline sem ML, ao menos um modelo de maior capacidade e o Occasion Score.
4. Avaliação: métricas, análise de erros, análise por segmentos e explicabilidade.
5. Decision Engine com regras documentadas.
6. Aplicação funcional: entrada de contexto, saída com score, fatores e recomendação.
7. Vídeo de até 5 minutos e README. O vídeo é eliminatório: precisa mostrar o código e a interface funcionando de verdade.

Regras que não podem ser quebradas:

- O target é uma medida real e observável. Não se cria target artificial, e o tipo de ativação nunca é o target.
- Proxies de interesse (Google Trends) nunca são tratados como consumo nem como target.
- O Decision Engine é feito de regras explícitas, não de um segundo modelo supervisionado.
- Não se alega impacto causal de nenhuma ativação, nem que uma ação vende mais que outra.
- A validação tem componente temporal (treino em período anterior, teste em posterior). Embaralhamento aleatório não é a única avaliação final.
- O instante de previsão é declarado, e toda feature precisa estar disponível até ele.
- Fontes com período ou granularidade incompatíveis não são combinadas como se fossem a mesma observação.
- Um LLM, se usado, só comunica fatores sustentados pela camada analítica; não calcula score nem inventa fatores.

Contrato de saída da aplicação: `occasion_score`, `classification`, `top_factors`, `recommendation_category`, `recommendation_reason`, `model_metadata`.

## Direção escolhida

A Ambev já tem dados de venda melhores que qualquer base pública, então o projeto não tenta prever "onde a ocasião já acontece". A pergunta central é: **em que contextos o guaraná vence a cola, e quais contextos de pizza se parecem com eles.**

O Occasion Score tem duas partes:

- **Parte 1, ocasião:** chance de pizza com refrigerante no mesmo horário, dado o contexto. Carrega o "quando".
- **Parte 2, preferência:** chance de o refrigerante ser guaraná, dado o contexto. É treinada em todas as refeições com refrigerante, não só nas de pizza, porque pizza com guaraná sozinha tem poucos casos. Carrega o "quem e onde".

O Decision Engine escolhe a ativação pela combinação das duas partes. Promoção de pizza em app de delivery e influencers digitais são categorias de ativação, não o modelo. O briefing gerado por LLM a partir da decisão estruturada está no escopo, por decisão do autor do projeto, mesmo sendo opcional no TAPI.

## Fonte de dados: POF/IBGE

O ground truth é o bloco de Consumo Alimentar Pessoal da Pesquisa de Orçamentos Familiares. Cada linha é um alimento consumido por uma pessoa de 10 anos ou mais em um dia. A unidade de análise é pessoa x dia; treino na edição 2008-2009 e teste na 2017-2018.

Microdados públicos:

- `https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/` (`Dados_20230713.zip`, `Documentacao_20230713.zip`)
- `https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2008_2009/Microdados/` (`Dados_20231009.zip`, `Documentacao_20231009.zip`)

A documentação traz `Dicionários de variáveis.xls` (layout de cada registro) e `Cadastro de Produtos do Consumo Alimentar.xls` (códigos dos alimentos).

### Registro `CONSUMO_ALIMENTAR.txt` (2017-2018)

Arquivo de largura fixa, codificação latin1. Posições iniciais em base 1:

| Campo | Posição | Tamanho | Conteúdo |
|---|---|---|---|
| `UF` | 1 | 2 | Código IBGE da UF |
| `ESTRATO_POF` | 3 | 4 | Estrato (capital, resto da RM, resto da UF, rural) |
| `TIPO_SITUACAO_REG` | 7 | 1 | 1 urbano, 2 rural |
| `COD_UPA` | 8 | 9 | Unidade primária de amostragem |
| `NUM_DOM` | 17 | 2 | Domicílio |
| `NUM_UC` | 19 | 1 | Unidade de consumo |
| `COD_INFORMANTE` | 20 | 2 | Pessoa |
| `QUADRO` | 22 | 2 | 72 ou 73: cada quadro é um dos dois dias registrados |
| `V9001` | 32 | 7 | Código do alimento |
| `V9015` | 39 | 3 | Horário, no formato `20H` |
| `V9017` | 44 | 1 | Refeição: 1 café, 2 almoço, 3 lanche, 4 jantar, 5 ceia, 6 outra |
| `V9018` | 45 | 1 | Local: 1 em casa, 2 a 7 fora de casa |
| `PESO_FINAL` | 504 | 15 | Peso amostral calibrado |
| `RENDA_TOTAL` | 519 | 10 | Renda mensal da unidade de consumo |
| `DIA_SEMANA` | 529 | 15 | Dia da semana por extenso |

- **Chave da pessoa:** `COD_UPA + NUM_DOM + NUM_UC + COD_INFORMANTE`. A mesma chave liga ao `MORADOR.txt` (idade em `V0403`, sexo em `V0404`).
- **Chave do dia:** chave da pessoa + `QUADRO`.
- **Ocasião:** chave do dia + `V9015`. "Pizza com refrigerante" significa os dois na mesma ocasião.

### Códigos de alimento (2017-2018)

- **Pizza:** `8500903`, `8500914` a `8500917`, `8500924`, `8500927` a `8500930`, `8500932`, `8500934` a `8500939`, `8508017`.
- **Guaraná:** `8200301`, `8200302`, `8201601`, `8201602`, `8207401`, `8207402`.
- **Cola:** `8200101`, `8200102`, `8200701`, `8200702`, `8200801`.
- **Outros refrigerantes:** `8200211`, `8200407`, `8200411`, `8200507`, `8201511`, `8203501`, `8205402`, `8207501`, `8207601`, `8207701`, `8209102`, `8212906`.

`8200308` e `8500608` são refresco de guaraná natural, não refrigerante.

### Diferenças da edição 2008-2009

O arquivo é `T_CONSUMO_S.txt` e o layout é outro (conferir no dicionário da edição). Tem horário, dia da semana como código numérico, UF, renda e peso, mas não tem o tipo de refeição, e o local só distingue dentro e fora do domicílio. Os quadros são 71 e 72, e há menos códigos de pizza.

### Números de referência

Contagens já conferidas nos microdados, úteis para validar o pipeline:

| Medida | 2017-2018 | 2008-2009 |
|---|---|---|
| Dias-pessoa | 85.018 | 66.903 |
| Dias com pizza | 1.681 | 1.149 |
| Dias com pizza e refrigerante no mesmo horário | 783 | 635 |
| Dias com pizza e guaraná no mesmo horário | 176 | 101 |

Em 2017-2018 há 15.863 ocasiões com refrigerante: 27% com guaraná e 56% com cola (ponderado). Quando há pizza, o guaraná cai para 23% e a cola sobe para 64%.

### Armadilhas

- **Sabor, não marca:** a POF registra "refrigerante de guaraná", nunca Guaraná Antarctica. Toda conclusão é sobre o sabor.
- **Sem data de calendário:** o registro só tem dia da semana. Não dá para cruzar com clima, feriado ou Google Trends no nível do dia.
- **Sem canal:** não há delivery, só em casa ou fora.
- **Dias desbalanceados:** a amostra tem muito mais domingos (16.585) que sábados (3.862). Usar `PESO_FINAL` e não comparar contagens brutas entre dias.
- **Nomes no zip da documentação:** os arquivos têm acentos em codificação antiga, e o `unzip` do macOS falha. Extrair com `zipfile` do Python.
- **Base antiga:** a edição mais recente é de 2017-2018, anterior à expansão do delivery.

## Fluxo de trabalho com o superpowers

O projeto é construído com orquestração de subagentes pelo plugin superpowers. Isso é preferência do autor: usar subagentes na execução é o padrão aqui, não a exceção.

### Sequência

1. `superpowers:brainstorming` fecha o que ainda está indeciso e gera o documento de design em `docs/superpowers/specs/`.
2. `superpowers:writing-plans` transforma o design aprovado em plano de implementação.
3. `superpowers:subagent-driven-development` executa o plano, uma tarefa por subagente.
4. `superpowers:verification-before-completion` antes de declarar qualquer tarefa pronta.
5. `superpowers:requesting-code-review` ao fim de cada etapa do pipeline.

A direção da seção "Direção escolhida" já foi decidida. O brainstorming não a reabre; ele só entra para partes novas ou ainda em aberto. Correções pequenas e claras dispensam brainstorming e plano.

### Como escrever tarefas para subagentes

Um subagente começa sem o contexto da conversa. Cada tarefa do plano precisa trazer:

- os arquivos de entrada e de saída, com caminho e esquema das colunas;
- os trechos deste arquivo de que ela depende (layout da POF, códigos de alimento, armadilhas);
- o critério de aceite verificável, de preferência um número da tabela "Números de referência";
- as regras do TAPI que a tarefa toca.

### O que pode rodar em paralelo

A espinha é sequencial: ingestão, base por pessoa x dia e por ocasião, features, modelos, score, integração. Só entram em paralelo (`superpowers:dispatching-parallel-agents`) tarefas que não editam os mesmos arquivos, depois que o contrato entre elas estiver fixado:

- parser da POF 2008-2009 e parser da POF 2017-2018;
- Decision Engine, que é regra pura sobre o contrato de saída, em paralelo aos modelos;
- esqueleto da API e da aplicação, contra um exemplo fixo do contrato de saída;
- camada de briefing por LLM;
- Data Card e README.

Usar worktrees (`superpowers:using-git-worktrees`) apenas nessas tarefas paralelas.

### Onde o TDD se aplica

- **Teste antes do código:** parsers, montagem da base, funções de feature, regras do Decision Engine, contrato da API e função de inferência. Os números de referência viram testes dos parsers.
- **Sem teste antes:** EDA e experimentos de modelo. Ali a verificação é script reprodutível, semente fixa e métricas gravadas em arquivo.
- **Proibido:** teste que afirma um limiar de métrica ajustado para passar.

### O que conta como verificado

- O pipeline rodou do início ao fim e reproduziu os números de referência.
- Toda métrica citada em texto, README ou aplicação foi lida do arquivo gerado pelo código, nunca de memória.
- A aplicação foi aberta e usada de verdade (skill `run`), porque o vídeo exige funcionamento real.

### Checklist de método em toda revisão

Além do código, o revisor confere:

- **Vazamento:** nenhuma feature usa informação posterior ao instante de previsão, e nada da edição 2017-2018 entra no ajuste de modelo ou de pré-processamento.
- **Pesos:** estimativas populacionais usam `PESO_FINAL`.
- **Target:** a definição de ocasião não mudou em relação ao Data Card.
- **Linguagem:** nenhum texto afirma causalidade, e "guaraná" nunca é apresentado como a marca Guaraná Antarctica.
- **LLM:** o briefing só usa fatores que vieram da camada analítica.

### Quando um número não bate

Usar `superpowers:systematic-debugging`. Se uma contagem divergir da tabela de referência, os primeiros suspeitos são posição de campo, codificação latin1 e a lista de códigos de alimento.

### Outras ferramentas instaladas que entram no projeto

- `context7` (servidor MCP de documentação): antes de escrever código que usa uma biblioteca (pandas, scikit-learn, LightGBM, SHAP, Streamlit, FastAPI, Pydantic), consultar a documentação da versão fixada no projeto com `resolve-library-id` e `query-docs`, em vez de confiar na memória da API. Vale também para subagentes: a tarefa deve dizer quais bibliotecas consultar. As consultas vão para um serviço externo; mandar só nome de biblioteca e dúvida técnica, nunca dados ou detalhes do projeto.
- `pyright-lsp` (checagem de tipos em Python): código novo leva anotações de tipo nas funções públicas, e os diagnósticos do Pyright são resolvidos antes de dar a tarefa como pronta. O Pyright está em `~/.local/bin`.
- `claude-md-management`: ao fim de cada etapa do pipeline, rodar `/revise-claude-md` para registrar aqui o que a etapa ensinou (comandos reais, decisões, armadilhas). Usar a skill `claude-md-improver` para auditar este arquivo quando ele crescer.
- `dataviz`: carregar antes de escrever qualquer gráfico, na EDA ou na aplicação.
- `claude-api`: carregar antes de escrever o código do briefing por LLM.
- `frontend-design` ou `impeccable`: só se a interface não for Streamlit.

## Em aberto

- Stack, formato da aplicação e regras exatas do Decision Engine.
- Comparabilidade entre as duas edições da POF (o método de coleta pode ter mudado) e estabilidade da preferência guaraná x cola em 2008-2009.
- Uso da "mesa" (moradores do mesmo domicílio comendo pizza no mesmo horário) como unidade para recomendar formato de embalagem.
