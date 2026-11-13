# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado do repositório

O Plano 1 foi executado na branch `plano-1`: ingestão, bases, auditoria, tabela de taxas, score, Decision Engine, inferência e tela Streamlit. Comandos reais do `Makefile` (rodar da raiz):

- `make ambiente`: instala as dependências (`uv sync`).
- `make dados-brutos`: baixa, confere os checksums e extrai os microdados em `data/raw` (não baixa de novo o que já existe).
- `make bases`: gera os parquets de `data/interim` e `data/processed` (consumo, ocasiões, pessoa x dia, células e ocasiões com refrigerante).
- `make auditoria`: grava `reports/auditoria.json` (disponibilidade, tamanho de amostra, validação adversarial).
- `make baseline`: ajusta a tabela de taxas em 2008-2009 e grava o artefato em `models/avaliado/artefato.joblib`.
- `make tudo`: `dados-brutos`, `bases`, `auditoria` e `baseline` em sequência (cerca de 34 s com os zips já baixados).
- `make app`: abre a tela (`streamlit run app/streamlit_app.py`).
- `make teste`: testes que não precisam dos microdados (`pytest -m "not dados"`).
- `make teste-dados`: todos os testes, inclusive os que leem `data/` e conferem os números de referência.
- `make qualidade`: `ruff check .` e `pyright`.

Formatar só `src/`, `tests/` e `app/`; nunca `ruff format .` na raiz (ver "Armadilhas").

O que existe:

- `Occasion_Intelligence_Pizza_Guarana.md`: o TAPI, enunciado do case.
- `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md`: o design aprovado pelo autor em 12/11/2026. Ele prevalece sobre os resumos deste arquivo.
- `docs/registro-de-decisoes.md`: como cada decisão foi tomada, com as alternativas descartadas e os motivos.
- `docs/superpowers/plans/2026-11-12-plano-1-dados-e-caminho-inteiro.md`: o Plano 1, executado na branch `plano-1`. Vai dos microdados até a tela funcionando com a tabela de taxas. Os planos 2 (modelos, teste final e Trends) e 3 (tela completa, briefing e documentação) são escritos agora que o Plano 1 rodou.

A stack está definida na seção 16 do design: Python 3.12 fixado pelo `uv`, pandas, pandera, scikit-learn, LightGBM, SHAP, pytest, Hypothesis, Pydantic, Streamlit e FastAPI.

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

- **Parte 1, ocasião:** chance de pizza com refrigerante no mesmo horário, dado o contexto. Carrega o "quando". A unidade é pessoa x dia x célula, em que a célula é uma faixa de horário combinada com o local (em casa ou fora). Horário e local só existem quando a pessoa come algo; com células, eles viram entradas legítimas.
- **Parte 2, preferência:** chance de o refrigerante ser guaraná, dado o contexto. É treinada em todas as refeições com refrigerante, não só nas de pizza, porque pizza com guaraná sozinha tem poucos casos. Carrega o "quem e onde". A variável "tem pizza" entra no modelo e é fixada em "sim" na inferência.

O score é o produto das duas partes, mostrado como posição de 0 a 100 entre os contextos da população de referência.

O Decision Engine escolhe a ativação pela combinação das duas partes, em duas camadas: objetivo (sustentar, converter, criar ocasião ou não priorizar) e canal. Promoção de pizza em app de delivery e influencers digitais são categorias de ativação, não o modelo. O briefing gerado por LLM a partir da decisão estruturada está no escopo, por decisão do autor do projeto, mesmo sendo opcional no TAPI. O provedor é um serviço com interface compatível com a da OpenAI, como Groq ou NVIDIA NIM.

O modelo avaliado é treinado em 2008-2009. Depois do teste final, a mesma receita é reajustada para a aplicação (modelo servido), conforme a seção 12 do design.

## Fonte de dados: POF/IBGE

O ground truth é o bloco de Consumo Alimentar Pessoal da Pesquisa de Orçamentos Familiares. Cada linha é um alimento consumido por uma pessoa de 10 anos ou mais em um dia. A base parte de pessoa x dia; as unidades de modelagem estão em "Direção escolhida". Treino na edição 2008-2009 e teste na 2017-2018.

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

### Códigos de alimento

A fonte de verdade é `src/occasion/config.py` (`CODIGOS`), com a regra R13 comentada lá: "refrigerante" é definido pelo produto, com a mesma regra nas duas edições, incluindo as marcas de cada sabor. Abaixo, o resumo.

#### 2017-2018

- **Pizza:** `8500903`, `8500914` a `8500917`, `8500924`, `8500927` a `8500930`, `8500932`, `8500934` a `8500939`, `8508017`.
- **Guaraná:** `8200301`, `8200302`, `8201601`, `8201602`, `8207401`, `8207402`.
- **Cola:** `8200101`, `8200102`, `8200701`, `8200702`, `8200801`.
- **Outros refrigerantes:** `8200202`, `8200208`, `8200211`, `8200407`, `8200411`, `8200505`, `8200507`, `8200902`, `8201017`, `8201202`, `8201502`, `8201511`, `8201802`, `8201902`, `8203501`, `8204902`, `8205402`, `8207501`, `8207601`, `8207701`, `8207705`, `8209002`, `8209102`, `8212902`, `8212906`. Inclui refrigerantes de marca (Fanta, Sukita, Sprite, Minuano, Bidu, Tubaína, Paraguai), água tônica, quinino e a bebida refrigerante tipo H2O.

#### 2008-2009

- **Pizza:** `8500903`, `8508003`, `8500914` a `8500917`, `8508014` a `8508017`. A mini pizza semipronta (`6502403`, `6506003`) fica fora, para manter as duas edições coerentes.
- **Guaraná e cola:** as mesmas listas de 2017-2018.
- **Outros refrigerantes:** `8200202`, `8200204`, `8200208`, `8200407`, `8200505`, `8200902`, `8201202`, `8201302`, `8201402`, `8201502`, `8201504`, `8201508`, `8201702`, `8201802`, `8201902`, `8203501`, `8204902`, `8205402`, `8207601`, `8207705`, `8207902`, `8208202`, `8208302`, `8208402`, `8208502`, `8209002`, `8209102`, `8212902`, `8212906`.

`8200308` e `8500608` são refresco de guaraná natural, não refrigerante. Mate (`8201104`, `8208104`, `8212802`) é chá e fica fora, assim como Q-Refresko, energéticos, isotônicos e águas.

### Diferenças da edição 2008-2009

O arquivo é `T_CONSUMO_S.txt` e o layout é outro. As posições, conferidas no dicionário da edição, estão em `src/occasion/ingest/layouts.py`. Tem horário, dia da semana como código numérico, UF, renda e peso, mas não tem o tipo de refeição. Os quadros são 71 e 72, e há menos códigos de pizza.

- **Moradores:** `T_MORADOR_S.txt` (idade em `IDADE_ANOS`, sexo em `V0405`).
- **Local:** `V9018` é a "fonte do alimento", e só os valores 1 (dentro) e 2 (fora do domicílio) valem como local. Não é o "local de refeição" de 2017-2018.
- **Dia da semana:** 1 é segunda e 7 é domingo. Conferido por sinal externo: o consumo de cerveja tem pico em 6 e 7.

### Números de referência

Contagens já conferidas nos microdados, úteis para validar o pipeline:

| Medida | 2017-2018 | 2008-2009 |
|---|---|---|
| Dias-pessoa | 85.018 | 66.903 |
| Dias com pizza | 1.681 | 1.149 |
| Dias com pizza e refrigerante no mesmo horário | 797 | 651 |
| Dias com pizza e guaraná no mesmo horário | 180 | 101 |
| Ocasiões com refrigerante | 16.122 | 19.182 |

A ocasião é dia + hora nas duas edições, e "refrigerante" é definido pelo produto (decisões R12 e R13, aprovadas pelo autor em 12/11/2026). A tabela antiga (783/176 e 635/101, com 15.863 ocasiões com refrigerante em 2017-2018) misturava a chave dia + hora + refeição (`V9017`, que só existe em 2017-2018) e listas de refrigerante sem os sabores registrados por marca. Com a ocasião dia + hora, as listas antigas reproduzem 635/101 (2008-2009) e 793/180 com 15.863 ocasiões com refrigerante (2017-2018), e foi isso que conferiu o parser; 783/176 só aparecem com a chave dia + hora + refeição. Na base da parte 2, 10 ocasiões com refrigerante de 2017-2018 saem por local indeterminado (16.112 linhas).

Antes da mudança de lista, em 2017-2018: 27% das ocasiões com refrigerante eram com guaraná e 56% com cola (ponderado). Quando há pizza, o guaraná caía para 23% e a cola subia para 64%. Esses percentuais foram calculados com a lista antiga e não são recalculados antes do teste final.

Esses percentuais são da edição de teste e já foram vistos. Eles não orientam o modelo sem serem refeitos em 2008-2009. Até a abertura do teste final, nenhuma outra estatística do target de 2017-2018 é calculada; as contagens da tabela continuam valendo como teste do parser.

### Armadilhas

- **Sabor, não marca:** a POF registra "refrigerante de guaraná", nunca Guaraná Antarctica. Toda conclusão é sobre o sabor.
- **Sem data de calendário:** o registro só tem dia da semana. Não dá para cruzar com clima, feriado ou Google Trends no nível do dia.
- **Sem canal:** não há delivery, só em casa ou fora.
- **Dias desbalanceados:** a amostra tem muito mais domingos (16.585) que sábados (3.862). Usar `PESO_FINAL` e não comparar contagens brutas entre dias.
- **Dias da semana diferentes entre as edições:** por desenho de coleta, 2008-2009 tem muito mais terças (14.699) e quintas (17.383) e poucos domingos (4.378). Isso domina a validação adversarial (AUC ≈ 0,70) em `reports/auditoria.json`.
- **`ruff format .` na raiz:** reformata os blocos de código dos `.md` em `docs/`. Formatar só `src/`, `tests/` e `app/`.
- **`.gitignore`:** a regra `models/` ignorava também `src/occasion/models/`. Agora é `/models/`.
- **Nomes no zip da documentação:** os arquivos têm acentos em codificação antiga, e o `unzip` do macOS falha. Extrair com `zipfile` do Python.
- **Base antiga:** a edição mais recente é de 2017-2018, anterior à expansão do delivery.
- **Método de coleta diferente:** 2008-2009 usou registro alimentar preenchido pelo morador; 2017-2018 usou recordatório de 24 horas. A queda de desempenho no teste mistura mudança de hábito com mudança de método.
- **Renda em reais não é comparável entre as edições:** usar a renda por pessoa em salários mínimos da data de referência de cada edição.

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
- **Teste:** a receita não muda depois do teste final, e o modelo servido só é reajustado depois de o resultado estar commitado.
- **Hipóteses:** cada hipótese da seção 7 do design foi commitada antes da etapa em que é testada.

### Quando um número não bate

Usar `superpowers:systematic-debugging`. Se uma contagem divergir da tabela de referência, os primeiros suspeitos são posição de campo, codificação latin1 e a lista de códigos de alimento.

### Outras ferramentas instaladas que entram no projeto

- `context7` (servidor MCP de documentação): antes de escrever código que usa uma biblioteca (pandas, scikit-learn, LightGBM, SHAP, Streamlit, FastAPI, Pydantic), consultar a documentação da versão fixada no projeto com `resolve-library-id` e `query-docs`, em vez de confiar na memória da API. Vale também para subagentes: a tarefa deve dizer quais bibliotecas consultar. As consultas vão para um serviço externo; mandar só nome de biblioteca e dúvida técnica, nunca dados ou detalhes do projeto.
- `pyright-lsp` (checagem de tipos em Python): código novo leva anotações de tipo nas funções públicas, e os diagnósticos do Pyright são resolvidos antes de dar a tarefa como pronta. O Pyright está em `~/.local/bin`.
- `claude-md-management`: ao fim de cada etapa do pipeline, rodar `/revise-claude-md` para registrar aqui o que a etapa ensinou (comandos reais, decisões, armadilhas). Usar a skill `claude-md-improver` para auditar este arquivo quando ele crescer.
- `dataviz`: carregar antes de escrever qualquer gráfico, na EDA ou na aplicação.
- Briefing por LLM: o provedor não é a Anthropic, então a skill `claude-api` não se aplica. Consultar pelo `context7` a documentação do provedor escolhido (Groq ou NVIDIA NIM) e da biblioteca cliente.
- `frontend-design` ou `impeccable`: só se a interface não for Streamlit.

## Em aberto

As decisões de stack, formato da aplicação e Decision Engine estão fechadas no design. Falta conferir nos dados:

Resolvido no Plano 1:

- Posições dos campos e códigos de alimento de 2008-2009 (`src/occasion/ingest/layouts.py` e `src/occasion/config.py`).
- Trimestre da coleta: 2008-2009 tem `PERIODO_REAL` (semana da pesquisa) no registro de domicílio; 2017-2018 não tem campo de período.
- Salário mínimo das datas de referência: R$ 415 em 15/01/2009 e R$ 954 em 15/01/2018, conferidos.

Continua em aberto:

- Datas de coleta de cada edição para as janelas do Google Trends (a data de referência já está resolvida; falta a janela de coleta).

Fora do escopo, citados no README como próximos passos: a "mesa" (moradores do mesmo domicílio comendo pizza no mesmo horário), o coletor de cardápios de delivery e a POF 2024-2025.
