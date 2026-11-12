# Occasion Intelligence Pizza + Guaraná: documento de design

- **Data:** 12/11/2026
- **Entrega do case:** por volta de 25/11/2026
- **Situação:** aguardando aprovação do autor do projeto. Já passou por um revisor independente, e as respostas às perguntas dele estão incorporadas
- **Documento irmão:** `docs/registro-de-decisoes.md` conta como cada decisão foi tomada, quais alternativas foram consideradas e por que cada uma foi descartada. Este documento diz o que será construído.

Onde uma escolha foi feita por quem escreveu este documento, e não pelo autor do projeto, ela está marcada com **(escolha do redator)** e listada de novo na seção 19.

---

## 1. Intenção

O case pede um sistema que estime em quais contextos a ocasião Pizza + Guaraná é mais provável e transforme essa estimativa em uma recomendação de ativação justificada.

A Ambev já tem dados de venda melhores que qualquer base pública. Por isso o projeto não tenta prever onde a ocasião já acontece. A pergunta central é: **em que contextos o guaraná se sai melhor contra a cola, e quais contextos de pizza se parecem com eles.**

O que o projeto entrega é um método demonstrado em dado real e público, que a Ambev poderia rodar nos dados dela. O valor está no rigor: target observável, validação temporal, hipóteses registradas antes do teste e limites declarados.

### Quem usa e para quê

- **Usuário:** uma pessoa de marketing ou trade da Ambev planejando ativações.
- **O que informa:** um contexto (UF, área urbana ou rural, dia da semana, faixa de horário, local e perfil do público).
- **O que recebe:** o Occasion Score, os fatores que o explicam e uma categoria de ativação com justificativa.
- **Que decisão toma:** para qual contexto direcionar qual tipo de ativação.

---

## 2. Objetivos, não objetivos e o que seria um fracasso

### Objetivos

1. Cumprir os sete entregáveis obrigatórios do TAPI (seção 17).
2. Estimar o Occasion Score em duas partes, ocasião e preferência, com avaliação temporal honesta.
3. Registrar hipóteses estruturadas antes de abrir o teste e relatar o resultado de cada uma.
4. Entregar uma aplicação que funciona de verdade, pronta para o vídeo desde a primeira semana.
5. Ter cada etapa coberta por testes (seção 15).

### Fora do escopo

- **Trend Radar, Mapa de Oportunidade e Laboratório de Preço**, as extensões da seção 9 do TAPI.
- **"Mesa":** agrupar moradores do mesmo domicílio que comem pizza no mesmo horário para recomendar formato de embalagem. Vai para o README como próximo passo.
- **Coletor de cardápios de delivery.** Vai para o README como próximo passo.
- **Clusterização de contextos, MLflow e monitoramento contínuo de drift.**
- **Veiculação de campanha, peças gráficas e integrações com plataformas privadas.**

### O que seria um fracasso mesmo com tudo rodando

- Um vídeo que não mostra o código e a interface funcionando. É eliminatório.
- Qualquer número em texto, README ou tela que não tenha sido lido de um arquivo gerado pelo código.
- Alterar a receita do modelo depois de abrir o teste.
- Apresentar "guaraná" como se fosse a marca Guaraná Antarctica, ou afirmar que uma ativação vende mais que outra.
- Um briefing de LLM que cita um fator ou número que a camada analítica não produziu.

---

## 3. Restrições

### Regras fixas do TAPI

- O target é uma medida real e observável. O tipo de ativação nunca é o target.
- Proxies de interesse, como o Google Trends, nunca são tratados como consumo nem como target.
- O Decision Engine é feito de regras explícitas, não de um segundo modelo supervisionado.
- Não se alega impacto causal de nenhuma ativação.
- A validação tem componente temporal.
- O instante de previsão é declarado, e toda variável precisa estar disponível até ele.
- Fontes com período ou granularidade incompatíveis não são combinadas como se fossem a mesma observação.
- Um LLM só comunica fatores sustentados pela camada analítica.

### Prazo e equipe

O projeto é individual e tem cerca de duas semanas. A execução é feita por subagentes, conforme o fluxo descrito no `CLAUDE.md`.

### Limites da fonte, que vão para o Data Card

- **Sabor, não marca:** a POF registra "refrigerante de guaraná". Toda conclusão é sobre o sabor.
- **Sem data de calendário:** o registro só tem dia da semana.
- **Sem canal:** não há delivery, só em casa ou fora de casa.
- **Base anterior à expansão do delivery:** a edição mais recente é de 2017-2018.
- **Método de coleta diferente entre as edições:** 2008-2009 usou registro alimentar preenchido pelo morador; 2017-2018 usou recordatório de 24 horas por entrevista. O cadastro de alimentos cresceu de 1.503 para 1.832 itens. Fonte: Rodrigues e colegas, Revista de Saúde Pública, 2021.
- **Só dois pontos no tempo.**

---

## 4. Dados

### 4.1 Fonte do target

Bloco de Consumo Alimentar Pessoal da POF/IBGE, edições 2008-2009 e 2017-2018. Cada linha é um alimento consumido por uma pessoa de 10 anos ou mais em um dia. O layout dos arquivos, as chaves, os códigos de alimento e as armadilhas estão no `CLAUDE.md`, seção "Fonte de dados: POF/IBGE", e valem como parte deste design.

Só essas duas edições têm consumo individual. As edições de 1987-1988, 1995-1996 e 2002-2003 mediram a compra de alimentos pelo domicílio. A edição 2024-2025 ainda não tem microdados publicados.

### 4.2 Papel de cada edição

| Edição | Papel | O que pode ser feito com ela |
|---|---|---|
| 2008-2009 | Desenvolvimento | EDA, escolha de variáveis, de modelo, de hiperparâmetros, de calibração e dos cortes do Decision Engine |
| 2017-2018 | Teste | Antes do teste: conferir contagens do parser, harmonizar o esquema e rodar a validação adversarial, que só usam as variáveis. No teste: medir a receita congelada uma vez. Depois: diagnósticos declarados e o reajuste do modelo servido |

Nada que venha de 2017-2018 muda a receita antes do teste.

### 4.3 O que já foi visto da edição de teste

O teste já foi olhado antes deste design, e isso precisa ficar registrado:

- As contagens da tabela "Números de referência" do `CLAUDE.md`.
- A participação de guaraná (27%) e cola (56%) nas ocasiões com refrigerante, e a queda do guaraná para 23% quando há pizza.
- Uma leitura descritiva de que o guaraná é mais forte no Nordeste, na renda baixa e entre pessoas de 60 anos ou mais, e mais fraco no Sul, na renda alta e entre 18 e 29 anos; e de que dia e horário quase não mudam a preferência.

Consequências:

- Nenhuma dessas observações pode orientar o desenho do modelo sem ser refeita em 2008-2009.
- Daqui em diante, nenhuma estatística do target em 2017-2018 é calculada antes da abertura do teste. A única exceção são as contagens de referência, usadas como teste do parser.

### 4.4 Arquivos e etapas do pipeline

Os nomes de pasta e de arquivo são sugestão; quem implementa pode ajustar, desde que as etapas e os critérios de aceite se mantenham.

| Etapa | Entrada | Saída | Critério de aceite |
|---|---|---|---|
| Download | Endereços do IBGE no `CLAUDE.md` | `data/raw/<edição>/` com os zips e um arquivo de checksums | Checksums gravados; extração com `zipfile` do Python |
| Parser 2017-2018 | `CONSUMO_ALIMENTAR.txt`, `MORADOR.txt` | `data/interim/consumo_2017.parquet`, `morador_2017.parquet` | 85.018 dias-pessoa; 1.681 com pizza; 783 com pizza e refrigerante no mesmo horário; 176 com pizza e guaraná |
| Parser 2008-2009 | `T_CONSUMO_S.txt` e o registro de moradores da edição | `data/interim/consumo_2008.parquet`, `morador_2008.parquet` | 66.903 dias-pessoa; 1.149 com pizza; 635 com pizza e refrigerante; 101 com pizza e guaraná |
| Esquema comum | Os quatro arquivos acima | Mesmas colunas, tipos e códigos nas duas edições | Validação de esquema passa nas duas |
| Base da parte 1 | Esquema comum | `data/processed/celulas_<edição>.parquet`: uma linha por pessoa, dia e célula | Número de linhas igual a dias-pessoa vezes número de células; soma dos positivos compatível com as contagens de referência |
| Base da parte 2 | Esquema comum | `data/processed/ocasioes_refri_<edição>.parquet`: uma linha por ocasião com refrigerante | Em 2017-2018, 15.863 ocasiões |

Pontos de atenção para os parsers:

- **Layout de 2008-2009:** é diferente do de 2017-2018. As posições vêm do dicionário de variáveis da edição. Os quadros são 71 e 72.
- **Códigos de alimento de 2008-2009:** as listas de pizza, guaraná, cola e outros refrigerantes precisam ser montadas a partir do cadastro de produtos da edição. Elas só são aceitas quando reproduzem as três contagens de referência.
- **Dia da semana:** texto por extenso em 2017-2018 e código numérico em 2008-2009. O esquema comum usa um código único.

### 4.5 Auditoria dos dados, antes de qualquer modelo

A auditoria usa as variáveis das duas edições e o target só de 2008-2009. Ela produz um relatório gravado em arquivo com:

1. **Variáveis disponíveis nas duas edições,** com a regra de harmonização de cada uma.
2. **Ficha de vazamento:** para cada variável candidata, um argumento escrito de por que ela é conhecida no instante de previsão.
3. **Conta de tamanho de amostra,** pelos critérios de Riley e colegas: quantos parâmetros os positivos de 2008-2009 sustentam em cada parte e no recorte de pizza da parte 2. O resultado limita o número de interações do degrau C.
4. **Validação adversarial:** um classificador tenta distinguir linhas de 2008-2009 das de 2017-2018 usando só as variáveis de entrada. Relata-se a AUC e as variáveis que mais separam as edições. Não usa o target. É só relato: nenhuma variável é retirada ou transformada por causa desse resultado antes do teste.
5. **Trimestre de coleta:** verificar se algum registro informa o período da coleta. Se informar, serve só para checar sazonalidade na EDA.

---

## 5. Formulação do problema

### 5.1 Instante de previsão

O momento do planejamento da ativação, antes de a ocasião acontecer. Só entram variáveis que descrevem o contexto e o público e que são conhecidas de antemão. Nada do que foi observado sobre o consumo da pessoa naquele dia entra como informação na inferência.

A parte 2 é condicional por construção: ela responde "se houver pizza com refrigerante neste contexto, qual a chance de ser guaraná". A variável "tem pizza" faz parte dessa pergunta e é fixada em "sim" na inferência; ela não é uma informação sobre o que alguém comeu. Esse argumento vai para a ficha de vazamento.

### 5.2 Parte 1: ocasião

- **Unidade:** pessoa × dia × célula. Uma célula é uma faixa de horário combinada com um local (em casa ou fora de casa).
- **Target:** houve pizza e refrigerante no mesmo horário dentro da célula, sim ou não.
- **Por que células:** horário e local só existem quando a pessoa come algo. Com uma linha por pessoa e dia, usá-los como entrada seria vazamento. Com células, eles viram entradas legítimas, porque a célula existe mesmo quando nada foi consumido nela.
- **Faixas de horário:** definidas na EDA de 2008-2009 e congeladas antes de qualquer modelo. Ponto de partida **(escolha do redator)**: madrugada (0h a 5h), manhã (6h a 10h), almoço (11h a 14h), tarde (15h a 17h) e noite (18h a 23h).
- **Local da ocasião positiva** **(escolha do redator)**: o local registrado no item de pizza.
- **"No mesmo horário":** mesmo valor do campo de horário dentro do mesmo dia da mesma pessoa, como definido no `CLAUDE.md`.

### 5.3 Parte 2: preferência

- **Unidade:** ocasião com refrigerante, isto é, pessoa × dia × horário em que houve algum refrigerante.
- **Target:** a ocasião inclui refrigerante de guaraná, sim ou não.
- **População de treino:** todas as ocasiões com refrigerante, não só as de pizza, porque pizza com guaraná sozinha tem poucos casos.
- **Variável "tem pizza":** indica se a ocasião inclui pizza. No treino ela é observada; na inferência é sempre "sim". Ela mede diretamente se a preferência muda quando há pizza e permanece no modelo qualquer que seja o resultado de H4.

### 5.4 Variáveis candidatas

A lista final é fixada com os dados de 2008-2009. A única restrição que vem de 2017-2018 é estrutural: a variável precisa existir nas duas edições, o que se confere no dicionário de variáveis. Candidatas:

| Variável | Parte 1 | Parte 2 | Observação |
|---|---|---|---|
| Dia da semana | sim | sim | |
| Faixa de horário | sim, como célula | sim, da ocasião | |
| Local (em casa, fora) | sim, como célula | sim, da ocasião | Em 2008-2009 só há essa distinção |
| UF e grande região | sim | sim | |
| Área urbana ou rural | sim | sim | |
| Idade | sim | sim | Do registro de moradores |
| Sexo | sim | sim | Do registro de moradores |
| Renda por pessoa em salários mínimos | sim | sim | Ver abaixo |
| Número de moradores | sim | sim | Candidata **(escolha do redator)** |
| Tem pizza | não | sim | |

**Renda:** o valor em reais não é comparável entre as edições, por causa de dez anos de inflação. A variável é a renda mensal por pessoa da unidade de consumo, dividida pelo salário mínimo vigente na data de referência da edição **(escolha do redator)**. O salário mínimo é uma constante pública, conhecida no instante de previsão, então nenhum pré-processamento é ajustado nos dados de 2017-2018. A data de referência e o valor do salário mínimo de cada edição vêm da documentação do IBGE. O ganho real do salário mínimo no período é um limite declarado dessa comparação.

### 5.5 Peso amostral

`PESO_FINAL` entra sempre no cálculo de métricas e de estimativas populacionais. Se ele entra também no treino é uma hipótese registrada (H5).

---

## 6. Protocolo de avaliação

### 6.1 Validação dentro de 2008-2009

- **Validação cruzada aninhada.** Cinco dobras externas para comparar modelos e três internas para escolher hiperparâmetros.
- **Dobras que imitam o desenho amostral.** O setor (`COD_UPA`, ou o campo equivalente de 2008-2009) fica inteiro em uma dobra só, e os setores são distribuídos dentro de cada estrato. Assim a mesma pessoa, o mesmo domicílio e o mesmo setor nunca ficam dos dois lados.
- **Conferência das dobras:** cada dobra precisa ter positivos suficientes nas duas partes. Se não tiver, a semente é trocada antes de qualquer modelo, e a troca fica registrada.
- **Semente fixa** em tudo.

### 6.2 Métricas

Fixadas antes de ver qualquer resultado:

| Papel | Métrica | Por quê |
|---|---|---|
| Principal, para escolher modelo | Log loss ponderado | A saída é uma probabilidade que vira score; a métrica pune ordenar mal e errar o tamanho da probabilidade |
| Secundária | PR-AUC ponderada | Mede se os contextos mais prováveis ficam no topo quando o evento é raro |
| Secundária | ROC-AUC e Brier ponderados | Pedidas pelo TAPI e úteis para comparação |
| Calibração | Média, inclinação e curva | Os três níveis de Van Calster e colegas; o nível da curva é o que basta para decidir |
| No corte de classificação | Precision, recall, F1 e matriz de confusão | Pedidas pelo TAPI; calculadas no corte que separa a classe "alta" |
| Uso prático | Concentração de positivos no décimo superior do score | Traduz o desempenho para a pergunta de negócio |

**Custo dos erros,** como o TAPI pede: um falso positivo gera uma ativação pouco relevante; um falso negativo deixa passar uma ocasião relevante. O projeto não tem valores para esses custos, então não fixa um limiar ótimo. A métrica principal é uma que avalia a probabilidade inteira, e o corte da classe "alta" é relatado com precision e recall para o leitor julgar.

### 6.3 Critério para um modelo ganhar

- A diferença na métrica principal precisa ser maior que a incerteza.
- A incerteza é medida por reamostragem de setores inteiros, sobre as previsões feitas fora da dobra de treino, com intervalo de 95%.
- Em caso de empate, fica o mais simples.

### 6.4 Avaliações adicionais

- **Parte 2 no recorte de pizza:** além de todas as ocasiões com refrigerante, a parte 2 é avaliada só nas ocasiões com pizza.
- **Generalização por região** **(escolha do redator)**: uma rodada em que cada grande região fica inteira fora do treino. É análise secundária e não escolhe modelo.
- **Segmentos:** período (dia e faixa de horário), região, público (idade, sexo e renda) e local. O TAPI pede também canal; a POF não tem canal, e o local é o substituto declarado.
- **Análise de erros:** os contextos em que o modelo mais erra, para cima e para baixo.

### 6.5 Limite de experimentos

O espaço de busca é escrito no plano antes de rodar qualquer experimento, e todo experimento é registrado, inclusive os que falharam.

- **Comparações confirmatórias:** as três subidas da escada (B contra A, C contra B, D contra C) em cada parte, mais as hipóteses da seção 7. Só elas sustentam conclusões.
- **Busca de hiperparâmetros** **(escolha do redator)**: até 8 valores de regularização para a logística e até 30 configurações sorteadas para o LightGBM.
- **Tudo o mais é exploratório** e é rotulado assim no relatório.

### 6.6 Registro

Cada experimento grava um arquivo JSON com configuração, semente, commit, dados usados e métricas. Uma ferramenta de rastreio como o MLflow não é usada.

### 6.7 Teste final

1. A receita é congelada em um commit marcado com uma tag.
2. O teste roda uma vez em 2017-2018 e grava o resultado em arquivo.
3. O resultado é relatado como vier, incluindo a queda em relação à validação, por segmento.
4. A queda é apresentada como efeito conjunto do tempo e da mudança no método de coleta. O projeto não consegue separar os dois.

Rodar o mesmo script de novo é permitido e deve reproduzir os mesmos números. O que não é permitido é mudar a receita depois de ver o resultado.

### 6.8 Diagnóstico depois do teste

Com o resultado gravado, a mesma receita é treinada por validação cruzada dentro de 2017-2018. A diferença entre esse desempenho e o do teste temporal indica quanto se perde ao atravessar dez anos e uma troca de método. Esse diagnóstico não altera nenhum número relatado.

---

## 7. Hipóteses registradas

Cada hipótese tem seis campos. Elas são commitadas antes da etapa em que são testadas, e o relatório final diz o resultado de cada uma.

**Sobre hipóteses com padrão declarado.** Em H5, H6 e H9, a regra mantém uma opção padrão a menos que haja evidência contra ela. Com poucos positivos, o intervalo de incerteza pode ser largo, e não achar diferença não prova que as opções são equivalentes. Nesses casos o relatório escreve "sem evidência contra o padrão", e não "confirmada", e informa a largura do intervalo.

### H1. O contexto acrescenta informação além da contagem

- **Enunciado:** em cada parte, a regressão logística prevê melhor que a tabela de taxas.
- **Motivo:** a tabela só enxerga poucas variáveis por vez.
- **Teste:** degrau B contra degrau A, na validação aninhada de 2008-2009.
- **Métrica:** diferença de log loss ponderado.
- **Regra:** confirmada se o intervalo de 95% da diferença excluir zero a favor de B.
- **Se refutada:** a tabela de taxas vira o modelo daquela parte, e o relatório diz que o ML não acrescentou.

### H2. Curvas e interações escolhidas melhoram a logística

- **Enunciado:** a logística com curvas para idade e renda e com interações definidas na EDA prevê melhor que a logística simples.
- **Motivo:** efeitos como o da idade raramente são retos, e combinações como dia e horário são plausíveis.
- **Teste:** degrau C contra degrau B.
- **Métrica e regra:** as mesmas de H1.
- **Se refutada:** fica o degrau B.

### H3. Existem combinações que não foram previstas

- **Enunciado:** o LightGBM prevê melhor que o melhor modelo logístico.
- **Motivo:** árvores capturam interações sem que alguém as especifique.
- **Teste:** degrau D contra o vencedor entre B e C.
- **Métrica e regra:** as mesmas de H1.
- **Se refutada:** o modelo logístico é o escolhido. Com poucos positivos, esse resultado é plausível e é relatado como achado.

### H4. A pizza muda a preferência

- **Enunciado:** dado o mesmo contexto, a presença de pizza reduz a chance de o refrigerante ser guaraná.
- **Motivo:** indício visto em 2017-2018 (seção 4.3), que precisa ser refeito em 2008-2009.
- **Teste:** coeficiente da variável "tem pizza" no degrau B da parte 2, em 2008-2009.
- **Métrica:** o coeficiente, com intervalo de 95% por reamostragem de setores.
- **Regra:** confirmada se o intervalo ficar inteiro abaixo de zero. Se ficar inteiro acima de zero, o relatório diz que o efeito existe no sentido contrário. Se contiver zero, o relatório diz que os dados não mostram o efeito e informa a largura do intervalo.
- **Se refutada:** nada muda na receita. A variável permanece no modelo em todos os casos, para que o produto continue sendo a regra da cadeia.

### H5. O peso amostral no treino melhora a previsão

- **Enunciado:** treinar com `PESO_FINAL` dá previsões melhores para a população do que treinar sem peso.
- **Motivo:** a literatura sobre ML em pesquisas amostrais recomenda o peso, mas trata o ajuste de hiperparâmetros com peso como área ainda em aberto.
- **Teste:** o degrau B de cada parte treinado com e sem peso; a métrica é sempre ponderada. O resultado vale para todos os degraus daquela parte.
- **Regra:** usa-se o peso no treino, a menos que a versão sem peso seja melhor por mais que a incerteza.
- **Se refutada:** treino sem peso, com a métrica ponderada mantida.

### H6. A decomposição em duas partes compensa

- **Enunciado:** o produto das duas partes prevê pizza com guaraná melhor que um modelo treinado direto nesse evento.
- **Motivo:** o evento direto tem 101 dias positivos em 2008-2009; as duas partes têm mais casos cada.
- **Teste:** produto das partes contra um modelo direto da mesma família, na mesma validação, com o target "pizza e guaraná no mesmo horário na célula".
- **Métrica:** log loss ponderado e PR-AUC ponderada contra o evento observado.
- **Regra:** o produto é o padrão. O modelo direto só assume se for melhor que o produto por mais que a incerteza.
- **Se refutada:** o modelo direto assume o cálculo do score, e as duas partes ficam só como componentes para o Decision Engine.

### H7. O produto está calibrado contra o fato observado

- **Enunciado:** o produto das duas partes corresponde à frequência observada de pizza com guaraná.
- **Teste:** calibração do produto contra o evento observado, nas previsões fora da dobra de treino.
- **Métrica:** os três níveis de calibração.
- **Regra:** o produto recebe uma correção sigmoide se ela melhorar o log loss ponderado contra o evento observado, na validação aninhada. A inclinação de calibração é relatada com intervalo de 95%, mas não é o critério: com cerca de cem dias positivos, um intervalo que contém 1 indicaria falta de casos, não boa calibração.
- **Se a correção entrar:** ela passa a fazer parte da receita.

### H8. O ordenamento sobrevive ao tempo

- **Enunciado:** a receita congelada ordena os contextos de 2017-2018 melhor que a tabela de taxas.
- **Motivo:** é a afirmação central do projeto sobre generalização temporal.
- **Teste:** teste final, com o score (o produto das duas partes) contra o evento observado de pizza com guaraná na célula. A referência é o produto das duas tabelas de taxas do degrau A.
- **Métrica:** PR-AUC ponderada. Ela mede ordenamento e não depende do nível geral da frequência.
- **Regra:** confirmada se o intervalo de 95% da diferença excluir zero a favor da receita.
- **Relatados à parte, sem entrar na regra:** a mesma comparação em cada parte separada e a concentração no décimo superior.
- **Se refutada:** o relatório diz que o modelo não superou a contagem simples em 2017-2018. A aplicação continua funcionando, e a limitação vai para a tela e para o README.
- **Se a receita for a própria tabela nas duas partes:** a comparação perde o sentido. Nesse caso relata-se o desempenho da tabela contra uma previsão constante igual à frequência geral.

### H9. Juntar as duas edições não prejudica o modelo servido

- **Enunciado:** treinar com 2008-2009 somado não piora a previsão de 2017-2018 em relação a treinar só com 2017-2018.
- **Motivo:** o volume de casos é o gargalo do projeto, mas a relação entre contexto e consumo pode ter mudado.
- **Teste:** depois do teste final, validação cruzada dentro de 2017-2018 comparando os dois treinos.
- **Métrica:** log loss ponderado nas dobras de 2017-2018, nas duas partes e no recorte de pizza.
- **Regra:** a base inteira é adotada se não for pior que a recente por mais que a incerteza.
- **Se refutada:** o modelo servido é treinado só em 2017-2018.
- **Ordem:** o reajuste na base inteira nunca acontece sem este experimento. Se ele for cortado por prazo, vale o plano B da seção 12.

### H10. O interesse de busca acompanha a preferência por UF

- **Enunciado:** por UF, a participação do guaraná nas buscas (guaraná contra cola) tem correlação positiva com a participação do guaraná nas ocasiões com refrigerante da POF, no período de cada edição.
- **Motivo:** saber se esse proxy de interesse acompanha a medida real quando os dois cobrem o mesmo período e a mesma unidade geográfica.
- **Teste:** correlação de postos entre as 27 UFs, uma vez por edição.
- **Regra:** confirmada se o intervalo de 95%, por reamostragem de UFs, excluir zero nas duas edições.
- **Se refutada:** o relatório diz que o interesse de busca não acompanha a POF por UF.
- **O que o resultado não autoriza:** mesmo confirmada, a hipótese não transforma busca em consumo. O Trends recente é mostrado como interesse de busca, e nenhum texto o usa para afirmar que o consumo atual segue o padrão da POF.
- **Ordem:** a metade de 2017-2018 só roda depois do teste final, porque usa o target daquela edição.

---

## 8. Modelos

### 8.1 A escada

As duas partes passam pelos mesmos degraus.

| Degrau | Modelo | Pergunta que responde |
|---|---|---|
| A | Tabela de taxas ponderadas, sem ML | Quanto se consegue só contando? |
| B | Regressão logística regularizada, efeitos simples | As variáveis somam informação além da tabela? |
| C | Logística com curvas para idade e renda e poucas interações definidas na EDA | Os efeitos são curvos, e combinações como dia e horário importam? |
| D | LightGBM com árvores rasas | Existem combinações em que ninguém pensou? |

- **Degrau A:** as dimensões da tabela são fixadas na EDA. Ponto de partida **(escolha do redator)**: dia da semana por célula na parte 1; grande região por local na parte 2. Células da tabela com poucos casos são puxadas para a média geral.
- **Degraus B e C:** a regularização puxa o efeito de cada UF para perto da média. Isso protege os estados com poucos casos e dispensa uma biblioteca de modelo hierárquico.
- **Degrau C:** o número de interações é limitado pela conta de tamanho de amostra da auditoria.
- **Degrau D:** profundidade pequena e número mínimo alto de casos por folha.

### 8.2 Regras que valem para todos

- **Sem reamostragem do evento raro.** Equilibrar as classes distorce a probabilidade, e o score depende dela.
- **Calibração como parte da receita.** A correção sigmoide entra se melhorar o log loss na validação aninhada; é comparada como qualquer outra opção.
- **Mesmo pré-processamento no treino e na inferência,** dentro de um único pipeline.
- **Nada de 2017-2018 entra no ajuste** do modelo avaliado nem do seu pré-processamento.

### 8.3 Ordem das escolhas e o que é a receita

As escolhas são encadeadas nesta ordem **(escolha do redator)**, todas em 2008-2009:

1. **Variáveis e faixas de horário,** fixadas na EDA.
2. **Peso no treino (H5),** medido no degrau B de cada parte. O resultado vale para todos os degraus daquela parte.
3. **Escada (H1 a H3),** degrau por degrau, cada um com os hiperparâmetros escolhidos nas dobras internas.
4. **Calibração,** com e sem a correção sigmoide, só no vencedor da escada.
5. **Produto (H6 e H7),** com os vencedores das duas partes.
6. **Cortes** do score e do Decision Engine.

O efeito de "tem pizza" (H4) é medido no degrau B da parte 2, pelo coeficiente da logística.

A receita congelada é o conjunto: variáveis, faixas de horário, uso do peso, degrau vencedor de cada parte com seus hiperparâmetros, calibração, forma de cálculo do score e cortes.

### 8.4 O que foi descartado

- **Redes neurais e combinação de modelos:** custo alto para o volume de dados.
- **Reponderar o treino para parecer com 2017-2018:** usaria o teste no ajuste.
- **Modelo hierárquico em biblioteca própria:** a regularização cumpre o mesmo papel com menos dependências.

---

## 9. Occasion Score

### 9.1 Cálculo

- **Parte 1:** chance de pizza com refrigerante na célula, dado o contexto.
- **Parte 2:** chance de o refrigerante ser guaraná, dado o contexto e com "tem pizza" marcado como sim.
- **Produto:** chance de pizza com guaraná na célula.

Como a parte 2 é calculada já com "tem pizza = sim", o produto é a regra da cadeia de probabilidades, não uma suposição de independência. As hipóteses H6 e H7 conferem esse produto contra o evento observado.

### 9.2 Como o número aparece

| Campo | Conteúdo |
|---|---|
| `occasion_score` | De 0 a 100: posição do contexto entre todos os contextos da população de referência, ponderada. Score 90 significa estar entre os 10% mais prováveis |
| `classification` | `alta`, `média` ou `baixa` |
| `score_components` | Posição de 0 a 100 na parte 1 e na parte 2, e a probabilidade calibrada de cada uma. É um campo a mais no contrato **(escolha do redator)** |
| Probabilidade | A probabilidade calibrada do produto acompanha o score |

- **Por que posição relativa:** a probabilidade bruta de uma célula é da ordem de centésimos de por cento, difícil de ler. E uma posição resiste à mudança na frequência geral entre as edições.
- **População de referência:** todas as linhas da base da parte 1, com peso amostral. Para o modelo avaliado, é 2008-2009. Para o modelo servido, é a edição mais recente presente no treino **(escolha do redator)**.
- **Posição de cada parte:** as posições da parte 1, da parte 2 e do produto são calculadas sobre essa mesma população. Para a parte 2, a probabilidade é prevista em cada linha com "tem pizza = sim".
- **Cortes de `classification`** **(escolha do redator)**: `alta` a partir de 80, `média` de 50 a 79, `baixa` abaixo de 50. Podem ser revistos na validação de 2008-2009 e são congelados junto com a receita.

---

## 10. Explicabilidade

- **Logística:** a contribuição de cada variável sai dos coeficientes.
- **LightGBM:** a contribuição sai do SHAP.
- **Formato único:** nas duas, a contribuição é medida na escala de log-chances, em relação ao contexto médio.
- **`top_factors`** **(escolha do redator)**: lista dos fatores de maior contribuição absoluta, somando as duas partes. Cada item traz a parte a que pertence (ocasião ou preferência), a variável, o valor informado, a direção (aumenta ou diminui) e o tamanho da contribuição.
- **Visão global:** importância das variáveis por modelo, no relatório de avaliação.
- **Estabilidade:** as contribuições são comparadas entre as dobras da validação. Fatores que mudam de sinal entre dobras são marcados como instáveis no relatório.
- **Quando o modelo é a tabela de taxas** **(escolha do redator)**: a contribuição de cada dimensão da tabela é a diferença, em log-chances, entre a taxa daquele valor e a taxa geral. É o que a aplicação mostra enquanto só o degrau A existe.
- **Quando o modelo direto assume,** no caso previsto em H6: os fatores vêm do modelo direto e são marcados como parte "direto".

---

## 11. Decision Engine

Regras explícitas sobre o score, seus componentes e o contexto. Não há modelo nesta camada. A recomendação tem duas camadas.

### 11.1 Primeira camada: objetivo

Sai do cruzamento das duas partes do score.

| | Preferência por guaraná alta | Preferência por guaraná baixa |
|---|---|---|
| **Ocasião alta** | Sustentar: manter presença onde já funciona | Converter: a ocasião existe e o guaraná fica atrás |
| **Ocasião baixa** | Criar ocasião: o guaraná é preferido e falta a pizza | Não priorizar |

Definições **(escolha do redator)**:

- **Ocasião alta:** posição na parte 1 igual ou acima de 75.
- **Preferência alta:** posição na parte 2 igual ou acima de 50, isto é, o contexto está na metade em que o guaraná é mais provável.

Em termos absolutos, a cola lidera na maioria dos contextos. Por isso "preferência alta" significa guaraná acima do contexto mediano, e não guaraná à frente da cola. Os textos da aplicação usam essa formulação.

Os dois cortes podem ser revistos na validação de 2008-2009. Eles são congelados com a receita e acompanhados de uma análise de sensibilidade: quantos contextos mudam de quadrante se o corte se mover dez pontos.

### 11.2 Segunda camada: canal

Regras em ordem; vale a primeira que se aplicar **(escolha do redator para a ordem e os limites)**:

| Ordem | Condição | Canal principal |
|---|---|---|
| 1 | Objetivo é "não priorizar" | Nenhum |
| 2 | Objetivo é "converter" e idade abaixo de 30 | Influencers e conteúdo digital |
| 3 | Local é fora de casa | Ação na pizzaria: combo de mesa e presença no cardápio |
| 4 | Local é em casa e área urbana | Combo em app de delivery |
| 5 | Local é em casa e área rural | Pizza e guaraná juntos no varejo |

Canal complementar, que pode acompanhar o principal:

- **Mídia regional,** quando a UF está entre as de maior preferência por guaraná.

A regra 5 é uma regra de negócio, não um resultado da POF: parte da suposição de que apps de delivery têm cobertura menor em área rural. Ela é documentada como suposição.

### 11.3 Saída

- **`recommendation_category`:** objetivo e canal principal, por exemplo "Converter, via combo em app de delivery". São dez categorias possíveis mais "Não priorizar".
- **`recommendation_reason`:** texto montado por modelo fixo a partir da regra que disparou e dos fatores de `top_factors`.

### 11.4 Limites

- **Sem alegação de efeito.** A recomendação diz que a ativação combina com o contexto. Ela não diz que uma ação vende mais que outra.
- **Só adultos** **(escolha do redator)**: a aplicação aceita idade a partir de 18 anos. O modelo é treinado com todas as pessoas de 10 anos ou mais, mas a aplicação não recomenda ativação de refrigerante para público menor de idade.

---

## 12. Modelo avaliado e modelo servido

São dois artefatos com nomes distintos.

| Artefato | Treinado em | Para que serve |
|---|---|---|
| Modelo avaliado | 2008-2009 | Produz todas as métricas do projeto |
| Modelo servido | Base inteira, ou só 2017-2018 se H9 for refutada | É o que a aplicação usa |

Travas do reajuste:

1. **Ordem:** só acontece depois que o resultado do teste final está gravado e commitado.
2. **Receita congelada:** variáveis, tipo de modelo, hiperparâmetros e método de calibração são os escolhidos em 2008-2009. Nada é reescolhido.
3. **Calibração refeita** por validação cruzada, com o mesmo método, usando só as linhas da edição mais recente **(escolha do redator)**. Assim a probabilidade exibida e a população de referência do score são da mesma época.
4. **Sem indicador de edição.** O modelo servido usa exatamente as variáveis da receita.
5. **Transparência:** o `model_metadata` diz em que dados o modelo servido foi treinado e que as métricas exibidas são as da receita no teste temporal.

O modelo servido nunca é testado, porque não sobra dado para isso. Essa limitação é declarada.

**Plano B de prazo:** servir o próprio modelo avaliado, sem reajuste.

O reajuste na base inteira nunca acontece sem o experimento de H9. Juntar as duas edições no treino é um limite declarado, porque o método de coleta mudou entre elas.

---

## 13. Google Trends

- **Papel:** conferência, dentro da EDA. Nunca é variável do modelo, target, nem entrada do score ou da recomendação.
- **O que se compara:** hipótese H10. A comparação é feita por UF e dentro do mesmo período, que são chaves coerentes entre as duas fontes.
- **Termos** **(escolha do redator)**: os tópicos "Guaraná Antarctica" e "Coca-Cola", comparados na mesma consulta para ficarem na mesma escala. São as marcas dominantes de cada sabor e entram como aproximação do sabor; as demais marcas ficam de fora, e isso é limite declarado.
- **Filtros:** Brasil, pesquisa na web, todas as categorias, interesse por sub-região (estado).
- **Medida por UF:** interesse por guaraná dividido pela soma do interesse por guaraná e por cola.
- **Coleta:** arquivos CSV exportados do site do Google Trends pelo autor do projeto, um por janela, e guardados no repositório com a data da extração, os termos e os filtros. A exportação pode ser feita a qualquer momento, porque não usa a POF. A biblioteca não oficial não é usada, porque costuma ser bloqueada.
- **Linguagem:** o Trends é sempre descrito como interesse de busca. Nenhum texto o usa como evidência de consumo atual.
- **Janelas:** o período de coleta de cada edição da POF, conforme a documentação do IBGE, e os doze meses mais recentes.
- **Limites declarados:** são 27 pontos; o volume de buscas em 2008-2009 era menor, e estados pequenos podem vir zerados; busca por marca tem outros motivos além de consumo.
- **Extras, só se sobrar tempo:** sazonalidade ao longo do ano e conferência do padrão semanal, mostrados como contexto de interesse de busca.

---

## 14. Aplicação

### 14.1 Função de inferência

Uma função única recebe o contexto e devolve o contrato de saída. A tela, a API e os testes chamam essa mesma função. Ela usa o mesmo pipeline de pré-processamento do treino.

### 14.2 Contrato de entrada

| Campo | Valores |
|---|---|
| `uf` | Uma das 27 UFs |
| `situacao` | `urbano` ou `rural` |
| `dia_semana` | Segunda a domingo |
| `faixa_horario` | Uma das faixas congeladas |
| `local` | `em_casa` ou `fora_de_casa` |
| `idade` | Inteiro, a partir de 18 |
| `sexo` | Como registrado na POF |
| `faixa_renda` | Uma faixa de renda mensal por pessoa, em salários mínimos |
| `numero_moradores` | Inteiro, se a variável entrar na receita |
| `versao_contrato` | Texto |

As faixas de renda são definidas na EDA de 2008-2009. Cada faixa é convertida para o valor que o modelo usa pelo seu ponto médio **(escolha do redator)**.

### 14.3 Contrato de saída

Os seis campos do TAPI, mais `score_components`:

```json
{
  "occasion_score": 87,
  "classification": "alta",
  "score_components": {
    "ocasiao": {"posicao": 92, "probabilidade": 0.0031},
    "preferencia": {"posicao": 31, "probabilidade": 0.19}
  },
  "top_factors": [
    {"parte": "ocasiao", "variavel": "faixa_horario", "valor": "noite", "direcao": "aumenta", "contribuicao": 1.4}
  ],
  "recommendation_category": "Converter, via combo em app de delivery",
  "recommendation_reason": "…",
  "model_metadata": {
    "modelo": "servido",
    "treinado_em": ["2008-2009", "2017-2018"],
    "receita_commit": "…",
    "metricas_da_receita": "treino em 2008-2009, teste em 2017-2018",
    "instante_de_previsao": "planejamento, antes da ocasião",
    "aviso": "guaraná é o sabor registrado na POF, não a marca"
  }
}
```

Os valores acima são ilustrativos. O contrato é definido em Pydantic.

### 14.4 Tela

Streamlit, em uma página:

1. **Formulário de contexto** à esquerda.
2. **Resultado** à direita: score, classificação, posição nas duas partes, fatores em barras e recomendação com a justificativa.
3. **Mapa de calor** abaixo do resultado: o score para cada dia da semana e faixa de horário, mantidos os demais campos do formulário. É calculado chamando a função de inferência para cada combinação.
4. **Botão de briefing,** que redige um texto a partir da decisão.
5. **Painel de metadados:** em que dados o modelo foi treinado, as métricas do teste temporal lidas do arquivo e os limites da fonte.

Os gráficos seguem a skill `dataviz`. Formatos descartados: painel de exploração como formato principal, por custo de tempo, e conversa em texto livre, pelo risco de o LLM entender o contexto errado ao vivo.

### 14.5 API

FastAPI como camada fina sobre a função de inferência, com um teste de contrato. A tela não depende dela: chama a função direto, e o vídeo só precisa de um processo rodando. É o primeiro item da ordem de corte.

### 14.6 Briefing por LLM

- **Entrada:** só a decisão estruturada, isto é, o contrato de saída. Nenhum microdado sai da máquina.
- **Provedor:** qualquer um com interface compatível com a da OpenAI, como Groq ou NVIDIA NIM. Endereço, modelo e chave vêm de variáveis de ambiente. O modelo é escolhido na implementação, consultando a documentação do provedor.
- **Instrução ao LLM:** redigir um briefing curto usando apenas os fatores, os números e a categoria recebidos; não afirmar efeito de venda; tratar guaraná como sabor.
- **Conferência automática:** depois da resposta, o código verifica se todo número citado no texto existe na entrada e se o texto não contém termos proibidos. Se falhar, descarta a resposta.
- **Texto reserva:** sem chave, sem rede, com erro ou com a conferência reprovada, a aplicação mostra um texto montado por modelo fixo. O vídeo não depende de internet.
- **Chave:** fica em um arquivo `.env` que não é commitado.

---

## 15. Testes

Teste antes do código em parsers, bases, funções de variável, Decision Engine, contrato e função de inferência. EDA e experimentos de modelo não têm teste antes; ali a verificação é script reprodutível, semente fixa e métricas gravadas em arquivo. É proibido teste que afirma um limiar de métrica ajustado para passar.

| Camada | O que é testado |
|---|---|
| Dados | Esquema validado em cada etapa: tipos, faixas de valores, chave única e peso positivo. Os números de referência nos parsers |
| Código | Testes unitários das funções de variável; teste de integração do pipeline inteiro em amostra pequena |
| Reprodutibilidade | Treinar duas vezes com a mesma semente dá as mesmas métricas |
| Treino e aplicação iguais | A função de inferência devolve o mesmo score que o pipeline em lote para as mesmas linhas |
| Comportamento do modelo | O score fica entre 0 e 100; mudar um campo que não é entrada não altera o resultado; a saída respeita o contrato para qualquer entrada válida |
| Vazamento | Com o target embaralhado, o modelo não pode prever melhor que uma previsão constante igual à frequência geral, além da incerteza. A comparação é feita na mesma rodada, contra os rótulos embaralhados, não contra um número fixo |
| Decision Engine | Testes gerados automaticamente: toda entrada válida cai em exatamente uma categoria; as regras não se sobrepõem; idade abaixo de 18 é recusada |
| Briefing | A conferência automática reprova um texto com número inventado ou termo proibido; o texto reserva aparece quando não há chave |
| Contrato | A API e a função devolvem os campos obrigatórios com os tipos certos |

---

## 16. Arquitetura e stack

### 16.1 Camadas

Coleta, processamento, modelo, regras e interface ficam em módulos separados, como o TAPI pede:

1. **Ingestão:** download e parsers.
2. **Bases:** esquema comum, base da parte 1 e base da parte 2.
3. **Variáveis:** funções de transformação, usadas no treino e na inferência.
4. **Modelos:** escada, validação, calibração e registro de experimentos.
5. **Score:** produto, posição relativa e classificação.
6. **Decisão:** regras do Decision Engine.
7. **Inferência:** função única e contrato.
8. **Interface:** tela, API e briefing.

### 16.2 Stack

Tudo nesta lista é **escolha do redator**, apresentada ao autor e não vetada.

| Para quê | Ferramenta |
|---|---|
| Linguagem e ambiente | Python 3.12 fixado pelo `uv`, com arquivo de trava de versões. O Python do sistema é o 3.14, recente demais para garantir a instalação de LightGBM e SHAP |
| Dados | pandas e arquivos Parquet |
| Validação de esquema | pandera |
| Modelos | scikit-learn, LightGBM e SHAP |
| Testes | pytest e Hypothesis |
| Tipos e estilo | Pyright e ruff |
| Contrato | Pydantic |
| Tela | Streamlit |
| API | FastAPI |
| Comandos | `Makefile` com os passos: baixar, montar bases, auditar, treinar, avaliar, testar e abrir a aplicação |
| Versionamento | Git |

Antes de escrever código que usa uma biblioteca, a documentação da versão fixada é consultada pelo servidor `context7`, como o `CLAUDE.md` determina.

---

## 17. Entregáveis e documentação

| Entregável do TAPI | Onde é atendido |
|---|---|
| 1. Dataset, Data Card e pipeline | Seções 4 e 16; Data Card com os itens que o TAPI lista |
| 2. EDA e variáveis | EDA em 2008-2009; seção 5.4 |
| 3. Baseline, modelo de maior capacidade e score | Seções 8 e 9 |
| 4. Métricas, erros, segmentos e explicabilidade | Seções 6 e 10 |
| 5. Decision Engine com regras documentadas | Seção 11 |
| 6. Aplicação funcional | Seção 14 |
| 7. Vídeo e README | Abaixo |

- **Data Card:** unidade de análise, target, dicionário de variáveis, fontes e versões, regras de integração, estratégia contra vazamento, população coberta e excluída, vieses e limitações, licença e cuidados com a LGPD. Os microdados da POF são públicos e anonimizados; o projeto não tenta reidentificar ninguém e não envia microdados a serviços externos.
- **Relatório de hipóteses:** o resultado de cada hipótese da seção 7.
- **README:** arquitetura, dados, metodologia, resultados, limitações e execução. Inclui os próximos passos: rodar o método na POF 2024-2025 quando o consumo alimentar for publicado, o que transforma a validação em duas rodadas e testa o modelo servido; coletor de cardápios como medida de oferta; e a "mesa".
- **Vídeo, até 5 minutos:** uma execução completa da entrada à recomendação, as métricas, a estratégia de validação, as fontes, a unidade de análise, o target e a arquitetura. Uma frase explica os dois modelos: o desempenho foi medido treinando em 2008-2009 e testando em 2017-2018, e a aplicação usa a mesma receita reajustada.

Toda métrica citada em texto é lida do arquivo gerado pelo código.

---

## 18. Ordem de construção e ordem de corte

### 18.1 Ordem

O caminho inteiro fica pronto primeiro com o degrau A; os modelos melhores entram depois.

| Dias | O que fica pronto |
|---|---|
| 12 e 13/11 | Design, plano e parsers das duas edições, conferidos pelos números de referência |
| 14 e 15/11 | Bases, auditoria, degrau A e o caminho inteiro funcionando: função de inferência, Decision Engine e tela |
| 16 a 18/11 | EDA em 2008-2009, hipóteses commitadas, escada de modelos e calibração |
| 19/11 | Receita congelada, teste final, análise de erros, segmentos e explicabilidade |
| 20/11 | Diagnóstico depois do teste, H9, modelo servido e bloco do Trends |
| 21/11 | Tela completa: mapa de calor e briefing |
| 22/11 | Data Card, README e revisão geral |
| 23 a 25/11 | Vídeo e folga |

Os parsers das duas edições podem ser feitos em paralelo. O Decision Engine, o esqueleto da tela e da API, a camada de briefing, o Data Card e o README também, depois que o contrato de saída estiver fixado.

### 18.2 Ordem de corte, se o prazo apertar

1. Camada FastAPI.
2. Degrau C da escada.
3. Bloco do Trends.
4. Hipótese H9 e reajuste, caindo para o plano B de servir o modelo avaliado.
5. Briefing por LLM, caindo para o texto reserva.

Os entregáveis obrigatórios do TAPI não entram na lista de corte.

---

## 19. Escolhas feitas pelo redator

Estas decisões não foram discutidas uma a uma com o autor. Estão aqui para ele conferir.

1. Faixas de horário iniciais: madrugada, manhã, almoço, tarde e noite, com os limites da seção 5.2.
2. Local da ocasião positiva é o do item de pizza.
3. Número de moradores como variável candidata.
4. Renda por pessoa em salários mínimos da data de referência de cada edição.
5. Dimensões iniciais da tabela de taxas do degrau A.
6. Tamanho da busca de hiperparâmetros: até 8 valores na logística e até 30 configurações no LightGBM.
7. Rodada extra de generalização deixando uma grande região de fora.
8. Campo `score_components` a mais no contrato.
9. População de referência do score: a edição mais recente presente no treino.
10. Cortes de `classification`: 80 e 50.
11. Formato de `top_factors`, com a parte a que cada fator pertence.
12. Cortes do Decision Engine: ocasião alta a partir da posição 75; preferência alta a partir da posição 50.
13. "Preferência alta" definida como guaraná acima do contexto mediano, e não à frente da cola.
14. Ordem das regras de canal, o limite de 30 anos para influencers e a regra de área rural para o varejo.
15. Aplicação restrita a público de 18 anos ou mais.
16. Conversão da faixa de renda pelo ponto médio.
17. Toda a stack da seção 16.2.
18. Registro de decisões em arquivo separado deste.
19. Ordem das escolhas encadeadas e definição da receita (seção 8.3).
20. "Tem pizza" permanece no modelo qualquer que seja o resultado de H4.
21. Regra de H8: o score contra o evento observado, com PR-AUC, comparado ao produto das tabelas de taxas.
22. Calibração do modelo servido só com as linhas da edição mais recente, e sem indicador de edição.
23. Fatores explicativos quando o modelo é a tabela de taxas ou o modelo direto.
24. Termos e filtros do Google Trends.

---

## 20. O que fica para quem implementa

- Estrutura de pastas e nomes de módulos.
- Posições dos campos e listas de códigos de alimento de 2008-2009, a partir da documentação da edição.
- Nomes das colunas do esquema comum.
- Valores exatos dos hiperparâmetros dentro dos limites da seção 6.5.
- Aparência da tela, seguindo a skill `dataviz` para os gráficos.
- Texto do modelo fixo de `recommendation_reason` e do briefing reserva.
- Escolha do modelo de LLM dentro do provedor.
- Lista de termos proibidos da conferência do briefing.
