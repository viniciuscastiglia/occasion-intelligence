# Pré-registro: receita e hipóteses antes dos modelos

**Status:** rascunho para aprovação do autor, 20/11/2026. Nenhum modelo da escada rodou até este commit.

Este documento fixa, antes de qualquer experimento, tudo o que a seção 8.3 do design manda fixar na EDA e os detalhes operacionais das hipóteses H1 a H10. Depois da aprovação, nada daqui muda sem uma nova entrada no registro de decisões, feita antes da etapa afetada rodar.

**Onde cada valor mora.** Os valores estão em código, e este texto os resume:

- `src/occasion/experimentos/especificacao.py`: variáveis de cada parte, espaços de busca, interações e graus de liberdade;
- `src/occasion/features/variaveis.py`: domínios das categóricas e variáveis derivadas;
- `src/occasion/validacao/dobras.py` e `src/occasion/validacao/bootstrap.py`: dobras e bootstrap;
- `src/occasion/experimentos/regras.py`: regras de decisão.

Os números da EDA citados aqui vêm de `reports/eda.json`, gerado por `make eda` só com 2008-2009.

**Origem de cada item.** Cada item diz de onde veio:

- **design:** a especificação aprovada em 12/11/2026;
- **plano:** o Plano 2;
- **EDA:** `reports/eda.json`;
- **autor:** decisão do autor em 20/11/2026.

---

## 1. Faixas de horário

**Valor:** mantidas como no Plano 1 (`config.FAIXAS_HORARIO`).

| Faixa | Horas |
|---|---|
| madrugada | 0 a 5 |
| manha | 6 a 10 |
| almoco | 11 a 14 |
| tarde | 15 a 17 |
| noite | 18 a 23 |

**Origem:** decisão do autor, a partir da seção 1 da EDA.

**O que a EDA mostrou** (`eda.json`, chave `horario`):

- A noite reúne 87,5% das ocasiões de pizza com refrigerante (ponderado), com pico às 20h (34,4% das ocasiões).
- Nenhuma fronteira corta um pico da série do alvo da parte 1.
- Os positivos de `y_pizza_refri` nas células, por faixa, são: madrugada 1, manhã 17, almoço 42, tarde 38, noite 556.
- O único pico cortado é o da série de **todas** as ocasiões com refrigerante. Às 15h estão 6,0% das ocasiões, e a vizinha 14h, com 4,4%, cai no almoço. Essa série não é o alvo da parte 1. Ela aparece porque a parte 2 é treinada nela, mas lá a faixa é só uma variável entre dez, e a divisão não muda o que se pergunta. Nenhuma mudança foi sugerida (`sugestao_de_fronteira: null`).

---

## 2. Variáveis de cada parte

**Origem:** design, seção 5.4. A lista é `COLUNAS_CONTEXTO`, a mesma do Plano 1.

| Parte | Variáveis |
|---|---|
| Ocasião (parte 1), alvo `y_pizza_refri` nas células | dia da semana, faixa de horário, local, UF, região, situação (urbano ou rural), idade, sexo, renda por pessoa em salários mínimos, número de moradores |
| Preferência (parte 2), alvo `y_guarana` nas ocasiões com refrigerante | as mesmas, mais `tem_pizza` (fixada em 1 na inferência) |
| Direto (só em H6), alvo `y_pizza_guarana` nas células | as mesmas da ocasião |

**Categóricas e numéricas.**

- **Categóricas:** dia, faixa, local, UF, região, situação e sexo.
- **Numéricas:** idade, renda e número de moradores.
- **Derivada:** `fim_de_semana` vale 1 de sexta a domingo, o agrupamento "sexta a domingo" do TAPI. Ela é calculada a partir do dia dentro do modelo (`adicionar_derivadas`), então as bases e o contrato da aplicação não mudam. Só entra como dimensão do degrau A e como parte de uma interação do degrau C.

**Domínios fechados** (`DOMINIOS`, origem: plano):

- **Como são montados:** os valores possíveis de cada categórica saem de constantes, nunca dos dados.
  - dia: 1 a 7;
  - faixa: as cinco faixas acima;
  - local: `em_casa` e `fora_de_casa`;
  - UF: os 27 códigos do IBGE;
  - região: as cinco grandes regiões;
  - situação: `urbano` e `rural`;
  - sexo: `masculino` e `feminino`;
  - `tem_pizza` e `fim_de_semana`: 0 e 1.
- **Valor fora do domínio:** os três tipos de modelo o recusam com erro (`checar_dominios`).
- **Valor do domínio que não apareceu no treino de uma dobra:** é aceito, por exemplo uma UF pequena ou "madrugada fora de casa".
- **Conferência nas bases reais:** as bases das duas edições estão dentro dos domínios. Em 2017-2018 só o contexto foi lido, sem nenhuma coluna de alvo (teste `test_dominios_cobrem_as_duas_edicoes`).

---

## 3. A escada e os espaços de busca

Cada degrau escolhe os hiperparâmetros nas dobras internas da validação aninhada, pelo log loss ponderado. No empate fica a primeira configuração da lista, e toda lista está ordenada da mais simples para a mais complexa.

### 3.1 Degrau A: tabela de taxas, escolhida na validação

**Origem:** decisão do autor depois da EDA. O baseline é o mais forte possível, escolhido por teste.

**Por quê.** A tabela do Plano 1 (dia × faixa × local) tem 21 das 70 combinações sem nenhum positivo e 50 com menos de 5 (`eda.json`, `degrau_a.parte_1`). Uma tabela assim perde por ruído. Se B, C e D fossem comparados com ela, uma vitória diria pouco. Por isso o degrau A também é escolhido na validação aninhada, entre 12 configurações por parte: 4 conjuntos de dimensões vezes 3 forças de suavização.

**Suavização `m`.** É a força com que cada combinação é puxada para a taxa geral, em "linhas" de peso médio 1. Os valores são 10, 100 e 1000. A tabela do Plano 1 usava `m = 50`, que fica entre os valores da grade. Ela continua em `models/baseline/` só como artefato do Plano 1 (seção 27 do registro).

**Dimensões.** A coluna "Combinações" diz em quantas células a tabela se divide.

| Parte | Dimensões | Combinações | Origem |
|---|---|---|---|
| Ocasião e direto | faixa × local | 10 | EDA: 1 sem positivo, mediana 15,5 |
| | fim de semana × faixa × local | 20 | EDA: 3 sem positivo, mediana 8 |
| | dia × faixa | 35 | exemplo literal do TAPI |
| | dia × faixa × local | 70 | tabela do Plano 1 |
| Preferência | região | 5 | |
| | região × local | 10 | tabela do Plano 1; EDA: nenhuma combinação sem positivo, mediana 407,5 |
| | UF | 27 | |
| | UF × local | 54 | |

As 12 configurações de cada parte seguem a ordem da tabela, das dimensões com menos combinações para as com mais. Dentro de cada conjunto de dimensões, a ordem vai de `m = 1000` para `m = 10`.

### 3.2 Degrau B: logística com efeitos simples

**Origem:** design (6.5) e plano.

- **Regularização:** L2, com `C` em `GRADE_C_LOGISTICA` (8 valores de 0,001 a 100 em escala logarítmica, do mais regularizado para o menos).
- **Categóricas:** one-hot com todos os níveis do domínio. A regularização puxa cada UF para perto da média, e UF e região juntas dão o efeito de "UF perto da média da região".
- **Numéricas:** padronizadas; a renda entra como `log1p`.

### 3.3 Degrau C: curvas e interações

**Origem:** design (8.1), plano e decisão do autor sobre o conjunto.

- **Curvas:** spline cúbica em idade e em `log1p(renda)`, com 4 nós nos quantis e sem coluna constante. São 5 colunas por curva, no lugar do termo linear. Os nós saem sem peso amostral, porque o peso não chega ao `SplineTransformer` dentro do `ColumnTransformer`. Isso é declarado, não corrigido.
- **Interações:** são conjuntos nomeados e pré-declarados. O conjunto é hiperparâmetro do degrau C, escolhido na validação aninhada junto com `C` (decisão do autor). São 2 × 8 = 16 configurações na ocasião e 8 na preferência.

| Parte | Conjunto | Interações |
|---|---|---|
| Ocasião e direto | `pre_declarado` | dia × faixa; faixa × local; local × situação |
| Ocasião e direto | `fim_de_semana` | fim de semana × faixa; faixa × local; local × situação |
| Preferência | `pre_declarado` (único) | tem pizza × local; tem pizza × região; região × local; faixa × local |

**Por que as interações vêm de argumento e não do tamanho do efeito na EDA.**

- **O risco evitado:** escolher interações pelo alvo de 2008-2009 inteiro e depois medir C contra B na mesma base deixaria H2 otimista.
- **De onde vêm as candidatas:**
  - do enunciado do TAPI, "sexta a domingo, à noite e em casa";
  - das perguntas do projeto: a pizza muda a preferência dentro e fora de casa, e por região?
- **O papel da EDA:** só as descreveu (`eda.json`, `interacoes`). Dos 50 desvios calculados, 16 têm intervalo que exclui zero, contra 2,5 esperados por acaso.

**O conjunto `fim_de_semana`** foi acrescentado depois da EDA, que mostrou que dia × faixa só tem 10 ou mais positivos à noite, além do almoço de domingo.

- **O que a EDA mostrou:** na tabela dia × faixa, 8 das 35 combinações têm 10 ou mais positivos: as 7 noites e o almoço de domingo (10 positivos).
- **O custo do conjunto original:** gasta 24 graus de liberdade nessa interação.
- **A alternativa:** agrupa o dia como o TAPI, e a interação passa a gastar 4 graus.

Como a escolha olhou a EDA, ela não é cega ao alvo. Escolher o conjunto dentro das dobras internas, como qualquer hiperparâmetro, atenua o problema, mas não o elimina: o candidato foi montado olhando 2008-2009 inteiro, inclusive as linhas que depois ficam fora da dobra. O viés possível é a favor do degrau C, então H2 pode sair otimista. A comparação C contra B continua sendo feita fora da dobra.

**Graus de liberdade contra o tamanho de amostra.** Os limites vêm do critério (i) de Riley e colegas, em `reports/auditoria.json`. Cada parâmetro conta assim:

- categórica com `L` níveis: `L − 1`;
- numérica: 1;
- curva: 5;
- interação `a × b`: `(La − 1)(Lb − 1)`.

| Parte | Degrau | Conjunto | Graus | Limite |
|---|---|---|---|---|
| Ocasião | B | | 46 | 171 |
| Ocasião | C | `pre_declarado` | 46 − 2 + 10 + 29 = 83 | 171 |
| Ocasião | C | `fim_de_semana` | 46 − 2 + 10 + 9 = 63 | 171 |
| Preferência | B | | 47 | 227 |
| Preferência | C | `pre_declarado` | 47 − 2 + 10 + 13 = 68 | 227 |
| Preferência, parâmetros de pizza | C | `pre_declarado` | 1 + 1 + 4 = 6 | 6 (recorte de pizza) |
| Direto (H6) | B | | 46 | **32** |
| Direto (H6) | C | `pre_declarado` / `fim_de_semana` | 83 / 63 | **32** |

O recorte de pizza fica exatamente no limite: `tem_pizza`, mais tem pizza × local (1) e tem pizza × região (4). Uma interação a mais envolvendo a pizza passaria do limite.

O limite do direto (32) não está em `auditoria.json`. Ele foi calculado com o mesmo código da auditoria e as contagens simples das células de 2008-2009: 669.030 células, 101 com pizza e guaraná (`LIMITE_RILEY_DIRETO` em `especificacao.py`). O direto passa do limite em B e em C; a decisão sobre isso está em H6.

### 3.4 Degrau D: LightGBM

**Origem:** design (6.5 e 8.1) e plano.

São 30 configurações sorteadas com a semente do projeto (`espaco_lightgbm`):

| Parâmetro | Espaço |
|---|---|
| `learning_rate` | log-uniforme em [0,01; 0,2] |
| `n_estimators` | inteiro em [100, 800] |
| `max_depth` | 2, 3 ou 4, com `num_leaves = 2^max_depth` |
| `min_child_samples` | 200, 500, 1000 ou 2000 |
| `reg_lambda` | log-uniforme em [0,01; 10] |
| `subsample` | 0,7 ou 1,0, com `subsample_freq = 1` |
| `colsample_bytree` | 0,7 ou 1,0 |

- **Fixos:** `objective="binary"`, `deterministic=True`, `force_row_wise=True`, `n_jobs=4`, `random_state=20261112`, `verbose=-1`.
- **Ordem:** a lista vai da configuração mais simples para a mais complexa, por profundidade e depois número de árvores. No sorteio saíram 11 configurações de profundidade 2, 13 de profundidade 3 e 6 de profundidade 4.
- **Categóricas:** entram como categorias do LightGBM, com os níveis de `DOMINIOS`.
- **Contribuições:** saem do TreeSHAP do próprio LightGBM (`pred_contrib=True`).

---

## 4. Validação, incerteza e regras gerais

**Dobras** (origem: design 6.1 e plano):

- **Quantas:** `K_EXTERNO = 5` dobras externas e `K_INTERNO = 3` internas.
- **Como são formadas:** o setor (`cod_upa`) fica inteiro numa dobra, e os setores de cada estrato são espalhados entre as dobras.
- **Internas iguais nas duas partes:** as dobras internas saem da tabela de setores, então são as mesmas nas duas partes. Isso garante que o produto e a calibração do produto nunca misturem previsões de modelos que viram aquele setor.
- **Conferência** (critério pré-declarado): cada dobra externa precisa ter entre 10% e 30% dos positivos de cada alvo. Os alvos conferidos são:
  - `y_pizza_refri` e `y_pizza_guarana` nas células;
  - `y_guarana` nas ocasiões com refrigerante;
  - `y_guarana` no recorte de pizza.
- **Se a conferência reprovar:** a semente avança de 100 em 100, e a troca fica em `reports/dobras.json`.

**Bootstrap** (origem: plano):

- **Como sorteia:** setores inteiros, com reposição, dentro de cada UF; `B = 1000` réplicas; intervalo percentil de 95%.
- **Comparação pareada:** os dois modelos comparados usam as mesmas réplicas.
- **Por que a UF e não o estrato:** os 550 estratos de 2008-2009 têm mediana de 5 setores, e 7 têm um só. Com tão poucos setores, o bootstrap dentro do estrato subestima a variância em cerca de 20%. Por UF são dezenas a centenas de setores.

**Regras que valem para toda comparação** (origem: design 6.2, 6.3 e 8.2):

- **Métricas:** a principal é o log loss ponderado por `peso_final`. PR-AUC, ROC-AUC e Brier ponderados são relatados ao lado.
- **Quando o mais complexo vence:** só se o intervalo do ganho excluir zero a seu favor. No empate, fica o mais simples.
- **Classes:** sem reamostragem do evento raro e sem balanceamento.
- **Pesos de treino:** quando o peso entra no treino (H5), ele é normalizado para média 1 em cada conjunto de treino.
- **Calibração:** usa sempre `peso_final`, porque mira a população.
- **Nada de 2017-2018 no ajuste:** nenhuma linha dessa edição entra no ajuste do modelo avaliado nem do seu pré-processamento. O alvo de 2017-2018 só pode ser lido depois da tag `receita-congelada`.
- **Calibração de cada parte** (seção 8.2 do design): a sigmoide é ajustada nas previsões das dobras internas da dobra externa `k` e aplicada às previsões fora da dobra `k`. Ela entra na receita se o ganho no log loss confirmar superioridade.

**Natureza de cada experimento** (design 6.5):

- **Confirmatórios:** as comparações da escada e as hipóteses H1 a H10. Só eles sustentam conclusões.
- **Exploratórios:**
  - a conferência das dobras;
  - os rótulos embaralhados, que são teste de sanidade contra vazamento;
  - a avaliação por segmento;
  - a análise de erros;
  - a explicabilidade;
  - a rodada de região fora;
  - o diagnóstico depois do teste (6.8).

Todo experimento grava `reports/experimentos/<nome>.json`, inclusive os que falham.

---

## 5. Hipóteses

Cada hipótese é commitada aqui antes da etapa em que é testada.

As regras são funções de `src/occasion/experimentos/regras.py`. Elas recebem o intervalo de 95% do bootstrap de setores e devolvem o veredito:

- `superioridade`: confirmada se o limite inferior do ganho for maior que zero.
- `padrao`: o padrão é mantido a menos que o limite inferior do ganho da alternativa seja maior que zero.
- `sinal_negativo`: confirmada se o intervalo inteiro ficar abaixo de zero.
- `correlacao_positiva`: confirmada se todos os limites inferiores forem maiores que zero.

Em H5, H6 e H9, que têm opção padrão, o resultado sem diferença é escrito "sem evidência contra o padrão", nunca "confirmada", com a largura do intervalo.

### H1. O contexto acrescenta informação além da contagem

- **Enunciado:** em cada parte, a regressão logística prevê melhor que a tabela de taxas.
- **Motivo:** a tabela só enxerga poucas variáveis por vez.
- **Teste:** degrau B contra degrau A, os dois com hiperparâmetros escolhidos nas dobras internas, comparados nas previsões fora da dobra externa de 2008-2009. Na preferência, o mesmo também é relatado no recorte de pizza, sem entrar na regra.
- **Métrica:** ganho de B sobre A no log loss ponderado, com intervalo de 95%.
- **Regra:** `superioridade`.
- **Se refutada:** a tabela de taxas vira o modelo daquela parte, e o relatório diz que o ML não acrescentou.

### H2. Curvas e interações escolhidas melhoram a logística

- **Enunciado:** a logística com curvas para idade e renda e com as interações pré-declaradas prevê melhor que a logística simples.
- **Motivo:** efeitos como o da idade raramente são retos (a EDA, em `curvas`, encontrou faixas fora da reta na idade das duas partes e na renda da ocasião), e combinações como dia e horário são plausíveis.
- **Teste:** degrau C (com `C` e o conjunto de interações escolhidos nas dobras internas) contra degrau B.
- **Métrica:** ganho de C sobre B no log loss ponderado, com intervalo de 95%.
- **Regra:** `superioridade`.
- **Se refutada:** fica o degrau B.

### H3. Existem combinações que não foram previstas

- **Enunciado:** o LightGBM prevê melhor que o melhor modelo logístico.
- **Motivo:** árvores capturam interações sem que alguém as especifique.
- **Teste:** degrau D contra o melhor logístico. Esse melhor é `melhor_logistico` aplicado ao intervalo de H2: C, se H2 for confirmada, senão B.
- **Métrica:** ganho de D sobre o melhor logístico no log loss ponderado, com intervalo de 95%.
- **Regra:** `superioridade`.
- **Se refutada:** o modelo logístico é o escolhido. Com poucos positivos, esse resultado é plausível e é relatado como achado.

**Vencedor da escada (regra do campeão).** As hipóteses H1 a H3 são relatadas como acima. O degrau vencedor de cada parte sai de `vencedor_escada`:

1. O campeão começa em A.
2. B, C e D, nessa ordem, são comparados com o campeão do momento.
3. Cada um só assume se `superioridade` confirmar.

Isso cobre o caso em que B não supera A: aí C e D precisam superar a tabela, e não a logística que perdeu. Quando H1 é confirmada, as duas formas coincidem. Quando não coincidem, o JSON da escada grava as duas comparações, a da hipótese e a do campeão.

### H4. A pizza muda a preferência

- **Enunciado:** dado o mesmo contexto, a presença de pizza reduz a chance de o refrigerante ser guaraná.
- **Motivo:** indício visto em 2017-2018 (seção 4.3 do design), que precisa ser refeito em 2008-2009. A descrição da EDA, sem controlar o contexto, vai no mesmo sentido:
  - guaraná em 23,8% das ocasiões com refrigerante sem pizza (intervalo de 21,9% a 25,8%);
  - 13,9% com pizza (intervalo de 9,2% a 19,3%);
  - diferença de −9,9 pontos (intervalo de −14,9 a −4,4) (`eda.json`, `pizza_e_preferencia`).
- **Teste:** coeficiente de `tem_pizza` no degrau B da preferência, ajustado em 2008-2009 inteira.
  - O `C` é o escolhido por `selecao_final` nas 5 dobras externas.
  - O uso de peso é o decidido em H5 para a preferência.
  - O intervalo vem de 1000 réplicas do bootstrap de setores. Em cada réplica o modelo é retreinado com o `C` fixo e pesos iguais a (`peso_final`, ou 1 sem peso) vezes o multiplicador da réplica, renormalizados para média 1.
- **Métrica:** o coeficiente, em log-chances, com intervalo de 95%. O JSON avisa que o coeficiente fica encolhido pela regularização.
- **Regra:** `sinal_negativo`.
  - Intervalo inteiro abaixo de zero: confirmada.
  - Inteiro acima de zero: "efeito no sentido contrário".
  - Contendo zero: "os dados não mostram o efeito", com a largura do intervalo.
- **Se refutada:** nada muda na receita. A variável permanece no modelo em todos os casos, para o produto continuar sendo a regra da cadeia.

H4 mede associação, não efeito causal da pizza sobre a escolha.

### H5. O peso amostral no treino melhora a previsão

- **Enunciado:** treinar com `PESO_FINAL` dá previsões melhores para a população do que treinar sem peso.
- **Motivo:** a literatura sobre ML em pesquisas amostrais recomenda o peso, mas trata o ajuste de hiperparâmetros com peso como área em aberto.
- **Teste:** degrau B de cada parte, com e sem peso no treino, na validação aninhada. A métrica é sempre ponderada. A decisão vale para todos os degraus daquela parte. O modelo direto de H6 usa a decisão da ocasião.
- **Métrica:** ganho da versão sem peso no log loss ponderado, com intervalo de 95%.
- **Regra:** `padrao`, e o padrão é treinar com peso.
- **Se refutada:** treino sem peso naquela parte, com a métrica ponderada mantida. A calibração continua usando `peso_final`.

### H6. A decomposição em duas partes compensa

- **Enunciado:** o produto das duas partes prevê pizza com guaraná melhor que um modelo treinado direto nesse evento.
- **Motivo:** o evento direto tem 101 células positivas em 2008-2009 (os 101 dias com pizza e guaraná no mesmo horário); as duas partes têm mais casos cada.
- **Teste:** o produto fora da dobra contra o modelo direto, os dois contra `y_pizza_guarana` nas células.
  - **Produto:** a ocasião prevista nas células vezes a preferência prevista nas mesmas células com `tem_pizza = 1`, cada parte com a calibração decidida.
  - **Modelo direto:** do mesmo degrau do vencedor da ocasião, com o espaço de busca desse degrau na parte "direto". Usa o peso decidido em H5 para a ocasião e a calibração decidida para a ocasião, na mesma validação aninhada.
- **Métrica:** ganho do direto sobre o produto no log loss ponderado, com intervalo de 95%. A PR-AUC ponderada é relatada ao lado.
- **Regra:** `padrao`, e o padrão é o produto.
- **Se refutada:** o modelo direto assume o cálculo do score, e as duas partes ficam só como componentes do Decision Engine.
- **Tamanho de amostra do direto (decisão proposta pelo controlador, sujeita à aprovação do autor):**
  - **O número:** pelo critério (i) de Riley, com o mesmo cálculo da auditoria, 101 eventos em 669.030 células sustentam 32 parâmetros. O direto tem 46 graus de liberdade nominais em B, e 63 ou 83 em C. Ele passa do limite nos dois degraus.
  - **A decisão:** o direto é mantido em H6 como está, sem cortar variáveis.
  - **Por quê:** todos os modelos logísticos e o LightGBM são regularizados, com a força escolhida na validação interna, então o número efetivo de parâmetros fica abaixo da contagem nominal. Além disso, H6 compara fora da dobra: um direto que superajusta perde para o produto, que é o padrão.
  - **Se o direto vencer H6:** terá vencido fora da amostra. O relatório informa, ao lado do resultado, a contagem nominal de graus de liberdade e o limite de 32.

### H7. O produto está calibrado contra o fato observado

- **Enunciado:** o produto das duas partes corresponde à frequência observada de pizza com guaraná.
- **Motivo:** duas partes calibradas em separado não garantem que o produto fique calibrado contra o evento conjunto, e a probabilidade exibida na tela é a do produto.
- **Teste:** sobre o que calcula o score (o produto, ou o direto se H6 o promover):
  1. Em cada dobra externa `k`, uma sigmoide é ajustada nas previsões internas daquela dobra.
     - **Produto:** a ocasião interna vezes a preferência interna prevista nas células. As duas estão nas mesmas linhas, porque as dobras internas são as mesmas nas duas partes. Quando a calibração de uma parte entrou, essa parte já vai calibrada.
     - **Direto, se H6 o promover:** as previsões internas do próprio modelo direto, já com a calibração decidida para ele.
  2. A sigmoide é aplicada às previsões fora da dobra `k`.
  3. A avaliação usa só as previsões fora da dobra externa.
- **Métrica:**
  - os três níveis de calibração: média (O/E), inclinação e curva;
  - o ganho da versão corrigida no log loss ponderado, com intervalo de 95%.
- **Regra:** a correção entra na receita se `superioridade` confirmar o ganho no log loss. A inclinação é relatada com intervalo de 95% (reajustando a sigmoide em cada réplica), mas não decide. Com cerca de cem dias positivos, um intervalo que contém 1 indicaria falta de casos, não boa calibração.
- **Se a correção entrar:** ela passa a fazer parte da receita. Por ser monótona, ela muda a probabilidade exibida, não as posições do score.

### H8. O ordenamento sobrevive ao tempo

- **Enunciado:** a receita congelada ordena os contextos de 2017-2018 melhor que a tabela de taxas.
- **Motivo:** é a afirmação central do projeto sobre generalização temporal.
- **Teste:** uma única vez, depois da tag `receita-congelada`.
  - **Receita:** o score do modelo avaliado contra `y_pizza_guarana` nas células de 2017-2018.
  - **Referência:** o produto das duas tabelas do degrau A, com a configuração que `selecao_final` escolheu para cada parte em 2008-2009, ajustadas em 2008-2009 com o uso de peso de H5.
  - **Se a receita for a própria tabela nas duas partes:** a comparação é contra a constante igual à frequência ponderada de 2008-2009.
- **Métrica:** ganho da receita sobre a referência na PR-AUC ponderada. O intervalo vem do bootstrap de setores de 2017-2018 dentro da UF.
- **Regra:** `superioridade`.
- **Relatados à parte, sem entrar na regra:**
  - a mesma comparação em cada parte;
  - a concentração de positivos no décimo superior;
  - todas as métricas da receita, do degrau A e da constante;
  - a queda em relação à validação, por segmento.
- **Se refutada:** o relatório diz que o modelo não superou a contagem simples em 2017-2018. A aplicação continua funcionando, e a limitação vai para a tela e para o README.

A queda em relação à validação é descrita como efeito conjunto de dez anos e da mudança de método de coleta (registro alimentar em 2008-2009, recordatório de 24 horas em 2017-2018). O projeto não separa os dois.

### H9. Juntar as duas edições não prejudica o modelo servido

- **Enunciado:** treinar com 2008-2009 somado não piora a previsão de 2017-2018 em relação a treinar só com 2017-2018.
- **Motivo:** o volume de casos é o gargalo do projeto, mas a relação entre contexto e consumo pode ter mudado.
- **Teste:** depois de o teste final estar commitado, validação cruzada dentro de 2017-2018, com dobras pelo mesmo critério de 10% a 30%. A receita é a congelada, com os parâmetros fixos e sem nova busca. Ela é treinada em dois braços:
  - **recente:** só as dobras de treino de 2017-2018;
  - **base inteira:** as mesmas mais toda 2008-2009, cujas linhas ficam sempre no treino e nunca são avaliadas.

  Os detalhes:
  - **Chave das dobras:** é `(edicao, cod_upa)`. As linhas de 2008-2009 ficam sempre no treino, nas dobras externas e nas internas.
  - **Calibração:** quando a receita a usa, é ajustada nas previsões fora da dobra interna só das linhas de 2017-2018, nos dois braços. É a mesma regra do modelo servido (seção 12 do design).
  - **Modelo direto:** se H6 o promover, ele passa pelos mesmos dois braços.
- **Métrica:** ganho do braço recente sobre a base inteira no log loss ponderado, em três comparações: ocasião, preferência e recorte de pizza.
- **Regra:** `padrao` em cada comparação, com a base inteira como padrão.
  - **Agregação:** a base inteira é recusada se **qualquer uma** das três comparações a recusar. Nesse caso, as duas partes do modelo servido usam só 2017-2018. Senão, as duas usam a base inteira.
- **Se refutada:** o modelo servido é treinado só em 2017-2018.
- **Ordem:** o reajuste na base inteira nunca acontece sem este experimento. Se ele for cortado por prazo, vale o plano B da seção 12 do design: o modelo servido é o próprio avaliado.

### H10. O interesse de busca acompanha a preferência por UF

**Status: não executada.** O bloco do Google Trends foi cortado pelo autor em 20/11/2026, antes de qualquer cálculo com a POF (`dados_externos/google_trends/LEIAME.md` e seção 27 do registro). A hipótese fica registrada inteira, como estava no design, para o relatório dizer o que deixou de ser testado.

- **Enunciado:** por UF, a participação do guaraná no interesse de busca (guaraná contra cola) tem correlação positiva com a participação do guaraná nas ocasiões com refrigerante da POF, no período de cada edição.
- **Motivo:** saber se esse proxy de interesse acompanha a medida real quando os dois cobrem o mesmo período e a mesma unidade geográfica.
- **Teste:** correlação de postos (Spearman) entre as 27 UFs, uma vez por edição. A participação na POF é ponderada por `peso_final`. A janela do Trends é o período de coleta de cada edição.
- **Métrica:** a correlação de postos de cada edição, com intervalo de 95% por reamostragem de UFs (`intervalo_por_unidades`, `B = 1000`).
- **Regra:** `correlacao_positiva`: confirmada se o intervalo de 95% excluir zero, a favor de correlação positiva, **nas duas edições**.
- **Se refutada:** o relatório diz que o interesse de busca não acompanha a POF por UF.
- **O que o resultado não autoriza:** mesmo confirmada, a hipótese não transforma busca em consumo. O Trends recente seria mostrado só como interesse de busca, e nenhum texto o usaria para afirmar que o consumo atual segue o padrão da POF.
- **Ordem:** a metade de 2017-2018 só rodaria depois do teste final, porque usa o alvo daquela edição.
- **Por que não foi executada:**
  1. **Resolução:** a comparação por estado dá a participação em porcentagem inteira. A do guaraná fica entre 1% e 11% em 2008-2009 e quase toda em 2% em 2017-2018, o que não permite ordenar os estados.
  2. **Ruído de marca:** os tópicos juntam buscas que não são sobre a bebida, como bonés e maquiagem licenciada, mesmo com o filtro "Alimentos e bebidas".
  3. **Volume em 2008-2009:** 15 dos 27 estados vêm sem valor para o guaraná.
- **No relatório final:** H10 aparece como "não executada", com esse motivo. Os arquivos ficam no repositório só como registro, e nenhum número deles entra no score, na recomendação ou nos resultados.

---

## 6. Cortes do score e do Decision Engine

**Valores:** mantidos (decisão do autor; origem: design 9.2 e 11.1).

- **`classification`:**
  - `alta` a partir da posição 80;
  - `média` de 50 a 79;
  - `baixa` abaixo de 50.
- **Decision Engine:**
  - ocasião alta com a posição na parte 1 a partir de 75;
  - preferência alta com a posição na parte 2 a partir de 50, isto é, guaraná acima do contexto mediano, não guaraná à frente da cola.

**Análise de sensibilidade** (`cortes.json`, plano). Com a previsão fora da dobra do score final, a análise relata:

- precision, recall, F1 e matriz de confusão no corte de 80;
- a parcela ponderada das células que muda de quadrante quando cada corte do Decision Engine se move 10 pontos para cima e para baixo.

O JSON só relata. Os cortes ficam estes.

---

## 7. Segmentos da avaliação

**Origem:** design 6.4 e plano.

**Segmentos:** dia da semana, faixa de horário, grande região, faixa etária, sexo, faixa de renda e local.

- **Faixas etárias:** 10-17, 18-29, 30-44, 45-59 e 60+.
- **Faixas de renda:** até ½ salário mínimo por pessoa, ½ a 1, 1 a 2, 2 a 5 e mais de 5.
- **Local:** é o substituto declarado do canal, que a POF não tem.

**Mínimo de positivos:** um segmento com menos de 10 positivos sai sem métrica, com o motivo "positivos insuficientes".

**Outras análises por segmento:**

- **Erros:** os 10 contextos (região × dia × faixa × local, com ao menos 500 linhas) com maior diferença padronizada entre o observado e o previsto, para cima e para baixo.
- **Estabilidade:** um fator é marcado como instável se o sinal da contribuição muda entre as dobras e o maior valor absoluto passa de 0,05 log-chances.

---

## 8. O que a EDA mostrou e não virou variável

- **Sazonalidade:** a taxa por trimestre não separa os trimestres, e estação e ano estão confundidos em 2008-2009 (`eda.json`, `sazonalidade`). Nenhuma variável de data entra (seção 26 do registro).
- **Contexto do TAPI:** à noite e em casa, a taxa de pizza com refrigerante de sexta a domingo é 3,2 vezes a de segunda a quinta (intervalo de 2,1 a 5,1). É descrição. O modelo mede isso pelas variáveis dia, faixa e local e pelas interações acima.
- **Curvas:** idade nas duas partes e renda na ocasião têm faixas fora da reta além do esperado por acaso. Na preferência, a renda não tem nenhuma (`eda.json`, `curvas`). As curvas do degrau C valem para as duas variáveis nas duas partes, como o design pede. A EDA não tira a curva da renda da preferência, porque a regularização e a regra de vitória já protegem contra o excesso.

---

## 9. Limites declarados deste pré-registro

- **Escolhas que olharam o alvo de 2008-2009:** o conjunto de interações `fim_de_semana` e as alternativas do degrau A vieram depois de ver o alvo na EDA.
  - **O que atenua:** as duas escolhas são feitas dentro das dobras internas, como hiperparâmetros, e as comparações confirmatórias são fora da dobra.
  - **Por que não elimina:** os candidatos foram montados olhando 2008-2009 inteiro, inclusive as linhas que depois ficam fora de cada dobra.
  - **Direção do viés:** as alternativas do degrau A deixam a referência mais forte, o que é conservador para H1 e H8, porque fica mais difícil superar a tabela. O conjunto `fim_de_semana` favorece o degrau C, e por isso H2 pode sair otimista.
- **Modelo direto de H6 acima do limite de Riley:** o limite do direto é de 32 parâmetros (101 eventos em 669.030 células, mesmo cálculo da auditoria). O direto tem 46 graus nominais em B, e 63 ou 83 em C.
  - **Decisão proposta pelo controlador, sujeita à aprovação do autor:** manter o direto como está. A regularização, com força escolhida na validação interna, deixa o número efetivo de parâmetros abaixo do nominal, e a comparação de H6 é fora da dobra. Um direto que superajusta perde para o produto, que é o padrão.
  - **Se o direto vencer:** terá vencido fora da amostra, e o relatório traz a contagem nominal e o limite ao lado do resultado.
- **Nós das curvas:** saem sem peso amostral.
- **Sabor, não marca:** toda conclusão é sobre o sabor guaraná. A POF não registra marca.
