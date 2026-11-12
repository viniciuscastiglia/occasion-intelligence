# Occasion Intelligence – Pizza + Guaraná

*TAPI*

**IA | ambev**

## 1. Contexto e desafio

Guaraná Antarctica é uma marca centenária, presente em mais de 70% dos lares brasileiros e associada a cotidiano, festa, família e tradição. Ao longo das últimas décadas, a marca também explorou sua conexão com alimentos, especialmente a pizza.

A ocasião Pizza + Refrigerante é recorrente e se concentra de sexta a domingo, à noite e dentro de casa. Os dados da empresa mostram ainda maior consumo de pizza na Classe A e maior associação de Guaraná Antarctica com a ocasião nesse segmento, o que aponta espaço de expansão em outros públicos.

**O desafio** é aumentar a frequência da associação entre Pizza + Guaraná Antarctica, identificando em quais contextos essa combinação tem maior potencial. Uma ocasião não depende de uma única variável: dia, horário, público, região, canal e histórico interagem entre si. Regras fixas, como “ativar nas noites de sexta”, não capturam essas interações.

Por isso, tratamos o problema como **Occasion Intelligence**: usar dados para estimar quando e em quais contextos a ocasião é mais provável e, a partir disso, apoiar a decisão de ativação.

## 2. Objetivo e pergunta norteadora

Desenvolver um sistema orientado a dados que identifique contextos com maior propensão para a ocasião Pizza + Guaraná e transforme essa estimativa em recomendações estruturadas de ativação. O projeto cobre o pipeline completo: construção da base, features, modelos, score, interpretação, Decision Engine e aplicação funcional.

**Como usar dados comportamentais e contextuais para estimar a propensão ou a intensidade de contextos relevantes para a ocasião Pizza + Guaraná e transformar essa estimativa em recomendações estruturadas de ativação?**

## 3. Como o sistema funciona

O sistema separa três decisões, cada uma com um papel claro:

- **Occasion Prediction:** o modelo estima a propensão (ou intensidade) de uma observação representar a ocasião. A saída é o Occasion Score.
- **Decision Engine:** aplica regras explícitas sobre score, contexto e fatores explicativos para escolher uma categoria de ativação coerente, com justificativa.

**Inteli Academy x Ambev | TAPI | 1**

---

> **IA | ambev | Occasion Intelligence – Pizza + Guaraná**

- **Camada generativa (opcional):** um LLM pode transformar a decisão estruturada em texto, sem calcular o score nem inventar fatores.

O Occasion Score é a saída principal do modelo. Em classificação, corresponde à probabilidade estimada P(y=1|X); em regressão, ao valor previsto ou a uma transformação documentada dele. Seu significado é definido antes da avaliação. Um contexto de sexta à noite, com alto histórico de pizza, consumo doméstico e recorrência em delivery tende a receber score alto; um contexto de baixa afinidade, score menor.

O score responde “*quão relevante é este contexto para a ocasião?*” e não decide qual campanha fazer. Essa escolha pertence ao Decision Engine.

**Fluxo:** fontes de dados › base › features › modelo › Occasion Score e explicação › Decision Engine › aplicação.

## 4. Dados e formulação do problema

Não existe base nacional recente que registre diretamente cada ocasião Pizza + Guaraná. Construir e documentar a base faz parte do desafio: cada grupo seleciona e combina fontes abertas ou fornecidas pela empresa, justificando as escolhas.

- **Unidade de análise:** definida pelo grupo e mantida em todo o pipeline (por exemplo, pedido, consumidor x período ou município x semana).
- **Target:** uma ocorrência, intensidade ou medida observável da ocasião, em formulação binária ou de regressão. Deve ter significado de negócio, estar disponível de forma consistente, não depender de informação futura e ter limitações documentadas. Não se cria target artificial, e o tipo de ativação nunca é o target.
- **Papéis das fontes:** *ground truth* para o target (ex.: pedidos de delivery, POF, Vigitel); *features* contextuais e territoriais (ex.: clima INMET, Censo/SIDRA, densidade de pizzarias no OSM); *proxies* de interesse (ex.: Google Trends), que nunca são tratados como consumo real.
- **Integração:** fontes com períodos ou granularidades incompatíveis não são combinadas como se fossem a mesma observação; usam-se chaves coerentes (data, código IBGE, UF, município). Se nenhuma fonte sustentar um target defensável, a formulação é revista antes da modelagem.
- **Prevenção de data leakage:** o grupo declara o instante de previsão, e toda feature precisa estar disponível até ele. O mesmo processamento vale para treino e inferência.
- **Data Card obrigatório:** unidade de análise, target (ground truth ou proxy), dicionário de variáveis, fontes e versões, regras de integração, estratégia anti-leakage, população coberta e excluída, vieses e limitações, licença e cuidados com LGPD.

## 5. Escopo

**Dentro do escopo**

- Base documentada, Data Card e pipeline reprodutível de ingestão, integração e preparação.
- Análise exploratória e feature engineering.
- Baseline simples e pelo menos um modelo de maior complexidade, com Occasion Score e avaliação por segmentos.

**Inteli Academy x Ambev | TAPI | 2**

---

> **IA | ambev | Occasion Intelligence – Pizza + Guaraná**

- Decision Engine baseado em regras documentadas.
- API ou função de inferência, aplicação funcional e README com experimentos reprodutíveis.

**Extensões opcionais:** clusterização de contextos, LLM para texto, MLflow e monitoramento de drift. Os demais entregáveis estão na Seção 9.

**Fora do escopo**

- Veiculação real de campanhas, compra de mídia, publicação automática, peças gráficas finais e integrações com plataformas privadas.
- Alegar impacto causal de uma ativação sem dados ou experimento que o sustentem.
- Treinar um modelo para escolher a “melhor campanha” sem histórico de ações e resultados.

## 6. Modelagem, avaliação e explicabilidade

- **Baseline obrigatório:** uma estratégia que funcionaria sem Machine Learning, como a frequência histórica por dia da semana e período do dia. Ele mostra se o modelo aprende algo além de “sexta à noite é bom para pizza”.
- **Modelos candidatos:** ao menos um modelo de maior capacidade, como Regressão Logística, Random Forest ou Gradient Boosting (XGBoost/LightGBM). A escolha final é sustentada por experimentos.
- **Validação:** com componente temporal, treino em períodos anteriores e avaliação em posteriores. Splits por região ou canal medem generalização. Embaralhamento aleatório não é a única avaliação final.
- **Métricas:** em classificação, Precision, Recall, F1, ROC-AUC/PR-AUC, matriz de confusão e calibração quando o score for probabilístico; em regressão, métricas adequadas ao target. A métrica principal é justificada pelo custo dos erros: um falso positivo gera uma ativação pouco relevante; um falso negativo deixa passar uma ocasião relevante.
- **Análise por segmentos:** período, região, público e canal, para que o resultado global não mascare diferenças importantes.
- **Explicabilidade:** SHAP, feature importance ou permutation importance calculadas sobre o modelo. A aplicação exibe os principais fatores de cada predição; um LLM, se usado, apenas comunica fatores sustentados pela camada analítica.

## 7. Decision Engine

O Decision Engine recebe o Occasion Score, o contexto observado e os fatores relevantes, e devolve uma **categoria de ativação coerente com o contexto**, com justificativa. As regras que ligam contexto à recomendação ficam documentadas. Exemplos:

- **Score alto + forte presença de delivery:** ativação via delivery.
- **Score alto + consumo doméstico recorrente:** promoção, combo ou ação para consumo em casa.
- **Score relevante + segmento com baixa associação atual à marca:** awareness e construção de associação.

**Inteli Academy x Ambev | TAPI | 3**

---

> **IA | ambev | Occasion Intelligence – Pizza + Guaraná**

Sem histórico de contexto, ação e resultado, a recomendação é uma **decisão estruturada por regras**, e não uma segunda previsão supervisionada. O sistema não afirma causalidade nem que uma ação venderá mais que outra.

## 8. Arquitetura, entregáveis e demonstração

**Arquitetura e stack.** Coleta, processamento, modelo, regras de negócio e interface ficam em camadas separadas, e a aplicação permite inspecionar o funcionamento real do sistema. Stack de referência, com a escolha final justificada na documentação:

- **Coleta e integração:** Python com requests/httpx para as APIs abertas (Open-Meteo, BrasilAPI, SIDRA, Overpass) e exportação ou biblioteca como pytrends para o Google Trends.
- **Dados:** Pandas ou Polars, arquivos Parquet e DuckDB ou PostgreSQL/PostGIS quando houver dado geográfico.
- **Modelagem e explicabilidade:** Scikit-learn (baseline, pipelines e calibração), XGBoost ou LightGBM e SHAP.
- **Serviço e interface:** FastAPI com Pydantic para o contrato de entrada e saída, e Streamlit (com Folium ou pydeck para mapas) para a demonstração.
- **Reprodutibilidade:** Git, ambiente fixado (requirements ou uv, Docker se necessário), testes com pytest e MLflow como opcional.

**Contrato da aplicação.** A entrada reune atributos contextuais, temporais, de público, canal e região, além de metadados de versão. A saída traz `occasion_score`, `classification`, `top_factors`, `recommendation_category`, `recommendation_reason` e `model_metadata`.

**Entregáveis**

1. Dataset, Data Card e pipeline de dados.
2. EDA e feature engineering.
3. Occasion Prediction Model: baseline, candidatos, validação e Occasion Score.
4. Avaliação e interpretabilidade: métricas, análise de erros, segmentos e explicações.
5. Decision Engine com regras documentadas.
6. Aplicação funcional: entrada de contexto, score, fatores e recomendação.
7. Vídeo demonstrativo e README (arquitetura, dados, metodologia, resultados, limitações e execução).

**Vídeo (até 5 minutos).** Deve mostrar ao menos uma execução completa, da entrada ao Occasion Score e à recomendação, além das métricas, da estratégia de validação, das fontes, da unidade de análise, do target e da arquitetura, com resultados reprodutíveis a partir do código entregue.

**Critério eliminatório: a ausência do vídeo, ou um vídeo que não demonstre o funcionamento real do código e da interface, anula a entrega técnica.**

**Inteli Academy x Ambev | TAPI | 4**

---

> **IA | ambev | Occasion Intelligence – Pizza + Guaraná**

## 9. Demais entregáveis

O núcleo do case é prever a ocasião e recomendar uma categoria de ativação. As três entregas abaixo servem de guia para quem quiser ampliar esse núcleo para tendência, território e preço. Elas partem do modelo principal e de dados reais, sem simulações apresentadas como resultado, e não substituem nenhuma entrega das Seções 5 e 8.

### 1. Trend Radar

Identifica se o interesse por Pizza + Guaraná está crescendo ou caindo em cada cidade e período, combinando buscas (Google Trends), clima (Open-Meteo), feriados (BrasilAPI), perfil territorial (IBGE/SIDRA) e oferta local de pizzarias (OpenStreetMap/Overpass). Buscas medem interesse, não consumo, e entram como sinal de contexto, nunca como target.

**Entrega:** alerta de oportunidade com tendência, local, período e os fatores que explicam o movimento.

### 2. Mapa de Oportunidade

Prioriza regiões e contextos com maior potencial de ativação, cruzando o Occasion Score do modelo principal com a densidade de pizzarias (OpenStreetMap/Overpass) e o perfil territorial (IBGE/SIDRA). A priorização é uma regra documentada do Decision Engine aplicada ao território, e não uma nova previsão.

**Entrega:** mapa ou ranking de cidades e regiões, com justificativa da prioridade e recomendação de canal.

### 3. Laboratório de Preço

Testa combos e faixas de desconto para encontrar oportunidades de margem incremental. Exige dados transacionais que não são públicos: pedidos, itens, preço, desconto, frete, custo, margem, canal, horário e resultado da oferta. Eles vêm de uma integração autorizada com um restaurante parceiro, via Open Delivery (padrão aberto para cardápio, pedidos e dados transacionais, e não uma base pública de preços), PDV ou CRM.

**Referências de coleta pública.** Para observar preços de vitrine de forma exploratória, existem projetos open source no GitHub: *ifood-scraper* (Python e Scrapy, coleta restaurantes e preços a partir de coordenadas), *Web-Scraping-and-EDA-iFood* (Python, Selenium e pandas, com análise em R) e *Python-ExtracaoIfood* (Python, voltado ao iFood Mercado). São referências técnicas, e não fontes oficiais: dependem de interfaces que a plataforma pode mudar, dois deles não informam licença e o uso exige respeitar os termos da plataforma e a LGPD. Preço de vitrine não comprova vendas, margem ou demanda e não substitui os dados do parceiro; entra, no máximo, como contexto de oferta e concorrência.

**Entrega:** desenho de um experimento de preço ou promoção, com hipótese, grupos de teste e controle, métrica de margem e critério de decisão. Só depois de resultados reais e testes A/B é possível falar em elasticidade ou em preço ótimo. Sem parceiro e dados autorizados, a entrega termina no protocolo do experimento.

**Inteli Academy x Ambev | TAPI | 5**
