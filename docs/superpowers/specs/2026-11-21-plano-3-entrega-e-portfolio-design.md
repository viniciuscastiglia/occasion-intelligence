# Plano 3: tela completa, briefing, API, documentação e demo pública: documento de design

- **Data:** 21/11/2026
- **Entrega do case:** por volta de 25/11/2026
- **Situação:** aprovado pelo autor em 21/11/2026. Passou por um revisor independente; as perguntas dele estão incorporadas
- **Documento pai:** `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md`. As seções 14 (aplicação), 15 (testes) e 17 (entregáveis) continuam valendo. Este documento fecha o que elas deixaram em aberto e registra o que mudou depois do Plano 2. Onde os dois divergirem, vale este.

Onde uma escolha foi feita por quem escreveu este documento, e não pelo autor do projeto, ela está marcada com **(escolha do redator)** e listada de novo na seção 7.

---

## 1. Intenção

Os Planos 1 e 2 entregaram o núcleo: bases, escada de modelos, receita congelada, teste final e modelo servido. O Plano 3 transforma isso numa entrega que se sustenta sozinha diante de dois leitores:

- **O avaliador da Liga**, que confere os sete entregáveis do TAPI.
- **Um recrutador ou engenheiro sênior**, que abre o repositório no portfólio do autor.

O que os dois devem levar, nas palavras do autor: o quanto o projeto seguiu o que a literatura mostra como melhores práticas, o quanto foi atrás disso, e que o trabalho tem alta maturidade em todas as frentes, da arquitetura ao método, e não só na avaliação.

### Como o resultado de H8 é contado

H8 foi refutada: o score não superou o produto das tabelas do degrau A em 2017-2018 (`reports/teste_final.json`). A parte de ocasião supera a tabela; a de preferência não.

O resultado é apresentado como ponto forte do método, não como falha a esconder. As hipóteses foram registradas antes, o teste foi aberto uma vez, o resultado foi publicado como veio e o diagnóstico depois do teste explica onde está o limite (a preferência ordena pouco no recorte de pizza, risco já declarado na aprovação da receita). O README e o Data Card contam isso sem suavizar e sem pedir desculpa. **(escolha do redator, proposta ao autor e aceita na conversa)**

---

## 2. Objetivos, não objetivos e o que seria um fracasso

### Objetivos

1. **Tela completa:** mapa de calor de dia da semana × faixa de horário e botão de briefing (seção 14.4 do design pai).
2. **Briefing por LLM** com conferência automática, texto reserva, cache e limite de uso por sessão, funcionando também numa demo pública.
3. **API FastAPI** fina sobre a função de inferência, com teste de contrato.
4. **Data Card em inglês.**
5. **README em inglês**, com as tabelas de resultado geradas por script a partir dos JSON e um teste que falha se ficarem desatualizadas.
6. **Repositório público no GitHub e demo no Streamlit Community Cloud**, com o artefato servido commitado, para que qualquer pessoa experimente a tela sem rodar o pipeline.
7. **Dois caminhos de execução documentados:** abrir a tela a partir de um clone limpo, sem microdados, e reproduzir tudo dos microdados até o teste final.

### Fora do escopo

- **Vídeo.** Fica para depois de tudo pronto, como última etapa da entrega, com no máximo 5 minutos (TAPI, seção 8). O Plano 3 não escreve roteiro.
- **Tradução do resto do projeto para o inglês** (tela, docs em português, registro de decisões). É um projeto separado, depois da entrega. Os identificadores do código ficam em português, porque parte deles está em arquivos assinados (seção 3).
- **Qualquer mudança em modelo, receita, métrica ou experimento.**
- **Google Trends** (cortado no Plano 2), "mesa", coletor de cardápios, clima: continuam como próximos passos no README.
- **Endpoint de briefing na API** **(escolha do redator)**: a API expõe só a previsão. O briefing é da tela.
- **Publicar a API na internet.** Só a tela vai para a demo pública.

### O que seria um fracasso mesmo com tudo rodando

- Qualquer número no README, no Data Card ou na tela que não tenha sido lido de um arquivo gerado pelo código.
- Suavizar, omitir ou reformular H8 para parecer vitória.
- Um briefing que cita número, fator ou efeito que a camada analítica não produziu.
- Uma demo pública que quebra sem chave, sem rede ou com a cota do provedor esgotada.
- Uma mudança no `uv.lock` que altere a versão de numpy, pandas, scipy, scikit-learn, lightgbm ou pyarrow.
- Uma chave de API commitada.
- Texto que afirme efeito causal de ativação ou trate "guaraná" como a marca Guaraná Antarctica.
- Um README que um recrutador não entende em dois minutos de leitura.

---

## 3. Restrições

### Receita congelada

A tag `receita-congelada` (commit `fa700be`) e a assinatura em `src/occasion/bases/assinatura.py` (`ARQUIVOS_DA_RECEITA`) cobrem: `config.py`, `features/`, `models/`, `score/`, `validacao/`, `avaliacao/`, `ingest/`, `bases/{modelagem,ocasioes,assinatura}.py`, `eda/descritivas.py`, `receita.py`, `versoes_da_receita.py`, `inference/{artefato,treino}.py`, `experimentos/{especificacao,teste_final,regras,registro}.py` e `reports/receita.json`.

- O Plano 3 **não toca nesses arquivos**. Pode mexer em `inference/predict.py`, `inference/contract.py`, `pipeline.py`, `decision/`, `app/`, `pyproject.toml` e `uv.lock`, e criar módulos novos.
- Módulos novos ficam em pastas próprias, fora das pastas assinadas: `src/occasion/briefing/` e `src/occasion/api/` **(escolha do redator)**.
- Toda mudança no `uv.lock` (FastAPI, Uvicorn, cliente do LLM) é seguida da conferência das versões instaladas contra `versoes_da_receita.py`, com o mesmo código que o teste final usa.

### Regras do TAPI que o Plano 3 toca

- O LLM só comunica fatores sustentados pela camada analítica; não calcula score nem inventa fatores.
- Nenhuma alegação de efeito causal; "guaraná" é sempre o sabor.
- A aplicação mostra entrada de contexto, score, fatores e recomendação, com o contrato de saída (`occasion_score`, `classification`, `top_factors`, `recommendation_category`, `recommendation_reason`, `model_metadata`).
- O Data Card traz: unidade de análise, target, dicionário de variáveis, fontes e versões, regras de integração, estratégia contra vazamento, população coberta e excluída, vieses e limitações, licença e cuidados com a LGPD.
- O README traz arquitetura, dados, metodologia, resultados, limitações e execução.

### Prazo

Hoje é 21/11. O Plano 3 termina até 24/11, para deixar o vídeo para 25/11.

### Fatos do repositório que pesam

- `/models/` está no `.gitignore`. O artefato servido (`models/servido/artefato.joblib`) tem 39 MB, abaixo do limite de 100 MB por arquivo do GitHub.
- `reports/*.json` e `reports/experimentos/` já são commitados; `data/` não.
- O repositório não tem remoto configurado.
- `carregar_artefato` procura `servido`, depois `avaliado`, depois `baseline`; a tela já prefere o servido.
- O SHAP não é dependência: as contribuições do LightGBM saem do TreeSHAP do próprio LightGBM, e a receita servida é o degrau C (logística).
- O autor confirmou que o enunciado do TAPI (`Occasion_Intelligence_Pizza_Guarana.md`) pode ficar no repositório público.
- **Tudo vai a público (decidido pelo autor):** além do código e dos relatórios, o `CLAUDE.md`, `docs/superpowers/` (design, planos e specs) e `dados_externos/google_trends/` com o `LEIAME.md` do corte. O processo é evidência de método e não é escondido.
- O `gh` local está autenticado na conta do autor com escopo `repo`; o repositório pode ser criado pela linha de comando, com confirmação do autor. O app do Streamlit Community Cloud é criado pelo autor na interface web, depois que o repositório existir.

---

## 4. Abordagens consideradas

| | Como é | O que ganha | O que perde |
|---|---|---|---|
| A. Sequencial | Briefing → tela → API → Data Card → README → publicação | Simples de acompanhar | O deploy só acontece no fim; problemas de ambiente aparecem sem folga |
| B. Duas frentes em paralelo | Frente de código (briefing, tela, API) e frente de documentos (Data Card, README) em worktrees, depois de fixar o contrato | Mais rápido; o `CLAUDE.md` já prevê esse paralelismo | O deploy ainda fica no fim |
| C. Deploy primeiro | Publica a tela atual no primeiro dia; cada funcionalidade nova sobe sozinha | Riscos de ambiente aparecem cedo | O repositório fica público antes de o README estar pronto |

**Escolhida pelo autor: B, com a primeira tarefa vinda de C.** A tarefa 1 prova o ambiente sem expor o repositório: clone limpo, instalação, carga do artefato commitado e abertura da tela, e um deploy de verdade no Streamlit Community Cloud a partir de um repositório **privado**. Depois disso as duas frentes andam em paralelo, e a publicação fica para o fim, com o README pronto: passa a ser só tornar o repositório público e cadastrar os *secrets* do LLM.

---

## 5. O que será construído

### 5.1 Ordem

1. **Prova de clone limpo** (seção 5.2). Sequencial, antes de tudo.
2. **Frente de código**, numa worktree: briefing (5.3), tela (5.4), API (5.5). O briefing vem antes do botão da tela; a API é independente dos dois.
3. **Frente de documentos**, numa worktree, em paralelo à de código: gerador de tabelas e teste (5.6), Data Card (5.7), README (5.8). Só lê `reports/`; não edita arquivos da frente de código.
4. **Integração:** juntar as frentes, rodar `make teste`, `make qualidade` e a tela de verdade.
5. **Publicação** (5.9).
6. **Encerramento** (5.10).

A comparação de provedores do LLM (5.3) depende de o autor obter as chaves. Ela é um ponto de parada que não bloqueia o resto: todo o código do briefing é testado com um cliente falso.

### 5.2 Prova de clone limpo

- **Refazer o artefato servido a partir da árvore limpa.** O atual foi gravado com `receita_commit = "receita-congelada-13-gb5545fe4f41c-dirty"`, que apareceria no painel da demo. Com a árvore limpa, rodar `make servido` (permitido: o teste final já está commitado) e conferir que o novo artefato dá o mesmo `Resultado` que o antigo, exceto `model_metadata.receita_commit`, numa grade fixa de contextos (todas as combinações de dia e faixa para alguns perfis). Se diferir em qualquer número, a tarefa para e volta ao autor.
- Acrescentar ao `.gitignore` a exceção que permite commitar `models/servido/` (só ele; `avaliado` e `baseline` continuam fora) e a regra para `.DS_Store`.
- Commitar o artefato servido.
- Clonar o repositório numa pasta temporária, sem `data/`, rodar `uv sync` e `make app`, e confirmar que a tela calcula um resultado. Nada na tela pode exigir `data/` ou `reports/` além do que está commitado.
- Conferir na documentação atual do Streamlit Community Cloud como ele resolve as dependências (`uv.lock`, `pyproject.toml` ou `requirements.txt`), como fixar o Python 3.12 e qual o limite de memória.
- **Deploy de verdade, em repositório privado.** O autor cria o repositório privado no GitHub e o app no Streamlit Community Cloud (exigem o login dele); a tarefa envia a `main` e confere na URL do app que a tela calcula um resultado. Riscos que este passo existe para pegar: o pacote `occasion` precisa ser instalado (layout `src/` com hatchling); as versões de numpy, pandas e scikit-learn precisam ser as do `uv.lock`, ou o joblib do artefato pode não carregar; o lightgbm é importado por `models/arvores.py` e pode exigir `libgomp` no sistema; o conjunto precisa caber na memória. Registrar o que se apurou no ledger. Se algo impedir o deploy, a tarefa para e volta ao autor antes das frentes.
- Medir o tempo de um `prever` para dimensionar o mapa de calor (35 chamadas).

### 5.3 Briefing por LLM

Vale a seção 14.6 do design pai. O que este documento acrescenta:

**Módulo:** `src/occasion/briefing/`, com o montador do prompt, o cliente, a conferência, o texto reserva, o cache e o limite por sessão separados, para serem testados um a um.

**Entrada:** o `Contexto` informado na tela (UF, situação, dia, faixa, local, idade, sexo, faixa de renda, moradores) e o `Resultado` do contrato (score, classificação, componentes, fatores, categoria, justificativa e o aviso dos metadados). O contexto entra para que o briefing possa situar a ativação ("sábado à noite, em casa, em SP"); ele é a pergunta do usuário, não um fator. Nenhum microdado sai da máquina.

**Cliente:** a biblioteca `openai` apontada para o endereço do provedor, ou `httpx` direto **(escolha do redator: decidir consultando a documentação pelo `context7`; a preferência é a biblioteca que menos dependências acrescenta ao `uv.lock`)**. Endereço, modelo e chave vêm de variáveis de ambiente. Localmente, do `.env` (já no `.gitignore`); na demo, dos *secrets* do Streamlit Cloud. Tempo limite de cerca de 15 segundos.

**Idioma do briefing:** português, como a tela.

**Conferência automática**, depois de cada resposta:

- Todo número citado no texto existe na entrada, aceitando as formas equivalentes (vírgula ou ponto decimal, fração ou porcentagem, arredondamento para a casa exibida).
- O texto não contém termos proibidos. A lista fica num arquivo do módulo e cobre pelo menos alegações de efeito ("vende mais", "aumenta as vendas", "causa", "garante", "impacto", "eficácia") e nomes de marca ("Antarctica", "Coca-Cola" e afins).
- Reprovado, o texto é descartado e entra o texto reserva.

**Limite declarado da conferência** **(escolha do redator)**: ela não verifica automaticamente se cada fator citado está em `top_factors`. Barrar o identificador da variável quase nunca pega nada, e barrar o rótulo em português ("renda", "horário") reprovaria textos fiéis, porque o contexto também usa esses rótulos. A garantia sobre os fatores vem do prompt (só os fatores recebidos, com a direção recebida) e da conferência de números; o limite é dito no README.

**Texto reserva:** montado por modelo fixo a partir do `Resultado`. Entra sem chave, sem rede, com erro, com tempo esgotado, com a cota do provedor esgotada (erro 429), com a conferência reprovada ou com o limite da sessão atingido.

**Proteções da demo pública:**

- **Cache por decisão:** a chave é um hash do `Resultado` (sem campos que variam a cada chamada). A mesma decisão reaproveita o mesmo briefing e não chama o provedor de novo.
- **Limite por sessão:** até 5 chamadas ao provedor por sessão do Streamlit **(escolha do redator para o número)**. Depois disso, texto reserva com um aviso.
- **Plano gratuito do provedor:** a chave da demo é de plano gratuito, sem cartão. Cota esgotada vira erro, não cobrança.

**Transparência na tela:** o briefing mostra de onde veio ("gerado por <modelo> e conferido" ou "texto reserva: <motivo>").

**Comparação de provedores (ponto de parada com o autor):** um script manda o mesmo prompt, para um conjunto fixo de cerca de 10 contextos, ao Groq e ao NVIDIA NIM, e grava latência, taxa de aprovação na conferência e os textos em `reports/briefing_provedores.json`. Os limites do plano gratuito de cada um são lidos da documentação atual e anotados. O autor escolhe o provedor e o modelo com esse resultado. A preferência inicial do redator é o Groq, pela latência, mas a decisão é do teste.

**Testes (antes do código):** a conferência reprova texto com número inventado e com termo proibido, e aprova texto fiel; o texto reserva aparece sem chave, com erro do cliente, com 429 e com tempo esgotado; o cache não chama o cliente na segunda vez; o limite por sessão corta na sexta chamada. Tudo com cliente falso, sem rede.

### 5.4 Tela

A tela continua em português e em uma página. O que muda:

**Mapa de calor**, abaixo do resultado:

- Linhas: os sete dias da semana; colunas: as cinco faixas de horário. Cada célula é o `occasion_score` (0 a 100) de `prever` com aquele dia e faixa, mantidos os outros campos do formulário.
- A célula do contexto atual fica destacada.
- Escala sequencial de uma cor, com o valor escrito em cada célula. A skill `dataviz` é carregada antes de escrever o gráfico.
- Legenda curta: "posição de cada combinação entre os contextos da população de referência, com os demais campos fixos".
- Calculado com cache sobre o contexto sem dia e faixa, para não refazer as 35 chamadas a cada interação.

**Briefing**, abaixo da recomendação: um botão "Gerar briefing", o texto e a linha de transparência (5.3).

**Painel de metadados:** já existe e já traz o veredito de H8. O texto não é lido do JSON quando a tela roda: `make servido` o monta a partir de `reports/teste_final.json` e o grava no artefato (`inference/servido.py`, `_descricao`), e a tela mostra `model_metadata.descricao_metricas`. Como o artefato é refeito na tarefa 1 e o JSON do teste não muda, o texto continua fiel. Fica como está, salvo ajustes que a revisão pedir.

**Verificação:** a tela é aberta e usada de verdade (skill `run` ou Playwright, seguindo as armadilhas do `CLAUDE.md`), com e sem chave do LLM.

### 5.5 API

- **Módulo:** `src/occasion/api/`, com FastAPI e Pydantic usando os tipos de `inference/contract.py`.
- **Rotas:** `POST /predict` (recebe `Contexto`, devolve `Resultado`) e `GET /health` (situação e metadados do modelo carregado).
- **Comando:** `make api`, com Uvicorn.
- **Testes (antes do código),** com o cliente de teste do FastAPI: os campos obrigatórios com os tipos certos; o mesmo resultado que `prever` para o mesmo contexto; idade abaixo de 18 e UF inválida devolvem 422.
- O artefato é carregado uma vez, na subida do serviço.

### 5.6 Números da documentação gerados por script

- **Script:** `scripts/tabelas_docs.py` lê os JSON de `reports/` e escreve tabelas em Markdown entre marcadores (`<!-- BEGIN:nome -->` e `<!-- END:nome -->`) no README e no Data Card. Os rótulos são traduzidos para o inglês no script; os JSON continuam como estão.
- **Comando:** `make docs`.
- **Contagens de referência:** hoje elas só existem como constantes dos testes (`tests/bases/test_ocasioes.py`), e nenhum JSON as traz. Um comando novo, `make contagens` **(escolha do redator)**, lê as bases e grava `reports/contagens.json` com: dias-pessoa, dias com pizza, dias com pizza e refrigerante, dias com pizza e guaraná e ocasiões com refrigerante por edição, e os dias-pessoa por dia da semana nas duas edições. O alvo de 2017-2018 é lido pela porta `carregar_base` com `exigir_teste_commitado` (o teste final já está commitado). O código fica em `scripts/` ou num módulo novo fora das pastas assinadas.
- **Tabelas previstas** **(escolha do redator)**: contagens de referência (de `reports/contagens.json`); resultado de H1 a H10 (veredito e número principal de cada uma); teste final (PR-AUC do score e do degrau A, ganho e intervalo de 95%, por parte); métricas da receita em 2008-2009; segmentos principais do teste final.
- **Teste:** `tests/test_docs_atualizados.py` regenera as tabelas em memória e falha se o conteúdo entre marcadores diferir do arquivo.
- **Regra do texto corrido:** todo número decimal ou porcentagem fora dos blocos gerados tem de aparecer dentro de algum bloco gerado do mesmo arquivo. O mesmo teste confere isso. Números inteiros de contexto (anos, contagem de hipóteses, duração do vídeo) ficam fora da regra.

### 5.7 Data Card (em inglês)

- **Arquivo:** `docs/DATA_CARD.md`.
- **Conteúdo:** os itens do TAPI (seção 3), mais as armadilhas da fonte que já estão no `CLAUDE.md` e no design pai: sabor e não marca; sem data de calendário; sem canal; dias da semana desbalanceados, inclusive a explicação do excesso de terças e quintas em 2008-2009; método de coleta diferente entre as edições; renda em salários mínimos; local de 2008-2009 como fonte do alimento.
- **Unidade e target:** pessoa × dia × célula na parte 1, ocasião com refrigerante na parte 2, com a definição exata do evento e as regras R12 e R13.
- **Estratégia contra vazamento:** separação temporal, a trava de leitura do alvo de 2017-2018 (`carregar_base`), a tag e a assinatura da receita.
- **Licença e LGPD:** microdados públicos e anonimizados do IBGE, sem tentativa de reidentificação, nenhum microdado enviado ao LLM.
- Os números vêm dos blocos gerados (5.6).

### 5.8 README (em inglês)

Escrito para quem abre o repositório sem contexto. Ordem **(escolha do redator)**:

1. **Título e um parágrafo:** o problema, a pergunta ("in which contexts guaraná beats cola, and which pizza contexts look like them"), o link da demo e o resultado principal em uma linha, inclusive H8.
2. **Try it:** link da demo; `uv sync` e `make app` a partir do clone, com o artefato commitado.
3. **The question and why it is framed this way:** por que não prever onde a ocasião já acontece; o score em duas partes.
4. **Architecture:** diagrama Mermaid das camadas (ingestão → bases → variáveis → modelos → score → decisão → inferência → tela, API e briefing) e onde cada uma mora no código.
5. **Data and target:** POF/IBGE, unidade de análise, definição do evento, link para o Data Card.
6. **Method:** validação temporal, pré-registro, escada de modelos e regra de escolha, receita congelada por tag e assinatura, modelo avaliado e modelo servido. Cada escolha com a referência da literatura que a sustenta, extraída do registro de decisões. **Toda referência citada no README é conferida antes** (autor, ano, título e link que existe): o registro de decisões marca nove referências como "citadas de memória, sem link conferido", e nenhuma delas vai para o README sem conferência. Uma referência que não se confirma sai do README.
7. **Results:** os blocos gerados (5.6).
8. **What the negative result means:** a leitura honesta de H8 (seção 1).
9. **Decision Engine:** as duas camadas, os cortes e a regra rural declarada como suposição.
10. **Explainability e LLM briefing:** como os fatores são calculados; as travas do briefing.
11. **Limitations.**
12. **Reproduce everything:** os comandos do `Makefile` em ordem, com o tempo aproximado de cada um, inclusive as horas da escada.
13. **Repository map.**
14. **Next steps:** POF 2024-2025 como segunda rodada de teste (e o teste do modelo servido e da comparação entre C e D), "mesa", coletor de cardápios, clima nas capitais em 2008-2009.
15. **How this was built:** uma linha dizendo que o projeto foi construído com Claude Code, sob o processo descrito em `docs/` (design, pré-registro, planos, revisão e registro de decisões), com as decisões tomadas pelo autor.
16. **A note on language:** o código e parte dos documentos estão em português; os identificadores dos arquivos assinados não são traduzidos para não quebrar a assinatura da receita.
17. **License.**

**Licença do código:** MIT **(escolha do redator)**. Os microdados seguem os termos do IBGE e não são redistribuídos no repositório.

### 5.9 Publicação

Ações externas: cada uma é confirmada com o autor na hora.

1. Antes de abrir: procurar segredos no histórico do git (chaves, `.env`), conferir o `.gitignore` e o tamanho do repositório.
2. Enviar a `main` final e a tag `receita-congelada` ao repositório criado na tarefa 1, e o autor o torna público.
3. O autor cadastra os *secrets* do LLM no app do Streamlit Community Cloud. O plano traz o passo a passo.
4. Abrir a URL pública e usar de verdade: um cálculo, o mapa de calor, um briefing pelo LLM e um briefing com o limite da sessão atingido.
5. Colocar o link da demo no README e regenerar o que for preciso.

### 5.10 Encerramento

- Revisão de código (skill `superpowers:requesting-code-review`) com o checklist de método do `CLAUDE.md`, mais: nenhum arquivo assinado mudou (assinatura igual à da tag), versões das bibliotecas assinadas iguais, nenhum segredo no repositório.
- `make teste`, `make qualidade` e o teste dos documentos passando.
- Atualizar o `CLAUDE.md` com os comandos novos e o que o Plano 3 ensinou (`/revise-claude-md`).
- Registrar as decisões do Plano 3 numa seção 28 do registro de decisões.

---

## 6. Testes

| Camada | O que é testado |
|---|---|
| Briefing | Conferência (número inventado, termo proibido, texto fiel); texto reserva em cada falha; cache; limite por sessão. Cliente falso, sem rede |
| API | Contrato de saída; igualdade com `prever`; 422 para entrada inválida |
| Tela | Funções puras do mapa de calor (montagem da grade de contextos, 35 células, destaque do contexto atual); a tela aberta de verdade |
| Documentos | Blocos gerados iguais ao que o script produz; números do texto corrido presentes nos blocos |
| Receita | Assinatura igual à da tag; versões das bibliotecas assinadas iguais às de `versoes_da_receita.py` |

Proibido teste que afirma limiar de métrica ajustado para passar.

---

## 7. Escolhas feitas pelo redator

1. Módulos novos em `src/occasion/briefing/` e `src/occasion/api/`.
2. API sem endpoint de briefing e sem publicação na internet.
3. Limite de 5 chamadas ao LLM por sessão; tempo limite de cerca de 15 segundos.
4. Cache do briefing pela decisão estruturada.
5. Cliente do LLM decidido pela documentação, preferindo o que menos mexe no `uv.lock`.
6. Lista mínima de termos proibidos (seção 5.3).
7. Briefing em português.
8. Tabelas geradas previstas na seção 5.6 e a regra dos números do texto corrido.
9. Ordem das seções do README.
10. Licença MIT para o código.
11. Só o artefato servido vai para o repositório, refeito a partir da árvore limpa e conferido contra o anterior.
12. O `Contexto` entra no prompt do briefing junto com o `Resultado`.
13. A conferência não checa os fatores citados por nome; o limite é declarado.
14. `make contagens` e `reports/contagens.json` para as contagens de referência.

Decididas pelo autor nesta conversa: tudo publicado, inclusive `CLAUDE.md`, `docs/superpowers/` e os CSV do Trends, com a linha "How this was built" no README; H8 contada como ponto forte do método; README e Data Card em inglês; tela em português até o projeto de tradução; vídeo fora do Plano 3; repositório público com o TAPI; demo no Streamlit Community Cloud com o LLM chamado de verdade e protegido por cache, limite por sessão e plano gratuito; FastAPI mantida; provedor escolhido por comparação; organização B com a prova de clone limpo na frente.

---

## 8. O que fica para quem implementa

- Nomes das variáveis de ambiente do LLM e dos *secrets*.
- Texto do prompt e do modelo fixo do texto reserva.
- Normalização exata dos números na conferência.
- Os 10 contextos da comparação de provedores.
- Aparência do mapa de calor, seguindo a skill `dataviz`.
- Redação do README e do Data Card dentro da estrutura acima.
