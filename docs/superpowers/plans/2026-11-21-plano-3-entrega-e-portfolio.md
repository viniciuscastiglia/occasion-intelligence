# Plano 3: tela completa, briefing, API, documentação e demo pública

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** fechar a entrega do case e deixá-la pronta para portfólio: tela com mapa de calor e briefing por LLM conferido, API FastAPI, Data Card e README em inglês com números gerados dos JSON, e o repositório público com demo no Streamlit Community Cloud.

**Arquitetura:** dois pacotes novos fora das pastas assinadas: `occasion.briefing` (prompt, conferência, texto reserva, cliente HTTP compatível com a OpenAI, cache e limite por sessão) e `occasion.api` (FastAPI sobre `prever`). Mais `occasion.relatorios` (contagens de referência e tabelas da documentação) e `occasion.inference.grade` (grade do mapa de calor). A tela continua chamando só funções; não calcula nada. O artefato servido passa a ser commitado.

**Stack:** a dos Planos 1 e 2, mais `fastapi`, `uvicorn` e `httpx` (este já está no `uv.lock` como dependência transitiva e passa a ser explícito). O cliente do LLM é `httpx` chamando `POST {base_url}/chat/completions`, sem a biblioteca `openai`, para não acrescentar dependências.

**Spec:** `docs/superpowers/specs/2026-11-21-plano-3-entrega-e-portfolio-design.md` (aprovado em 21/11/2026). O design pai, `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md` (seções 14, 15 e 17), e o `CLAUDE.md` fazem parte da spec.

**Ponto de partida:** branch `plano-3`, criada da `main` com o design commitado (`b2456b1`). `data/processed/` e `reports/` completos na pasta principal. Ledger em `.superpowers/sdd/2026-11-21-plano-3-entrega-e-portfolio/progress.md`.

**Fora deste plano:** vídeo (última etapa da entrega, até 5 minutos) e tradução do resto do projeto para o inglês.

## Restrições globais

- **Receita congelada:** nenhum arquivo de `ARQUIVOS_DA_RECEITA` (`src/occasion/bases/assinatura.py`) muda. Conferir ao fim de cada tarefa: `uv run python -c "from occasion.bases.assinatura import exigir_receita_intacta; exigir_receita_intacta(); print('ok')"`.
- **Versões assinadas:** depois de qualquer `uv add` ou `uv lock`, rodar `uv run python -c "from occasion.experimentos.teste_final import _conferir_versoes; _conferir_versoes(); print('ok')"`. numpy, pandas, scipy, scikit-learn, lightgbm e pyarrow não mudam de versão; se mudarem, desfazer e fixar a versão.
- Código, nomes e comentários em português; **README, Data Card e `docs/REFERENCES.md` em inglês**; tela e briefing em português. Funções públicas com anotações de tipo. `make qualidade` sem erros. Formatar só `src/`, `tests/`, `app/`, `scripts/`.
- Antes de usar uma biblioteca (FastAPI, httpx, Streamlit, Altair), consultar a versão instalada pelo `context7`, mandando só o nome da biblioteca e a dúvida técnica. Carregar a skill `dataviz` antes do mapa de calor.
- **Números em texto** (README, Data Card, tela) só vêm de arquivos gerados pelo código. Proibido teste que afirma limiar de métrica.
- **Linguagem:** nenhuma alegação de efeito causal; "guaraná" é o sabor, nunca a marca; "pizza com refrigerante de guaraná" e não "Pizza + Guaraná Antarctica" nos textos de resultado.
- **LLM:** recebe só o `Contexto` e o `Resultado`; nenhum microdado sai da máquina. Variáveis de ambiente (e chaves de `st.secrets`): `BRIEFING_BASE_URL`, `BRIEFING_MODELO`, `BRIEFING_CHAVE`. Nenhuma chave commitada: `.env` e `.streamlit/secrets.toml` no `.gitignore`.
- **Alvo de 2017-2018:** só pela porta `carregar_base`, depois de `exigir_teste_commitado`.
- **Ações externas** (criar repositório, enviar código, mudar visibilidade, criar app no Streamlit Cloud): cada uma confirmada com o autor na hora.
- **Laços de espera:** nunca `while pgrep -f`; processos em segundo plano com `timeout` e encerrados no fim da tarefa (conferir `ps`).
- Um commit por tarefa (ou por passo marcado), em português, terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Foco de revisão

1. **Briefing fiel reprovado por formato de número:** o LLM escreve o score 87,3 como "87" e a probabilidade 0,0031 como "0,31%". A conferência aceita essas formas e só reprova número que não existe na entrada. Teste na tarefa 3.
2. **Tela sem `secrets.toml` nem variáveis de ambiente** (máquina do avaliador, clone limpo): `st.secrets` lança exceção quando o arquivo não existe. A tela não quebra e mostra o texto reserva com o motivo "sem chave configurada". Teste na tarefa 6.
3. **Contexto mudado sem recalcular:** o mapa de calor e o briefing usam o contexto do último cálculo (o do resultado exibido), nunca o formulário atual. Teste na tarefa 6.
4. **Resposta do provedor vazia, truncada ou fora do formato** (`choices` ausente, JSON inválido, HTTP 500): vira texto reserva com motivo, sem exceção na tela. Teste na tarefa 4.
5. **Números de versão e caminhos no README** ("Python 3.12", "scikit-learn 1.9") confundidos com resultado pela regra dos números: o que está entre crases, em bloco de código ou em URL fica fora da regra. Teste na tarefa 8.

---

## Estrutura de arquivos

```
src/occasion/
  briefing/__init__.py
  briefing/prompt.py         entrada_do_briefing, montar_mensagens
  briefing/conferencia.py    TERMOS_PROIBIDOS, Conferencia, conferir
  briefing/reserva.py        texto_reserva
  briefing/cliente.py        ConfiguracaoLLM, configuracao_de, ClienteLLM, ClienteHTTP, FalhaDoLLM
  briefing/servico.py        Briefing, LIMITE_POR_SESSAO, chave_do_cache, gerar_briefing
  api/__init__.py
  api/app.py                 criar_app, app (POST /predict, GET /health)
  inference/grade.py         grade_do_mapa, tabela_do_mapa
  relatorios/__init__.py
  relatorios/contagens.py    contar, main  (reports/contagens.json)
  relatorios/docs.py         gerar_blocos, aplicar_blocos, numeros_fora_dos_blocos, main
app/streamlit_app.py         + mapa de calor e briefing
scripts/comparar_artefatos.py   antigo × novo numa grade de contextos (tarefa 1)
scripts/comparar_provedores.py  Groq × NVIDIA NIM (tarefa 5)
docs/DATA_CARD.md  docs/REFERENCES.md  README.md  LICENSE  .env.example
tests/briefing/  tests/api/  tests/inference/test_grade.py  tests/relatorios/
```

---

### Tarefa 1: artefato servido limpo, clone limpo e deploy privado (ponto de parada com o autor)

**Arquivos:**
- Modificar: `.gitignore` (`/models/` vira `/models/*` + `!/models/servido/`; acrescentar `.DS_Store` e `.streamlit/secrets.toml`)
- Criar: `scripts/comparar_artefatos.py`; `models/servido/artefato.joblib` (commitado); conforme a documentação do Cloud, `requirements.txt` e/ou `packages.txt`
- Ledger: o que se apurou do Streamlit Cloud e o tempo de um `prever`

**Interfaces:**
- Produz: o artefato servido commitado; a URL do app privado; a forma de instalar dependências no Cloud (usada nas tarefas 4 e 7, que mexem no `pyproject.toml`)

- [ ] **Passo 1: guardar o artefato atual.** `cp models/servido/artefato.joblib data/experimentos/servido_antigo.joblib`.
- [ ] **Passo 2: escrever `scripts/comparar_artefatos.py`** com `comparar(antigo: ArtefatoModelo, novo: ArtefatoModelo) -> list[str]`. A grade: os 35 pares dia × faixa, cruzados com 4 perfis fixos (SP urbano em casa 28 anos F de 2 a 5 SM 3 moradores; BA rural fora 45 M até meio SM 5; RS urbano fora 22 F mais de 5 SM 1; AM urbano em casa 60 M de 1 a 2 SM 4). Devolve a descrição de cada campo de `Resultado.model_dump()` que difere, ignorando `model_metadata.receita_commit`. Floats comparados com igualdade exata. `main` imprime as diferenças e sai com código 1 se houver alguma.
- [ ] **Passo 3: refazer o servido a partir da árvore limpa.** `git status --porcelain` vazio (exceto `.DS_Store`), depois `make servido`. Esperado: `treinado em: ['2008-2009', '2017-2018']`.
- [ ] **Passo 4: comparar.** `uv run python scripts/comparar_artefatos.py data/experimentos/servido_antigo.joblib models/servido/artefato.joblib`. Esperado: nenhuma diferença, saída 0. Com diferença, parar e levar ao autor. Conferir também que `receita_commit` não termina em `-dirty`.
- [ ] **Passo 5: medir.** Tempo médio de `prever` em 35 chamadas; anotar no ledger.
- [ ] **Passo 6: commitar** `.gitignore`, o script e o artefato.
- [ ] **Passo 7: clone limpo.** `git clone` da pasta para o scratchpad, `uv sync --frozen`, e sem `data/`: `uv run python -c` carregando `carregar_artefato()` e chamando `prever` num contexto. Depois `timeout 60 uv run streamlit run app/streamlit_app.py --server.headless true --server.port 8599`, em segundo plano, e `curl -sf localhost:8599/_stcore/health`. Esperado: `ok`. Encerrar o processo.
- [ ] **Passo 8: documentação do Cloud.** Pelo `context7` ou pela documentação oficial: como o Community Cloud resolve dependências (`uv.lock`, `pyproject.toml`, `requirements.txt`), se instala o próprio pacote (layout `src/`, hatchling), como fixar o Python 3.12, `packages.txt` para bibliotecas do sistema (o lightgbm pode exigir `libgomp1`), limite de memória e se aceita repositório privado. Criar os arquivos que a documentação exigir (por exemplo, `uv export --no-dev --frozen` para `requirements.txt`, com o pacote local) e commitar.
- [ ] **Passo 9 (ponto de parada): repositório privado.** Propor ao autor o nome (`occasion-intelligence`) e, com a confirmação dele, rodar `gh repo create <nome> --private --source . --remote origin`, depois `git push -u origin main plano-3` e `git push origin receita-congelada`.
- [ ] **Passo 10 (autor): app no Cloud.** O autor cria o app pela interface com: repositório `<nome>`, branch `plano-3`, arquivo `app/streamlit_app.py`, Python 3.12. Esperar a URL.
- [ ] **Passo 11: verificar ao vivo.** Pelo Playwright, abrir a URL, clicar em Calcular (pelo rótulo dentro de `stSidebarUserContent`) e conferir o score e a recomendação. Se o deploy falhar, ler o log do app com o autor, corrigir o que for de ambiente (arquivos de dependência) e repetir. Se precisar mexer em arquivo assinado, parar.
- [ ] **Passo 12:** registrar no ledger o que funcionou, a URL e as limitações (memória, tempo de subida).

### Tarefa 2: contagens de referência em JSON

**Arquivos:**
- Criar: `src/occasion/relatorios/__init__.py`, `src/occasion/relatorios/contagens.py`, `tests/relatorios/test_contagens.py`
- Modificar: `Makefile` (alvo `contagens`)
- Gera: `reports/contagens.json` (commitado)

**Interfaces:**
- Consome: `carregar_base(nome, edicao)` e `exigir_teste_commitado()` de `occasion.bases.acesso`
- Produz: `contar(celulas: pd.DataFrame, ocasioes_refri: pd.DataFrame) -> dict[str, Any]`, com as chaves `dias_pessoa`, `celulas`, `dias_pizza_refri`, `dias_pizza_guarana`, `ocasioes_refri` e `dias_por_dia_semana` (dict de `"1"` a `"7"` para contagem bruta de dias-pessoa). `reports/contagens.json` tem `{"2008-2009": {...}, "2017-2018": {...}, "nota": str}`.

As contagens são das bases de modelagem, depois das exclusões. Em 2017-2018 saem 796 dias com pizza e refrigerante, e não os 797 da tabela do `CLAUDE.md`. A `nota` diz isso. "Dias com pizza" (sem refrigerante) não entra: exigiria ler `pessoa_dia` de 2017-2018 fora de `carregar_base`, porta que o registro de decisões descartou.

- [ ] **Passo 1: teste** `test_contar_em_base_sintetica`: duas pessoas-dia com 10 células cada, uma célula positiva em `y_pizza_refri` e em `y_pizza_guarana` no mesmo `id_dia`, três ocasiões com refrigerante. Afirma `dias_pessoa == 2`, `celulas == 20`, `dias_pizza_refri == 1`, `dias_pizza_guarana == 1`, `ocasioes_refri == 3`, e a soma de `dias_por_dia_semana` igual a 2.
- [ ] **Passo 2: teste com marca `dados`** `test_contagens_de_2008_batem_com_a_referencia`: 66.903 / 669.030 / 651 / 101 / 19.182.
- [ ] **Passo 3:** rodar e ver falhar; implementar `contar` (dias pelo `id_dia` único) e `main()` (`exigir_teste_commitado()`, as duas edições, grava o JSON com `ensure_ascii=False, indent=2`); alvo `contagens: uv run python -m occasion.relatorios.contagens`.
- [ ] **Passo 4:** `make contagens`. Esperado em 2017-2018: 85.018 dias-pessoa, 850.180 células, 796, entre 170 e 180, 16.112. Rodar `make teste-dados -k contagens`.
- [ ] **Passo 5: commit** (código, teste, Makefile, JSON).

### Tarefa 3: briefing: entrada, conferência e texto reserva

**Arquivos:**
- Criar: `src/occasion/briefing/{__init__,prompt,conferencia,reserva}.py`, `tests/briefing/test_conferencia.py`, `tests/briefing/test_reserva.py`, `tests/briefing/test_prompt.py`

**Interfaces:**
- Consome: `Contexto`, `Resultado` de `occasion.inference.contract`; `rotulos` de `occasion.rotulos`
- Produz:
  - `entrada_do_briefing(contexto: Contexto, resultado: Resultado) -> dict[str, Any]`: `{"contexto": contexto.model_dump(), "resultado": {...}}`, com todos os campos de `Resultado`, exceto `model_metadata`, do qual só entra `aviso`.
  - `montar_mensagens(contexto: Contexto, resultado: Resultado) -> list[dict[str, str]]`: mensagem `system` com as regras e mensagem `user` com o JSON da entrada.
  - `TERMOS_PROIBIDOS: tuple[str, ...]`; `@dataclass(frozen=True) Conferencia(aprovado: bool, motivos: tuple[str, ...])`; `conferir(texto: str, entrada: dict[str, Any]) -> Conferencia`.
  - `texto_reserva(contexto: Contexto, resultado: Resultado) -> str`.

**Regras do prompt** (texto livre; estas instruções são obrigatórias):
- Briefing de 3 a 5 frases para uma pessoa de marketing, em português.
- Só os fatores, números e categoria recebidos, com a direção recebida.
- Não afirmar efeito sobre vendas.
- Guaraná é o sabor registrado na POF, não uma marca.
- Não citar marcas.

**Conferência:**
- Os números do texto são extraídos com vírgula ou ponto decimal e `%` opcional.
- Um número `t` com `d` casas decimais é aceito se algum número `x` da entrada (percorrida recursivamente, inclusive os números dentro de strings como `recommendation_reason`), ou `100·x` quando `0 ≤ x ≤ 1`, arredondado para `d` casas, for igual a `t`.
- Os termos são comparados sem acento e sem diferença de maiúsculas.
- `TERMOS_PROIBIDOS` cobre pelo menos: `vende mais`, `aumenta as vendas`, `aumentar as vendas`, `causa`, `garante`, `garantir`, `impacto`, `eficácia`, `antarctica`, `coca-cola`, `coca cola`, `pepsi`, `kuat`.

- [ ] **Passo 1: testes da conferência**, com um `Resultado` fixo (score 87,3; probabilidade conjunta 0,0031; componentes 92,4 e 31,0; idade 28):
  - `test_aceita_formas_equivalentes`: "score 87", "87,3", "0,31%", "posição 92" aprovados;
  - `test_reprova_numero_inventado`: "45%" reprovado, com o motivo citando 45;
  - `test_reprova_termo_proibido`: "vende mais" e "Guaraná Antarctica" reprovados, também em maiúsculas e sem acento;
  - `test_aprova_texto_sem_numeros`.
- [ ] **Passo 2: testes do texto reserva e do prompt:**
  - `test_texto_reserva_passa_na_conferencia`, parametrizado com três resultados: um de cada objetivo e um "Não priorizar";
  - `test_texto_reserva_cita_categoria_e_fatores`;
  - `test_entrada_nao_leva_metadados_alem_do_aviso`;
  - `test_mensagens_tem_system_e_user_com_json_da_entrada`.
- [ ] **Passo 3:** rodar e ver falhar; implementar; rodar e ver passar. `make qualidade`.
- [ ] **Passo 4: commit.**

### Tarefa 4: briefing: cliente HTTP, cache, limite por sessão e serviço

**Arquivos:**
- Criar: `src/occasion/briefing/{cliente,servico}.py`, `tests/briefing/test_cliente.py`, `tests/briefing/test_servico.py`, `.env.example`
- Modificar: `pyproject.toml` e `uv.lock` (`httpx` explícito); arquivo de dependências do Cloud, se a tarefa 1 criou um

**Interfaces:**
- Consome: tarefa 3.
- Produz:
  - `@dataclass(frozen=True) ConfiguracaoLLM(base_url: str, modelo: str, chave: str)`; `configuracao_de(fonte: Mapping[str, str]) -> ConfiguracaoLLM | None` (devolve `None` se faltar algum dos três `BRIEFING_*`).
  - `class FalhaDoLLM(Exception)`, com o atributo `motivo: Literal["tempo_esgotado", "cota_esgotada", "resposta_invalida", "erro_do_provedor"]`.
  - `class ClienteLLM(Protocol)`, com `modelo: str` e `completar(mensagens: list[dict[str, str]]) -> str`.
  - `class ClienteHTTP`, criado com `ClienteHTTP(config: ConfiguracaoLLM, tempo_limite: float = 15.0, transporte: httpx.BaseTransport | None = None)`. Faz `POST {base_url}/chat/completions` com `Authorization: Bearer`, `temperature` 0,2 e `max_tokens` 400. Os erros são mapeados assim: 429 vira `cota_esgotada`; timeout vira `tempo_esgotado`; JSON sem `choices[0].message.content`, ou com conteúdo vazio, vira `resposta_invalida`; o resto vira `erro_do_provedor`.
  - `@dataclass(frozen=True) Briefing(texto: str, fonte: Literal["llm", "reserva"], modelo: str | None, motivo: str | None)`.
  - `LIMITE_POR_SESSAO = 5`; `TAMANHO_MAXIMO_DO_CACHE = 1000`.
  - `chave_do_cache(contexto: Contexto, resultado: Resultado) -> str`: SHA-256 do JSON de `entrada_do_briefing` com `sort_keys=True`.
  - `gerar_briefing(contexto, resultado, cliente: ClienteLLM | None, cache: MutableMapping[str, Briefing], chamadas_na_sessao: int) -> tuple[Briefing, bool]`. O `bool` diz se o provedor foi chamado. A ordem é: cache; sem cliente vira reserva com "sem chave configurada"; limite atingido vira reserva com "limite de briefings desta sessão atingido"; chamada; conferência. Só texto aprovado vai para o cache. Quando o cache passa do tamanho máximo, sai a entrada mais antiga.
- Motivos exibidos (português): "sem chave configurada", "limite de briefings desta sessão atingido", "tempo esgotado no provedor", "cota do provedor esgotada", "resposta inválida do provedor", "erro do provedor", "texto reprovado na conferência: <motivos>".

- [ ] **Passo 1:** consultar o `context7` sobre o `httpx` (`MockTransport`, timeout) e a documentação do formato `chat/completions` dos provedores compatíveis com a OpenAI.
- [ ] **Passo 2: testes do cliente**, com `httpx.MockTransport`:
  - `test_cliente_devolve_conteudo`;
  - `test_429_vira_cota_esgotada`;
  - `test_timeout_vira_tempo_esgotado`;
  - `test_json_sem_choices_vira_resposta_invalida`;
  - `test_conteudo_vazio_vira_resposta_invalida`;
  - `test_500_vira_erro_do_provedor`;
  - `test_configuracao_de_exige_os_tres_campos`.
- [ ] **Passo 3: testes do serviço**, com um cliente falso que conta as chamadas:
  - `test_sem_cliente_usa_reserva_com_motivo`;
  - `test_texto_aprovado_vem_do_llm_e_entra_no_cache`;
  - `test_segunda_chamada_igual_nao_chama_o_provedor`;
  - `test_limite_corta_a_sexta_chamada` (com `chamadas_na_sessao=5`, não chama e devolve o motivo do limite);
  - `test_cache_atende_mesmo_com_limite_atingido`;
  - `test_texto_reprovado_vira_reserva_e_nao_entra_no_cache`;
  - `test_falha_do_llm_vira_reserva_com_motivo`, parametrizado nos quatro motivos.
- [ ] **Passo 4:** rodar e ver falhar; `uv add httpx`; conferir as versões assinadas (restrição global); implementar; rodar e ver passar.
- [ ] **Passo 5:** criar `.env.example` com os três `BRIEFING_*` vazios e um comentário. Atualizar o arquivo de dependências do Cloud, se houver. `make qualidade`.
- [ ] **Passo 6: commit.**

### Tarefa 5: comparação de provedores (ponto de parada com o autor)

**Arquivos:**
- Criar: `scripts/comparar_provedores.py`
- Gera: `reports/briefing_provedores.json` (commitado)

**Interfaces:**
- Consome: `ClienteHTTP`, `ConfiguracaoLLM`, `montar_mensagens`, `conferir`, `entrada_do_briefing`, `carregar_artefato`, `prever`.
- Produz: `BRIEFING_BASE_URL` e `BRIEFING_MODELO` escolhidos pelo autor, anotados em `.env.example` e no ledger.

- [ ] **Passo 1:** na documentação atual de cada provedor, anotar no ledger o endereço compatível com a OpenAI (Groq: `https://api.groq.com/openai/v1`; NVIDIA NIM: `https://integrate.api.nvidia.com/v1`; conferir os dois), um ou dois modelos instrucionais de cada um e os limites do plano gratuito (requisições por minuto e por dia, tokens).
- [ ] **Passo 2:** o script lê `GROQ_API_KEY` e `NVIDIA_API_KEY` do ambiente e roda 10 contextos fixos (variando UF, dia, faixa, local e os quatro objetivos) em cada par provedor × modelo, uma chamada por vez. Grava no JSON, por par: latência mediana e máxima, aprovados na conferência, motivos das reprovações e os textos. Nenhum microdado entra no script.
- [ ] **Passo 3 (ponto de parada):** pedir ao autor as chaves, que ele mesmo coloca no `.env`, nunca no chat. Rodar com `set -a; . ./.env; set +a; uv run python scripts/comparar_provedores.py`.
- [ ] **Passo 4:** levar ao autor uma tabela curta (latência, aprovação e um texto de exemplo de cada par). Ele escolhe provedor e modelo, e o autor põe a chave escolhida em `.streamlit/secrets.toml` (no `.gitignore`).
- [ ] **Passo 5: commit** do script, do JSON e do `.env.example` atualizado.

Esta tarefa não bloqueia a 6 nem a 7: elas são testadas sem chave.

### Tarefa 6: tela: mapa de calor e botão de briefing

**Arquivos:**
- Criar: `src/occasion/inference/grade.py`, `tests/inference/test_grade.py`
- Modificar: `app/streamlit_app.py`, `tests/app/test_app.py`

**Interfaces:**
- Consome: `prever`, `ArtefatoModelo`, `Contexto`; `gerar_briefing`, `configuracao_de`, `ClienteHTTP`, `Briefing`, `LIMITE_POR_SESSAO`.
- Produz:
  - `grade_do_mapa(contexto: Contexto) -> list[Contexto]`: 35 contextos na ordem de `get_args(DiaSemana)` × `get_args(FaixaHorario)`, mudando só `dia_semana` e `faixa_horario`.
  - `tabela_do_mapa(contexto: Contexto, artefato: ArtefatoModelo) -> pd.DataFrame`: colunas `dia_semana`, `faixa_horario`, `score` e `atual` (bool, verdadeiro só na célula do contexto).

**Mapa de calor na tela:**
- Gráfico Altair `mark_rect`, com dia nas linhas e faixa nas colunas, ambos com rótulos de `rotulos.py`. Escala sequencial de uma cor, de 0 a 100, com o valor escrito em cada célula e a célula atual com contorno.
- Legenda: "Posição de cada combinação entre os contextos da população de referência, com os demais campos fixos."
- Cache por `st.cache_data`, com chave no JSON do contexto com dia e faixa normalizados.
- Usa `st.session_state["contexto_usado"]`.

**Briefing na tela:**
- Botão "Gerar briefing" abaixo da recomendação.
- A configuração vem de `os.environ` combinado com `st.secrets`. Se `st.secrets` não tiver arquivo, a exceção é tratada como fonte vazia.
- O cache vem de `st.cache_resource` (um `dict` do processo, compartilhado entre sessões). O contador fica em `st.session_state["chamadas_llm"]`.
- O briefing é guardado na sessão junto com o resultado e some quando o resultado é recalculado.
- Embaixo do texto, uma legenda: "Gerado por <modelo> e conferido automaticamente" ou "Texto reserva: <motivo>".

- [ ] **Passo 1:** carregar a skill `dataviz` e consultar o `context7` sobre Altair (`mark_rect`, `mark_text` em camadas) e Streamlit (`st.secrets` sem arquivo, `AppTest` com `session_state`).
- [ ] **Passo 2: testes da grade**, com `_artefato_unico()` de `tests/inference/test_predict.py`:
  - `test_grade_tem_35_contextos_e_so_muda_dia_e_faixa`;
  - `test_tabela_marca_so_o_contexto_atual`;
  - `test_score_da_tabela_igual_ao_de_prever`.
- [ ] **Passo 3: testes da tela**, com `AppTest`, sem `BRIEFING_*` no ambiente e sem `secrets.toml`:
  - `test_tela_mostra_mapa_de_calor_depois_de_calcular`: há um `vega_lite_chart` a mais;
  - `test_briefing_sem_chave_mostra_reserva_com_motivo`: clicar em "Gerar briefing" mostra texto e a legenda com "sem chave configurada", sem exceção;
  - `test_mapa_e_briefing_usam_o_contexto_calculado`: mudar o dia no formulário sem recalcular não muda a célula destacada.
- [ ] **Passo 4:** rodar e ver falhar; implementar; rodar e ver passar; `make teste`; `make qualidade`.
- [ ] **Passo 5:** abrir a tela de verdade (skill `run` ou Playwright), sem chave: calcular, conferir o mapa e gerar um briefing reserva. Se a tarefa 5 já tiver terminado, repetir com o `secrets.toml` do autor e gerar um briefing pelo LLM. Capturas em `reports/capturas/` (ignorado); apagar `.playwright-mcp/`.
- [ ] **Passo 6: commit.**

### Tarefa 7: API FastAPI

**Arquivos:**
- Criar: `src/occasion/api/{__init__,app}.py`, `tests/api/test_app.py`
- Modificar: `pyproject.toml`, `uv.lock` (`fastapi`, `uvicorn`), `Makefile` (alvo `api`)

**Interfaces:**
- Consome: `prever`, `carregar_artefato`, `ArtefatoModelo`, `Contexto`, `Resultado`.
- Produz: `criar_app(artefato: ArtefatoModelo | None = None) -> FastAPI`. Sem artefato, carrega um na subida (lifespan) por `carregar_artefato(Path(os.environ.get("OCCASION_MODELOS", MODELOS)))`. `app = criar_app()` no módulo. As rotas são `POST /predict` (corpo `Contexto`, resposta `Resultado`) e `GET /health`, que devolve `{"status": "ok", "modelo", "degrau", "treinado_em", "receita_commit", "versao_contrato"}`. Alvo `api: uv run uvicorn occasion.api.app:app --port 8000`.

- [ ] **Passo 1:** consultar o `context7` sobre FastAPI (lifespan, `TestClient`, `response_model`).
- [ ] **Passo 2: testes**, com `TestClient(criar_app(_artefato_unico()))`:
  - `test_predict_devolve_o_contrato`: os seis campos do TAPI mais `score_components`, com os tipos certos;
  - `test_predict_igual_a_prever`;
  - `test_idade_menor_que_18_da_422`;
  - `test_uf_invalida_da_422`;
  - `test_idade_em_texto_nao_numerico_da_422`;
  - `test_health_traz_o_modelo`.
- [ ] **Passo 3:** rodar e ver falhar; `uv add fastapi uvicorn`; conferir as versões assinadas; implementar; rodar e ver passar.
- [ ] **Passo 4:** `make api` em segundo plano, com `timeout`. Conferir que `curl -sf localhost:8000/health` e `curl -sf localhost:8000/docs` respondem. Encerrar o processo.
- [ ] **Passo 5:** `make qualidade`; commit.

### Tarefa 8: tabelas da documentação geradas dos JSON

**Arquivos:**
- Criar: `src/occasion/relatorios/docs.py`, `tests/relatorios/test_docs.py`, `tests/relatorios/test_docs_atualizados.py`, `docs/DATA_CARD.md` e `README.md` só com o esqueleto (títulos e marcadores)
- Modificar: `Makefile` (alvo `docs`)

**Interfaces:**
- Consome: `reports/contagens.json` (tarefa 2), `reports/hipoteses.json`, `reports/teste_final.json`, `reports/avaliacao_2008.json`.
- Produz:
  - `gerar_blocos(relatorios: Path) -> dict[str, str]`, com os blocos `contagens`, `hipoteses`, `teste_final`, `metricas_2008` e `segmentos_teste`.
  - `aplicar_blocos(texto: str, blocos: dict[str, str]) -> str`: substitui o conteúdo entre `<!-- BEGIN:nome -->` e `<!-- END:nome -->`; marcador sem bloco correspondente é erro.
  - `numeros_fora_dos_blocos(texto: str) -> list[str]`: decimais e porcentagens fora dos blocos que não aparecem em nenhum bloco do mesmo texto. Ignora o que está entre crases, em bloco de código e em URL.
  - `ARQUIVOS_COM_BLOCOS = ("README.md", "docs/DATA_CARD.md")`; `main()` reescreve os dois.
  - Alvo `docs: uv run python -m occasion.relatorios.docs`.

**Formato:**
- Tabelas Markdown em inglês, com ponto decimal.
- Métricas com 4 algarismos significativos; intervalos no formato `[ic_inf, ic_sup]`.
- Vereditos traduzidos: `confirmada` → `supported`, `refutada` → `refuted`, `sem evidência contra o padrão` → `no evidence against the default`, `não executada` → `not run`. Veredito desconhecido levanta `KeyError`.
- `hipoteses` tem uma linha por item de `hipoteses.json` (id, parte, veredito, estimativa e intervalo).
- `teste_final` traz H8 e as três partes de `metricas.partes`: PR-AUC da receita e da referência, ganho e intervalo.
- `segmentos_teste` sai de `metricas.segmentos`, com no máximo três dimensões, escolhidas lendo o JSON.

- [ ] **Passo 1: testes unitários:**
  - `test_aplicar_blocos_substitui_so_entre_marcadores`;
  - `test_marcador_sem_bloco_levanta_erro`;
  - `test_numero_no_texto_presente_no_bloco_passa`;
  - `test_numero_inventado_no_texto_e_apontado`;
  - `test_versoes_entre_crases_e_urls_sao_ignoradas` (com "`Python 3.12`" e um link com `v1.2`);
  - `test_veredito_desconhecido_levanta_erro`;
  - `test_blocos_reais_tem_h1_a_h10_e_h8_refutada` (lê `reports/`).
- [ ] **Passo 2: teste `test_docs_atualizados`**, parametrizado em `ARQUIVOS_COM_BLOCOS`: o arquivo existe; `aplicar_blocos(texto, gerar_blocos(...)) == texto`; `numeros_fora_dos_blocos(texto) == []`.
- [ ] **Passo 3:** rodar e ver falhar; implementar; criar os esqueletos com os marcadores; `make docs`; rodar e ver passar.
- [ ] **Passo 4:** `make qualidade`; commit.

### Tarefa 9: Data Card (em inglês)

**Arquivos:** modificar `docs/DATA_CARD.md`.

**Conteúdo obrigatório:**
- **Os itens do TAPI:** unidade de análise, target, dicionário de variáveis, fontes e versões, regras de integração, estratégia contra vazamento, população coberta e excluída, vieses e limitações, licença e LGPD.
- **As armadilhas da fonte:** sabor e não marca; sem data de calendário; sem canal; dias da semana desbalanceados, com a explicação da caderneta que começa na terça em 2008-2009; métodos de coleta diferentes; renda em salários mínimos (R$ 415 e R$ 954); local de 2008-2009 como fonte do alimento; regras R12 e R13.
- **O dicionário:** `COLUNAS_CONTEXTO` e as colunas de alvo, com tipo e domínio lidos de `config.py`, `contract.py` e `features/`.
- **A estratégia contra vazamento:** treino e teste separados no tempo, `carregar_base` e a tag com a assinatura da receita.
- **Números:** só nos blocos `contagens` e `metricas_2008`.

- [ ] **Passo 1:** escrever o texto em volta dos blocos.
- [ ] **Passo 2:** `make docs`, depois `uv run pytest tests/relatorios -v`. Esperado: tudo passando.
- [ ] **Passo 3: commit.**

### Tarefa 10: referências conferidas

**Arquivos:** criar `docs/REFERENCES.md` (em inglês).

- [ ] **Passo 1:** listar as referências de `docs/registro-de-decisoes.md` (seção "Fontes consultadas" e citações dentro das decisões) que sustentam as escolhas que o README vai citar. No mínimo: validação temporal e seleção (Hastie, Tibshirani e Friedman; Cawley e Talbot, 2010; Kapoor e Narayanan, 2023), modelo final reajustado (Steyerberg e Harrell, 2016; TRIPOD), eventos por variável (van der Ploeg, Austin e Steyerberg, 2014), curvas de aprendizado (Perlich, Provost e Simonoff, 2003; Figueroa e colegas, 2012), desbalanceamento (van den Goorbergh e colegas, 2022) e o que o registro citar sobre PR-AUC, calibração e bootstrap por conglomerado.
- [ ] **Passo 2:** conferir cada uma na web: autores, ano, título, veículo e DOI ou link que abre. Cada entrada leva a decisão do projeto que ela sustenta e a data da conferência.
- [ ] **Passo 3:** referência que não se confirma não entra. Anotar no ledger o que saiu e por quê.
- [ ] **Passo 4: commit.**

### Tarefa 11: README (em inglês) e licença

**Arquivos:** modificar `README.md`; criar `LICENSE` (MIT, com o titular confirmado com o autor).

**Interfaces:** consome os blocos da tarefa 8, `docs/DATA_CARD.md`, `docs/REFERENCES.md`, a URL do app da tarefa 1 e os alvos do `Makefile`.

**As 17 seções da seção 5.8 do design, nesta ordem:**

1. Title and pitch.
2. Try it.
3. The question.
4. Architecture: diagrama Mermaid das camadas, com o caminho de cada uma no código.
5. Data and target.
6. Method: cada escolha com a referência de `REFERENCES.md`.
7. Results: os blocos.
8. What the negative result means: H8 contada como no design, seção 1.
9. Decision Engine: as duas camadas, os cortes 75 e 50, e a regra rural como suposição.
10. Explainability and LLM briefing: inclui o limite declarado da conferência.
11. Limitations.
12. Reproduce everything: os comandos em ordem, com tempos aproximados (`make tudo` cerca de 34 s; escada, horas; o resto, minutos).
13. Repository map.
14. Next steps.
15. How this was built.
16. A note on language.
17. License.

O pitch e o "Try it" levam a URL do app. Enquanto o repositório não for público, ela fica marcada como "available after publication" e é trocada na tarefa 14.

- [ ] **Passo 1:** escrever. O texto corrido só cita decimais e porcentagens que estão nos blocos.
- [ ] **Passo 2:** `make docs`, depois `uv run pytest tests/relatorios -v`. Conferir que o Mermaid é válido (sintaxe do GitHub) e que todo link relativo aponta para um arquivo que existe (um laço `ls` simples sobre os caminhos citados).
- [ ] **Passo 3:** reler como um recrutador com dois minutos: o pitch e o "Try it" bastam? Ajustar.
- [ ] **Passo 4: commit.**

### Tarefa 12: integração e revisão

- [ ] **Passo 1:** juntar as frentes na `plano-3`, se foram feitas em worktrees. O conflito esperado é só no `Makefile`.
- [ ] **Passo 2:** `make teste`, `make teste-dados`, `make qualidade`, `make docs` sem diferença no `git diff`, e as duas conferências das restrições globais (receita intacta e versões).
- [ ] **Passo 3:** tela de verdade pelo Playwright, com e sem `secrets.toml`:
  - calcular;
  - mapa de calor;
  - briefing pelo LLM;
  - seis briefings em contextos diferentes, o sexto saindo como reserva por limite;
  - mudar o contexto sem recalcular.
  Apagar `.playwright-mcp/`.
- [ ] **Passo 4:** revisão da branch inteira (`superpowers:requesting-code-review`), com o checklist de método do `CLAUDE.md` mais: arquivos assinados intactos; versões iguais; nenhum segredo; README e Data Card sem alegação causal e sem marca; o briefing só recebe `Contexto` e `Resultado`. Corrigir o que a revisão achar.

### Tarefa 13: registro de decisões e `CLAUDE.md`

**Arquivos:** modificar `docs/registro-de-decisoes.md` (seção 28, "Decisões do Plano 3") e `CLAUDE.md`.

- [ ] **Passo 1:** seção 28 com as decisões do autor e do executor do Plano 3, no formato das seções anteriores (decisão, motivo, alternativas descartadas):
  - artefato commitado e refeito;
  - demo pública;
  - LLM com cache e limite;
  - provedor escolhido, com o resultado de `briefing_provedores.json`;
  - API mínima;
  - documentação em inglês e números gerados;
  - tudo publicado;
  - referências que saíram.
- [ ] **Passo 2:** `/revise-claude-md`. Comandos novos (`make contagens`, `make docs`, `make api`), as variáveis `BRIEFING_*`, o `secrets.toml`, a URL da demo e as armadilhas que o plano ensinou. Trocar "O Plano 3 ... ainda será escrito" pelo estado real.
- [ ] **Passo 3: commit.**

### Tarefa 14: publicação (ponto de parada com o autor)

- [ ] **Passo 1:** procurar segredos em todo o histórico com `git log -p --all | grep -nE "gsk_|nvapi-|sk-[A-Za-z0-9]{20}|BRIEFING_CHAVE *= *\S"`. Esperado: nada além de `.env.example` vazio. Conferir `git count-objects -vH` e que `.env` e `.streamlit/secrets.toml` não estão no git.
- [ ] **Passo 2 (autor confirma):** fundir a `plano-3` na `main` e enviar a `main` e a tag ao `origin`.
- [ ] **Passo 3 (autor confirma):** `gh repo edit <nome> --visibility public --accept-visibility-change-consequences`.
- [ ] **Passo 4 (autor):** no Streamlit Cloud, apontar o app para a `main`, cadastrar os `BRIEFING_*` em *Secrets* e deixar o app público.
- [ ] **Passo 5:** verificar ao vivo pelo Playwright, numa sessão anônima:
  - calcular;
  - mapa de calor;
  - briefing pelo LLM;
  - o sexto briefing sai como reserva por limite.
- [ ] **Passo 6:** trocar o marcador da URL no README pela URL real; `make docs`; `uv run pytest tests/relatorios`; commit; enviar.
- [ ] **Passo 7:** abrir o repositório no GitHub pelo Playwright e conferir que o README renderiza, com o diagrama Mermaid, as tabelas e os links funcionando.

---

## Ordem e paralelismo

1. **Tarefa 1** sozinha: ela define o arquivo de dependências do Cloud e o artefato commitado.
2. Depois, duas frentes em worktrees (`superpowers:using-git-worktrees`) saídas da `plano-3`:
   - **Código:** 3 → 4 → 6, com a 5 (ponto de parada) quando o autor tiver as chaves, e a 7 em qualquer ponto depois da 1. As tarefas 4 e 7 mexem no `pyproject.toml` e no `uv.lock`, por isso ficam na mesma frente, em sequência.
   - **Documentos:** 2 → 8 → 9, com a 10 em paralelo, e depois a 11. Na worktree de documentos, `data/` não existe: criar `ln -s <pasta principal>/data data` antes da tarefa 2.
3. **12 → 13 → 14**, em sequência, na `plano-3`.

Calendário-alvo: 22/11, tarefas 1 a 10; 23/11, tarefas 11 a 13; 24/11, tarefa 14. O dia 25/11 fica para o vídeo.

## Ordem de corte

1. Rota `/docs` e testes extras da API (a API mínima fica).
2. Bloco `segmentos_teste`.
3. Comparação de provedores: usar o Groq com um modelo da documentação.
4. Briefing pelo LLM na demo pública: publicar só com o texto reserva.

Os entregáveis obrigatórios e a publicação não entram na lista de corte.
