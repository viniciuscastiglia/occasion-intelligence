# Plano 2: EDA, escada de modelos, teste final e modelo servido

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** sair do degrau A e chegar à receita congelada, medida uma vez em 2017-2018, com o modelo servido reajustado e cada hipótese da seção 7 do design com resultado gravado em arquivo.

**Arquitetura:** três pacotes novos sobre o Plano 1. `occasion.validacao` (dobras do desenho amostral, métricas ponderadas, bootstrap por setor), `occasion.models` ganha a logística e o LightGBM atrás da mesma interface da tabela de taxas, e `occasion.experimentos` roda as etapas na ordem da seção 8.3 do design e grava um JSON por experimento. A receita escolhida vira `reports/receita.json`; o artefato da aplicação continua sendo um `ArtefatoModelo`, agora montado a partir da receita, e a tela não muda de interface.

**Stack:** a do Plano 1 mais `lightgbm` e `scipy` (dependência explícita), e no grupo de desenvolvimento `jupyter`, `nbconvert` e `ipykernel` para o caderno da EDA. As contribuições do LightGBM são valores SHAP calculados pelo TreeSHAP embutido no próprio LightGBM (`pred_contrib=True`), já em log-chances; o pacote `shap` não entra, porque o resultado é o mesmo e ele traz `numba` e atrito conhecido com divisões categóricas.

**Spec:** `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md`, seções 4.5 a 13. O `CLAUDE.md` e as decisões R9 a R14 e 26 de `docs/registro-de-decisoes.md` fazem parte da spec.

**Ponto de partida:** o estado final da branch `plano-1`, com `make tudo` já rodado (bases em `data/processed`, `reports/auditoria.json` e `models/avaliado/artefato.joblib`). Trabalhar numa branch `plano-2`.

**Fica para o Plano 3:** mapa de calor, briefing por LLM, API, Data Card, README e vídeo.

## Restrições globais

- Código, comentários, nomes e documentação em português. Funções públicas com anotações de tipo. `make qualidade` (ruff e pyright) sem erros antes de fechar cada tarefa. Formatar só `src/`, `tests/`, `app/`, `scripts/` e `notebooks/`.
- Antes de usar uma biblioteca, consultar a documentação da versão instalada pelo `context7` (`resolve-library-id` e `query-docs`), mandando só o nome da biblioteca e a dúvida técnica. O scikit-learn instalado é o 1.9: conferir a forma atual de pedir regularização L2 na `LogisticRegression`, que mudou nas versões recentes.
- **Edição de teste:** nenhuma coluna `y_*` nem `tem_pizza` de 2017-2018 é lida antes da tarefa 16. A única porta de leitura é `carregar_base` (tarefa 1), que recusa o alvo de 2017-2018 enquanto a tag `receita-congelada` não existir.
- **Nada de 2017-2018 entra no ajuste** do modelo avaliado nem do seu pré-processamento. Domínios de categoria vêm de constantes, não dos dados.
- **Pesos:** toda métrica e toda estimativa populacional usam `peso_final`. Se o peso entra no treino é decidido por H5 (tarefa 10). Pesos de treino são normalizados para média 1 dentro de cada conjunto de treino (e em cada réplica de bootstrap que retreina). A calibração (`ajustar_sigmoide`) usa sempre `peso_final`, porque mira a população, mesmo se H5 decidir treinar sem peso. Os nós das curvas (`SplineTransformer`) saem sem peso, porque o `sample_weight` não chega a ele dentro do `ColumnTransformer`; isso é declarado, não corrigido.
- **Sem reamostragem do evento raro.** Nenhum balanceamento de classes.
- **Semente:** `SEMENTE = 20261112` em tudo; derivações declaradas (`SEMENTE + k` para a dobra externa `k`).
- **Incerteza:** bootstrap de setores (`cod_upa`) inteiros, sorteados com reposição dentro de cada UF, `B = 1000`, intervalo percentil de 95%. A UF é o estrato do bootstrap porque os 550 estratos de 2008-2009 têm mediana de 5 setores (7 têm um só): com tão poucos setores, o bootstrap ingênuo dentro do estrato subestima a variância pelo fator (n−1)/n, cerca de 20%; por UF são dezenas a centenas de setores.
- **Regra de vitória:** o candidato mais complexo só vence se o intervalo do ganho excluir zero a seu favor. No empate, fica o mais simples.
- **Domínios fechados:** valor de contexto fora de `DOMINIOS` (tarefa 6) é recusado com `ValueError` pelos três tipos de modelo; valor do domínio ausente do treino de uma dobra é aceito.
- **Linguagem:** nenhum texto afirma causalidade; "guaraná" é o sabor, nunca a marca. O Google Trends é sempre "interesse de busca".
- **Registro:** todo experimento, inclusive os que falham, grava `reports/experimentos/<nome>.json` com configuração, semente, commit, dados usados e métricas. Previsões fora da dobra e modelos intermediários vão para `data/experimentos/` (fora do git).
- **Números em texto** são lidos de arquivos gerados pelo código. É proibido teste que afirma um limiar de métrica ajustado para passar.
- Testes que precisam dos microdados levam a marca `dados`. Um commit por tarefa, em português, terminando com `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Foco de revisão

1. **Alvo de 2017-2018 lido antes da hora**, por qualquer caminho: a leitura falha com `TesteBloqueado`. Teste na tarefa 1; a tarefa 14 acrescenta a trava de receita alterada depois da tag (pela assinatura dos arquivos da receita, não pelo commit, para o trabalho depois da tag continuar possível).
2. **Partes 1 e 2 com dobras internas diferentes:** o produto e a calibração do produto (H6, H7) misturariam previsões de modelos que viram aquele setor. As dobras internas vêm da tabela de setores compartilhada, nunca das linhas de cada base. Teste na tarefa 9.
3. **Categoria ausente no treino de uma dobra** (uma UF pequena, "madrugada fora de casa"): a previsão sai finita, sem `NaN` nem exceção. Testes nas tarefas 7 e 8.
4. **Probabilidade 0 ou 1** no log loss, no logit da calibração ou no produto: corte em `1e-15`, nunca infinito. Testes nas tarefas 3 e 8.
5. **Segmento ou dobra sem positivos:** a métrica sai como `None` com o motivo, sem `NaN` nem exceção. Testes nas tarefas 3 e 13.

---

## Estrutura de arquivos

```
src/occasion/
  bases/acesso.py            carregar_base com a trava da edição de teste
  validacao/dobras.py        dobras por setor dentro do estrato; dobras internas
  validacao/metricas.py      métricas ponderadas, calibração, corte, concentração, posições
  validacao/bootstrap.py     bootstrap de setores dentro da UF; intervalos
  eda/descritivas.py         taxas por grupo com intervalo; faixas de idade e renda
  experimentos/especificacao.py  tudo o que o pré-registro fixa (espaços, interações, B, k)
  experimentos/regras.py     regras de decisão das hipóteses e da escada
  experimentos/registro.py   JSON de experimento; salvar e carregar resultados
  experimentos/aninhada.py   validação aninhada e seleção final
  experimentos/etapas.py     uma função por etapa da seção 8.3
  experimentos/teste_final.py
  experimentos/pos_teste.py  diagnóstico 6.8 e H9
  experimentos/trends.py     leitura dos CSV e H10
  experimentos/hipoteses.py  consolidação do relatório de hipóteses
  experimentos/__main__.py   linha de comando
  models/base.py             interface ModeloParte, logit e sigmoide
  models/logistica.py        degraus B e C
  models/arvores.py          degrau D (contribuições por pred_contrib)
  models/calibracao.py       correção sigmoide
  models/escada.py           construir(degrau, parte, params) e espaços de busca
  avaliacao/segmentos.py  avaliacao/erros.py  avaliacao/explicacao.py
  receita.py                 Receita (Pydantic), salvar e carregar
  inference/treino.py        artefato avaliado a partir da receita (faz parte da receita)
  inference/servido.py       artefato servido (depois do teste; fora da receita)
notebooks/eda_2008.ipynb
docs/pre-registro.md
dados_externos/google_trends/   CSV exportados pelo autor e LEIAME.md
models/                      baseline/ (degrau A do Plano 1), avaliado/, servido/
reports/                     eda.json, dobras.json, receita.json, experimentos/*.json,
                             avaliacao_2008.json, teste_final.json, pos_teste.json,
                             trends.json, hipoteses.json
```

---

### Tarefa 1: dependências e porta única de leitura das bases

**Arquivos:**
- Criar: `src/occasion/bases/acesso.py`, `tests/bases/test_acesso.py`
- Modificar: `pyproject.toml`, `uv.lock`, `.gitignore`, `src/occasion/config.py`, `src/occasion/pipeline.py`, `src/occasion/inference/predict.py`, `tests/test_pipeline.py`, `tests/inference/test_predict.py`

**Interfaces:**
- Produz em `occasion.config`: `DADOS_EXPERIMENTOS: Path` (`data/experimentos`), `TAG_RECEITA: str = "receita-congelada"`.
- Produz em `occasion.bases.acesso`:
  - `class TesteBloqueado(RuntimeError)`
  - `COLUNAS_DE_ALVO: frozenset[str] = {"y_pizza_refri", "y_pizza_guarana", "y_guarana", "tem_pizza"}`
  - `teste_liberado(raiz: Path = RAIZ) -> bool`: `True` se `git tag -l TAG_RECEITA` devolve a tag.
  - `carregar_base(nome: Literal["celulas", "ocasioes_refri"], edicao: str, *, so_contexto: bool = False, pasta: Path = DADOS_PROCESSADOS, raiz: Path = RAIZ) -> pd.DataFrame`. Em 2008-2009 lê tudo. Em 2017-2018: com `so_contexto=True` lê sem `COLUNAS_DE_ALVO`; sem isso, exige `teste_liberado(raiz)` e senão levanta `TesteBloqueado`.
- Modifica a pasta do degrau A do Plano 1: `make baseline` passa a gravar em `models/baseline/`, para nunca ocupar o lugar do modelo avaliado. `carregar_artefato` procura `servido`, `avaliado` e `baseline`, nessa ordem; a mensagem de `ArtefatoAusente` continua mandando rodar `make baseline`.

- [ ] **Passo 1: dependências.** `uv add lightgbm scipy` e `uv add --dev jupyter nbconvert ipykernel`. Conferir, num script descartável no scratchpad, que um `LGBMClassifier` com uma coluna `pd.Categorical` treina e que `booster_.predict(X, pred_contrib=True)` devolve uma coluna por variável mais a base, somando a saída em log-chances. Se não somar, parar e relatar antes de seguir.
- [ ] **Passo 2: `.gitignore`.** Trocar as exceções de `reports/*` por `!reports/*.json` e `!reports/experimentos/`, mantendo `reports/capturas/` ignorada. `data/` já cobre `data/experimentos/`.
- [ ] **Passo 3: testes que falham** (repositório git criado em `tmp_path` com `git init` e um commit vazio feito com `git -c user.name=teste -c user.email=teste@exemplo.invalid commit`, para funcionar em máquina sem identidade configurada; parquets mínimos). Somar os testes de que `_baseline` grava em `models/baseline/` e de que `carregar_artefato` acha o artefato lá quando não há `servido` nem `avaliado`.

```python
def test_alvo_de_2017_bloqueado_sem_tag(repo_sem_tag, pasta_bases):
    with pytest.raises(TesteBloqueado):
        carregar_base("celulas", "2017-2018", pasta=pasta_bases, raiz=repo_sem_tag)

def test_contexto_de_2017_sai_sem_colunas_de_alvo(repo_sem_tag, pasta_bases):
    df = carregar_base("ocasioes_refri", "2017-2018", so_contexto=True,
                       pasta=pasta_bases, raiz=repo_sem_tag)
    assert not (set(df.columns) & COLUNAS_DE_ALVO)

def test_alvo_de_2017_liberado_com_tag(repo_com_tag, pasta_bases):
    df = carregar_base("celulas", "2017-2018", pasta=pasta_bases, raiz=repo_com_tag)
    assert "y_pizza_guarana" in df.columns

def test_2008_sempre_liberado(repo_sem_tag, pasta_bases):
    assert "y_guarana" in carregar_base("ocasioes_refri", "2008-2009",
                                        pasta=pasta_bases, raiz=repo_sem_tag).columns
```

- [ ] **Passo 4: rodar e ver falhar.** `uv run pytest tests/bases/test_acesso.py -v`
- [ ] **Passo 5: implementar.** A leitura com `so_contexto` usa `pd.read_parquet(columns=...)`, para o alvo nem ser carregado na memória.
- [ ] **Passo 6: rodar os testes, `make teste` e `make qualidade`.** Passam.
- [ ] **Passo 7: mover o degrau A.** Apagar `models/avaliado/` (é o degrau A do Plano 1), rodar `make baseline` e conferir `models/baseline/artefato.joblib`. Commit.

---

### Tarefa 2: dobras que imitam o desenho amostral

**Arquivos:**
- Criar: `src/occasion/validacao/__init__.py`, `src/occasion/validacao/dobras.py`, `tests/validacao/test_dobras.py`

**Interfaces:**
- Produz em `occasion.validacao.dobras`:
  - `K_EXTERNO = 5`, `K_INTERNO = 3`
  - `tabela_de_upas(base: pd.DataFrame) -> pd.DataFrame`: colunas `cod_upa`, `estrato`, uma linha por setor.
  - `atribuir_dobras(upas: pd.DataFrame, k: int, semente: int) -> pd.DataFrame`: colunas `cod_upa`, `estrato`, `dobra` (0 a `k-1`).
  - `dobras_internas(dobras: pd.DataFrame, externa: int, k: int = K_INTERNO, semente: int = SEMENTE) -> pd.DataFrame`: `atribuir_dobras` aplicada aos setores com `dobra != externa`, com semente `semente + externa`.
  - `dobra_das_linhas(base: pd.DataFrame, dobras: pd.DataFrame) -> np.ndarray`: a dobra de cada linha pelo `cod_upa`; setor ausente da tabela levanta `ValueError`.
  - `conferir_dobras(alvos: Mapping[str, tuple[pd.DataFrame, str]], dobras: pd.DataFrame) -> dict[str, dict]`: para cada nome, positivos por dobra (contagem simples), fração de cada dobra e `ok`.
  - `escolher_dobras(upas: pd.DataFrame, alvos: Mapping[str, tuple[pd.DataFrame, str]], semente: int = SEMENTE, tentativas: int = 20) -> tuple[pd.DataFrame, dict]`: tenta `semente`, `semente + 100`, `semente + 200`... até `conferir_dobras` aprovar tudo; devolve as dobras e o registro das tentativas.

**Algoritmo de `atribuir_dobras`:** permutar os setores com `np.random.default_rng(semente)`, ordenar de forma estável por `estrato` e atribuir `dobra = posição % k`. Cada estrato fica espalhado pelas dobras (diferença de no máximo um setor entre dobras dentro do estrato) e as dobras ficam do mesmo tamanho no total.

**Critério da conferência (pré-declarado aqui):** cada dobra externa tem entre 10% e 30% dos positivos de cada alvo. Os alvos conferidos são `y_pizza_refri` e `y_pizza_guarana` em `celulas`, `y_guarana` em `ocasioes_refri` e `y_guarana` em `ocasioes_refri` com `tem_pizza == 1`.

- [ ] **Passo 1: testes que falham.**

```python
def test_setor_inteiro_numa_dobra_so(base_sintetica):
    dobras = atribuir_dobras(tabela_de_upas(base_sintetica), 5, SEMENTE)
    assert dobras["cod_upa"].is_unique

def test_estratos_espalhados(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    por_estrato = d.groupby(["estrato", "dobra"]).size().unstack(fill_value=0)
    assert (por_estrato.max(axis=1) - por_estrato.min(axis=1)).max() <= 1

def test_mesma_semente_mesmas_dobras(upas_com_estratos):
    assert atribuir_dobras(upas_com_estratos, 5, 7).equals(atribuir_dobras(upas_com_estratos, 5, 7))

def test_dobras_internas_so_com_setores_de_treino(upas_com_estratos):
    d = atribuir_dobras(upas_com_estratos, 5, SEMENTE)
    internas = dobras_internas(d, externa=2)
    assert set(internas["cod_upa"]) == set(d.loc[d["dobra"] != 2, "cod_upa"])
    assert set(internas["dobra"]) == {0, 1, 2}

def test_setor_desconhecido_levanta_erro(base_sintetica, upas_com_estratos): ...

def test_conferencia_reprova_dobra_sem_positivos(base_com_positivos_num_setor): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar os testes.** Passam. Commit (o uso nos dados reais vem na tarefa 10, com a linha de comando).

---

### Tarefa 3: métricas ponderadas

**Arquivos:**
- Criar: `src/occasion/validacao/metricas.py`, `src/occasion/models/base.py` (só `logit` e `sigmoide`; a tarefa 7 completa o módulo), `tests/validacao/test_metricas.py`

**Interfaces:**
- Produz em `occasion.validacao.metricas` (todas recebem `y`, `p` e `w` como arrays 1-D do mesmo tamanho):
  - `EPS = 1e-15`
  - `log_loss_ponderado(y, p, w) -> float`, `brier_ponderado(y, p, w) -> float`
  - `pr_auc_ponderada(y, p, w) -> float | None` e `roc_auc_ponderada(y, p, w) -> float | None`: `None` quando só há uma classe.
  - `ajustar_sigmoide(p, y, w) -> tuple[float, float]`: `(inclinacao, intercepto)` de `logit(q) = inclinacao · logit(p) + intercepto`, por máxima verossimilhança ponderada (`scipy.optimize.minimize`), com `p` cortado em `[EPS, 1 - EPS]`.
  - `calibracao(y, p, w, n_faixas: int = 10) -> dict`: `o_e` (soma de `w·y` sobre soma de `w·p`), `intercepto_cil` (intercepto com a inclinação fixada em 1), `inclinacao` (de `ajustar_sigmoide`) e `curva` (lista de faixas por quantis ponderados de `p`, cada uma com `p_media`, `taxa_observada`, `n`, `positivos`). Valores iguais de `p` ficam sempre na mesma faixa; faixas vazias são omitidas (a tabela de taxas tem poucas dezenas de valores distintos).
  - `posicoes_ponderadas(valores, w) -> np.ndarray`: posição de 0 a 100 com a mesma regra de `ReferenciaScore.posicao` (fração ponderada com valor menor ou igual).
  - `metricas_no_corte(y, alta, w) -> dict`: `precision`, `recall`, `f1` ponderados e a matriz `{"vp", "fp", "fn", "vn"}` em soma de pesos; divisão por zero vira `None`.
  - `concentracao_topo(y, score, w, fracao: float = 0.10) -> float | None`: parcela ponderada dos positivos que cai nos 10% (em peso) de maior score. Um grupo empatado que atravessa a fronteira entra em proporção: a fração do peso dele que cabe nos 10%, aplicada aos positivos dele.
  - `resumo(y, p, w, posicao: np.ndarray | None = None) -> dict`: todas as anteriores; com `posicao`, o corte da classe alta é `posicao >= CORTE_ALTA` (de `occasion.score.score`).

- [ ] **Passo 1: testes que falham.**

```python
def test_log_loss_igual_ao_sklearn(y, p, w):
    assert log_loss_ponderado(y, p, w) == pytest.approx(log_loss(y, p, sample_weight=w))

def test_probabilidade_zero_ou_um_nao_da_infinito():
    y, p, w = np.array([1, 0]), np.array([0.0, 1.0]), np.ones(2)
    assert np.isfinite(log_loss_ponderado(y, p, w))
    assert np.isfinite(calibracao(y, p, w)["inclinacao"])

def test_sigmoide_recupera_parametros():
    rng = np.random.default_rng(SEMENTE)
    p = rng.uniform(0.02, 0.6, 200_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-(2 * logit(p) - 1))))
    a, b = ajustar_sigmoide(p, y, np.ones_like(p))
    assert a == pytest.approx(2, abs=0.1) and b == pytest.approx(-1, abs=0.1)

def test_auc_com_uma_classe_vira_none():
    assert pr_auc_ponderada(np.zeros(5), np.linspace(0, 1, 5), np.ones(5)) is None

def test_posicoes_batem_com_a_referencia_do_score(): ...   # mesma regra de ReferenciaScore.posicao

def test_concentracao_topo_quando_positivos_estao_no_topo():
    y = np.array([0] * 90 + [1] * 10); s = np.arange(100.0)
    assert concentracao_topo(y, s, np.ones(100)) == pytest.approx(1.0)

def test_corte_sem_positivos_previstos_da_precision_none(): ...
def test_concentracao_com_empate_na_fronteira_entra_em_proporcao(): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** `logit` corta `p` em `[EPS, 1 - EPS]`.
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 4: bootstrap de setores e regras de decisão

**Arquivos:**
- Criar: `src/occasion/validacao/bootstrap.py`, `src/occasion/experimentos/__init__.py`, `src/occasion/experimentos/regras.py`, `tests/validacao/test_bootstrap.py`, `tests/experimentos/test_regras.py`

**Interfaces:**
- Produz em `occasion.validacao.bootstrap`:
  - `B_PADRAO = 1000`
  - `@dataclass(frozen=True) class Intervalo`: `estimativa: float`, `ic_inf: float`, `ic_sup: float`, `b: int`, `nivel: float = 0.95`; propriedade `largura`.
  - `multiplicadores(cod_upa: np.ndarray, grupo: np.ndarray, b: int, semente: int) -> Iterator[np.ndarray]`: a cada réplica, sorteia com reposição, dentro de cada valor de `grupo` (a UF), tantos setores quantos o grupo tem, e devolve por linha quantas vezes o setor dela saiu. Gerador, para não guardar `b` vetores do tamanho da base.
  - `intervalo_estatistica(funcao: Callable[[np.ndarray], float], cod_upa, grupo, b=B_PADRAO, semente=SEMENTE) -> Intervalo`: `funcao` recebe os multiplicadores; a estimativa é `funcao(np.ones(n))`. Réplicas em que `funcao` devolve `None` (por exemplo, AUC sem positivos) são descartadas e contadas no campo `b`.
  - `intervalo_ganho(metrica: Callable, maior_e_melhor: bool, y, p_candidato, p_referencia, w, cod_upa, grupo, b=B_PADRAO, semente=SEMENTE) -> Intervalo`: ganho positivo favorece o candidato (`ref - cand` para perdas, `cand - ref` para PR-AUC). As duas previsões usam os mesmos multiplicadores em cada réplica (bootstrap pareado), aplicados como `w · multiplicador`.
  - `intervalo_por_unidades(funcao: Callable[[np.ndarray], float], n_unidades: int, b=B_PADRAO, semente=SEMENTE) -> Intervalo`: sorteia índices de unidades (as 27 UFs, em H10).
- Produz em `occasion.experimentos.regras`:
  - `superioridade(ic: Intervalo) -> Literal["confirmada", "refutada"]`: confirmada se `ic_inf > 0`.
  - `padrao(ic_ganho_da_alternativa: Intervalo) -> Literal["sem evidência contra o padrão", "padrão refutado"]`: refutado se `ic_inf > 0`. Usada em H5, H6 e H9.
  - `sinal_negativo(ic: Intervalo) -> Literal["confirmada", "efeito no sentido contrário", "os dados não mostram o efeito"]`: H4.
  - `correlacao_positiva(ics: Sequence[Intervalo]) -> Literal["confirmada", "refutada"]`: H10, confirmada se todos os `ic_inf > 0`.
  - `melhor_logistico(ic_c_contra_b: Intervalo) -> Literal["B", "C"]`: `"C"` se `superioridade` confirmar, senão `"B"`. É a referência de H3, e coincide com o campeão da escada entre os logísticos.
  - `vencedor_escada(comparar: Callable[[str, str], Intervalo]) -> tuple[str, list[dict]]`: campeão começa em `"A"`; para cada candidato em `B`, `C`, `D`, chama `comparar(candidato, campeao)` e troca o campeão se `superioridade` confirmar. Devolve o vencedor e a lista de comparações feitas.

**Por que o campeão sequencial:** H1, H2 e H3 continuam sendo relatadas como a seção 7 define (B contra A, C contra B, D contra o melhor logístico). O vencedor usa a regra do campeão para cobrir o caso em que B não supera A: aí C e D precisam superar a tabela, e não a logística que perdeu. Quando H1 é confirmada, as duas formas coincidem. Quando não coincidem, o JSON da escada grava as duas comparações (a da hipótese e a do campeão), para o relatório não se contradizer.

- [ ] **Passo 1: testes que falham.**

```python
def test_setor_sai_inteiro():
    upa = np.array(["a", "a", "b", "c"]); uf = np.array([1, 1, 1, 2])
    for m in itertools.islice(multiplicadores(upa, uf, 50, SEMENTE), 50):
        assert m[0] == m[1]                     # linhas do mesmo setor saem juntas
        assert m[3] == 1                        # UF com um setor só: ele sai uma vez
        assert m[0] + m[2] == 2                 # a UF 1 tem dois setores: dois sorteios

def test_ganho_nulo_quando_previsoes_iguais(y, p, w, upa, uf):
    ic = intervalo_ganho(log_loss_ponderado, False, y, p, p, w, upa, uf, b=200)
    assert ic.estimativa == 0 and ic.ic_inf == 0 and ic.ic_sup == 0

def test_ganho_positivo_para_previsao_melhor(): ...  # p verdadeira contra constante, n grande

def test_regras():
    assert superioridade(Intervalo(0.1, 0.01, 0.2, 1000)) == "confirmada"
    assert superioridade(Intervalo(0.1, -0.01, 0.2, 1000)) == "refutada"
    assert sinal_negativo(Intervalo(-0.5, -0.9, -0.1, 1000)) == "confirmada"
    assert sinal_negativo(Intervalo(0.5, 0.1, 0.9, 1000)) == "efeito no sentido contrário"
    assert sinal_negativo(Intervalo(0.1, -0.2, 0.4, 1000)) == "os dados não mostram o efeito"
    assert padrao(Intervalo(0.1, -0.01, 0.2, 1000)) == "sem evidência contra o padrão"

def test_escada_sem_h1_exige_superar_a_tabela():
    ganhos = {("B", "A"): neg, ("C", "A"): neg, ("D", "A"): pos}
    vencedor, comparacoes = vencedor_escada(lambda c, r: ganhos[(c, r)])
    assert vencedor == "D" and [c["referencia"] for c in comparacoes] == ["A", "A", "A"]

def test_escada_empate_fica_com_o_mais_simples(): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 5: EDA em 2008-2009

Sem teste antes para o caderno (seção 15 do design); as duas funções de faixa, que a avaliação por segmentos reusa, têm teste.

**Arquivos:**
- Criar: `src/occasion/eda/__init__.py`, `src/occasion/eda/descritivas.py`, `tests/eda/test_descritivas.py`, `notebooks/eda_2008.ipynb`
- Modificar: `Makefile` (alvo `eda`), `tests/inference/test_paridade.py` (passa a usar `faixa_de_renda`)

**Interfaces:**
- Produz em `occasion.eda.descritivas`:
  - `FAIXAS_ETARIAS: dict[str, tuple[int, int]] = {"10-17": (10, 17), "18-29": (18, 29), "30-44": (30, 44), "45-59": (45, 59), "60+": (60, 130)}` e `faixa_etaria(idade: pd.Series) -> pd.Series`
  - `faixa_de_renda(renda_pc_sm: pd.Series) -> pd.Series`: os códigos de `FaixaRenda` do contrato, com cortes 0,5, 1, 2 e 5 (fechado à esquerda).
  - `taxa_por_grupo(base: pd.DataFrame, alvo: str, por: list[str], b: int = 200, semente: int = SEMENTE) -> pd.DataFrame`: colunas `*por`, `n`, `positivos`, `taxa` (ponderada), `ic_inf`, `ic_sup` (por `intervalo_estatistica`).
- Produz `reports/eda.json` com os números que o pré-registro cita, e o comando `make eda` (`uv run jupyter nbconvert --to notebook --execute --inplace notebooks/eda_2008.ipynb`).

- [ ] **Passo 1: testes das faixas,** que falham: `faixa_etaria` de `[10, 17, 18, 29, 30, 60, 99]` dá `["10-17", "10-17", "18-29", "18-29", "30-44", "60+", "60+"]`; `faixa_de_renda` de `[0, 0.49, 0.5, 1, 1.99, 2, 5, 12]` dá os códigos na ordem esperada.
- [ ] **Passo 2: implementar e rodar.** Passam. Trocar o `_faixa_de_renda` local de `tests/inference/test_paridade.py`, que recebe um escalar, por `faixa_de_renda(pd.Series([valor])).iloc[0]`.
- [ ] **Passo 3: carregar a skill `dataviz`** antes do primeiro gráfico. Gráficos em Altair.
- [ ] **Passo 4: caderno.** Lê só `carregar_base(..., "2008-2009")` e `pessoa_dia_2008-2009.parquet`. Seções, cada uma com o gráfico ou tabela e uma frase de leitura:
  1. **Horário:** distribuição ponderada da hora (0 a 23) das ocasiões com pizza e refrigerante e de todas as ocasiões com refrigerante, com as faixas atuais de `FAIXAS_HORARIO` marcadas. Pergunta: as fronteiras das faixas cortam algum pico?
  2. **Taxas por variável:** `y_pizza_refri` em `celulas` e `y_guarana` em `ocasioes_refri`, por cada variável de `COLUNAS_CONTEXTO` (idade e renda pelas faixas acima), com intervalo.
  3. **Curvas:** as duas taxas por faixas de 5 anos de idade e por décimos ponderados de renda. Pergunta: os efeitos são retos (justifica ou não as curvas do degrau C)?
  4. **Interações candidatas** (as da tarefa 6): tabela de taxas por par e o desvio, em log-chances, entre a taxa do par e a soma dos efeitos marginais. É descrição; a lista de interações não é escolhida pelo tamanho desse desvio (ver tarefa 6).
  5. **Pizza e preferência:** participação ponderada do guaraná nas ocasiões com refrigerante com e sem pizza, com intervalo. Refaz em 2008-2009 a observação de 2017-2018 da seção 4.3 do design.
  6. **Sazonalidade:** taxa de `tem_pizza_refri` por trimestre, com a data reconstruída por `edicao_2008()` de `scripts/investigacao_datas.py`. Só descrição; nenhuma variável é criada (seção 26 do registro).
  7. **O que o TAPI diz:** "sexta a domingo, à noite e em casa" e a participação do guaraná por faixa de renda (contraste anotado no item 23 do registro).
  8. **Esparsidade do degrau A:** positivos por combinação das dimensões de `DIMENSOES_PARTE_1` e `DIMENSOES_PARTE_2`.
- [ ] **Passo 5: `make eda`** roda do zero e grava `reports/eda.json` com: positivos por faixa de horário, a hora de pico, a participação do guaraná com e sem pizza (estimativa e intervalo) e os positivos por combinação do degrau A.
- [ ] **Passo 6: conferir** que nenhuma célula do caderno lê 2017-2018. Commit do caderno com as saídas, do JSON e do código.

---

### Tarefa 6: pré-registro (ponto de parada para o autor)

Fixa, antes de qualquer modelo, tudo o que a seção 8.3 manda fixar na EDA, mais os detalhes operacionais das hipóteses. Nenhum experimento roda antes deste commit.

**Arquivos:**
- Criar: `docs/pre-registro.md`, `src/occasion/experimentos/especificacao.py`, `tests/experimentos/test_especificacao.py`
- Modificar: `src/occasion/features/variaveis.py`, `tests/features/test_variaveis.py`, `docs/registro-de-decisoes.md` (seção 27)

**Interfaces:**
- Produz em `occasion.features.variaveis`: `CATEGORICAS: list[str] = ["dia_semana", "faixa_horario", "local", "uf", "regiao", "situacao", "sexo"]`, `NUMERICAS: list[str] = ["idade", "renda_pc_sm", "n_moradores_uc"]` e `DOMINIOS: dict[str, list]` com os valores possíveis de cada categórica, montados a partir de constantes (`range(1, 8)`, `FAIXAS_HORARIO`, `LOCAIS`, `CODIGO_UF.values()`, as cinco regiões, `urbano`/`rural`, `masculino`/`feminino`, e `tem_pizza` em `[0, 1]`), nunca dos dados. E `checar_dominios(X: pd.DataFrame) -> None`, que levanta `ValueError` nomeando a coluna e o valor quando uma categórica presente em `X` tem valor fora de `DOMINIOS`.
- Produz em `occasion.experimentos.especificacao`:
  - `ParteModelo = Literal["ocasiao", "preferencia", "direto"]` e `variaveis_da_parte(parte: ParteModelo) -> list[str]`: `COLUNAS_CONTEXTO`, mais `tem_pizza` na preferência.
  - `GRADE_C_LOGISTICA: list[float] = list(np.logspace(-3, 2, 8))`, em ordem crescente (a primeira é a mais regularizada, isto é, a mais simples).
  - `SPLINE = {"n_knots": 4, "degree": 3, "knots": "quantile", "include_bias": False}`, aplicada a `idade` e a `log1p(renda_pc_sm)`: 5 colunas por curva, igual aos graus de liberdade.
  - `INTERACOES: dict[ParteModelo, list[tuple[str, str]]]`:
    - ocasião e direto: `("dia_semana", "faixa_horario")`, `("faixa_horario", "local")`, `("local", "situacao")`;
    - preferência: `("tem_pizza", "local")`, `("tem_pizza", "regiao")`, `("regiao", "local")`, `("faixa_horario", "local")`.
  - `N_CONFIGS_LIGHTGBM = 30` e `espaco_lightgbm(semente: int = SEMENTE) -> list[dict]`: `ParameterSampler` com `learning_rate` log-uniforme em [0,01; 0,2], `n_estimators` inteiro em [100, 800], `max_depth` em {2, 3, 4} (com `num_leaves = 2 ** max_depth`), `min_child_samples` em {200, 500, 1000, 2000}, `reg_lambda` log-uniforme em [0,01; 10], `subsample` em {0,7; 1,0} com `subsample_freq = 1`, `colsample_bytree` em {0,7; 1,0}. Fixos: `objective="binary"`, `deterministic=True`, `force_row_wise=True`, `n_jobs=4`, `random_state=SEMENTE`, `verbose=-1`. Ordenada da configuração mais simples para a mais complexa por `(max_depth, n_estimators)`, para o desempate.
  - `FORCA_SUAVIZACAO_A = 50.0` e as dimensões do degrau A de `models/taxas.py`, sem busca.
  - `graus_de_liberdade(parte: ParteModelo, degrau: Literal["B", "C"]) -> int`: categórica com `L` níveis conta `L - 1`; numérica conta 1; curva conta `n_knots + degree - 2`; interação `a × b` conta `(La - 1)(Lb - 1)`.
  - `parametros_de_pizza(parte: ParteModelo = "preferencia") -> int`: `tem_pizza` mais os graus das interações que a envolvem.

**Por que as interações vêm de argumento e não do tamanho do efeito na EDA:** escolher interações olhando o alvo de 2008-2009 inteiro e depois medir C contra B na mesma base deixaria H2 otimista. As candidatas saem do enunciado do TAPI ("sexta a domingo, à noite e em casa") e das perguntas do projeto (a pizza muda a preferência dentro e fora de casa, e por região). A EDA as descreve; o autor pode vetar ou trocar neste ponto de parada, antes de qualquer modelo.

- [ ] **Passo 1: testes que falham** (`tests/experimentos/test_especificacao.py`):

```python
def test_degrau_c_cabe_no_tamanho_de_amostra():
    riley = json.loads((RELATORIOS / "auditoria.json").read_text())["tamanho_de_amostra"]
    assert graus_de_liberdade("ocasiao", "C") <= riley["parte1"]["parametros_sustentados"]
    assert graus_de_liberdade("preferencia", "C") <= riley["parte2"]["parametros_sustentados"]
    assert parametros_de_pizza() <= riley["parte2_recorte_pizza"]["parametros_sustentados"]

def test_espaco_lightgbm_tem_30_configuracoes_reprodutiveis():
    assert len(espaco_lightgbm()) == 30 and espaco_lightgbm() == espaco_lightgbm()

def test_dominios_cobrem_as_bases_sinteticas(): ...   # todo valor das bases de teste está em DOMINIOS
def test_checar_dominios_recusa_valor_fora(): ...

@pytest.mark.dados
def test_dominios_cobrem_as_duas_edicoes():
    for edicao in EDICOES:
        for nome in ("celulas", "ocasioes_refri"):
            checar_dominios(carregar_base(nome, edicao, so_contexto=True))
```

- [ ] **Passo 2: rodar, ver falhar, implementar, rodar.** Passam. Com as interações acima: a ocasião B tem 46 graus; a C troca os dois termos lineares de idade e renda por duas curvas de 5 graus e soma 29 de interações, 46 − 2 + 10 + 29 = 83 (limite 171); `parametros_de_pizza()` dá 1 + 1 + 4 = 6 (limite 6). Se a conta der outro número, conferir a fórmula antes de mexer na lista.
- [ ] **Passo 2b: só se a EDA levar a mudar as faixas de horário.** Alterar `config.FAIXAS_HORARIO` e `contract.FaixaHorario` (e os rótulos), rodar `make bases auditoria baseline`, ajustar os testes de modelagem e de paridade que dependem do número de faixas e só então rodar de novo os testes do passo 1 (a conta de Riley em `auditoria.json` muda junto). Se as faixas ficarem como estão, pular.
- [ ] **Passo 3: escrever `docs/pre-registro.md`.** Para cada item, o valor e a origem (design, EDA ou este plano):
  - faixas de horário (mantidas ou alteradas pela seção 1 da EDA, com o número de `reports/eda.json` que sustenta);
  - dimensões e suavização do degrau A;
  - variáveis de cada parte e as interações do degrau C, com o argumento acima;
  - espaços de busca, dobras (`K_EXTERNO`, `K_INTERNO`, critério de 10% a 30%), bootstrap (setor dentro da UF, `B = 1000`);
  - para cada hipótese H1 a H10: enunciado, motivo, teste, métrica, regra (a função de `regras.py` que a aplica) e o que muda se for refutada, copiados da seção 7 do design e completados com os detalhes operacionais deste plano: a regra do campeão e o `melhor_logistico` de H3 (tarefa 4); H4 com o peso decidido em H5 (tarefa 10); o detalhe de H7 (tarefa 11); e a agregação de H9: a base inteira é o padrão e é recusada se qualquer uma das três comparações (ocasião, preferência, recorte de pizza) a recusar, e aí as duas partes do modelo servido usam só 2017-2018;
  - cortes de `classification` (80 e 50) e do Decision Engine (75 e 50), mantidos, com a análise de sensibilidade da tarefa 11;
  - segmentos da avaliação: dia, faixa de horário, região, faixa etária, sexo, faixa de renda e local, com mínimo de 10 positivos para relatar métrica.
- [ ] **Passo 3b: seção 27 em `docs/registro-de-decisoes.md`** ("Decisões do Plano 2"), no formato das anteriores, com decisão, motivo e alternativa descartada: bootstrap de setores dentro da UF; regra do campeão; calibração ajustada nas previsões internas; interações do degrau C por argumento, antes da EDA, embora o design diga "definidas na EDA"; critério de 10% a 30% de positivos por dobra; contribuições do LightGBM pelo TreeSHAP do próprio LightGBM; e a pasta separada do degrau A (`models/baseline/`). Marcar como escolha do executor, aprovada no passo 4.
- [ ] **Passo 4: parar e pedir a aprovação do autor.** Mostrar o pré-registro e os pontos de `reports/eda.json` que ele usa. Só seguir depois do "aprovado"; mudanças pedidas entram aqui, com os testes do passo 1 rodando de novo.
- [ ] **Passo 5: commit** com a mensagem "Pré-registro: receita e hipóteses antes dos modelos".

---

### Tarefa 7: interface comum dos modelos e a logística (degraus B e C)

**Arquivos:**
- Criar: `src/occasion/models/logistica.py`, `tests/models/test_logistica.py`, `tests/models/test_base.py`
- Modificar: `src/occasion/models/base.py` (criado na tarefa 3)
- Modificar: `src/occasion/models/taxas.py`, `src/occasion/inference/predict.py` (`_top_fatores`), `tests/inference/test_predict.py`

**Interfaces:**
- Produz em `occasion.models.base`:
  - `logit(p) -> np.ndarray` (com corte em `EPS`) e `sigmoide(z) -> np.ndarray`
  - `class ModeloParte(Protocol)`: `variaveis: list[str]`; `base_: float`; `fit(X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series | None) -> Self`; `predict_proba(X) -> np.ndarray` (1-D); `contribuicoes(X) -> pd.DataFrame` (uma coluna por item de `variaveis`, em log-chances, em relação ao contexto médio).
  - Nos degraus B, C e D, `contribuicoes(X).sum(axis=1) + base_ == logit(predict_proba(X))`. Na tabela de taxas não: a previsão vem da tabela conjunta e as contribuições são marginais (seção 10 do design), e isso fica dito na docstring.
- Modifica `TabelaDeTaxas`: propriedades `variaveis` (devolve `dimensoes`) e `base_` (`logit(taxa_geral_)`), que funcionam também em objetos carregados de artefatos antigos; `fit` aceita `sample_weight=None` (pesos iguais).
- Produz em `occasion.models.logistica`:
  - `ModeloLogistico(parte: ParteModelo, C: float, com_curvas: bool = False, interacoes: Sequence[tuple[str, str]] = ())`
  - `fit` e `predict_proba` chamam `checar_dominios` antes de tudo. `ColumnTransformer` com `OneHotEncoder(categories=[DOMINIOS[c] ...])` em todas as categóricas, sem descartar nível (a regularização puxa cada UF para perto da média, e `regiao` e `uf` juntas dão o efeito de "UF perto da média da região"); numéricas com `StandardScaler`, renda como `log1p`; com `com_curvas`, `SplineTransformer(**SPLINE)` em idade e renda; cada interação vira uma coluna categórica cruzada (`"6|noite"`) codificada em one-hot; `tem_pizza` passa sem escala, para o coeficiente ficar em log-chances.
  - `LogisticRegression` com regularização L2 e `solver="newton-cholesky"` (aceita `sample_weight`, converge rápido com muito mais linhas que colunas); conferir no `context7` se ele aceita a matriz esparsa do `ColumnTransformer` e, se não aceitar, densificar.
  - atributos depois do `fit`: `base_: float` (log-chances do contexto médio ponderado do treino) e `n_parametros: int`, calculado a partir do que foi ajustado (`categories_` do encoder: `L − 1` por categórica e `(La − 1)(Lb − 1)` por interação; 1 por numérica; colunas de cada curva), e não do número de colunas, porque o one-hot guarda todos os níveis.
  - `coeficiente(nome: str) -> float` (usado em H4 com `"tem_pizza"`).
  - `contribuicoes(X)`: para cada variável, a soma de `coef · (x − média ponderada do treino)` nas colunas que vêm dela; a coluna de uma interação é dividida meio a meio entre as duas variáveis.
- Modifica `_top_fatores`: itera `modelo.variaveis` e pula `tem_pizza`, que é fixada em "sim" e não é um fator do contexto.

- [ ] **Passo 1: testes que falham.**

```python
def test_contribuicoes_somam_o_logit(base_sintetica_p1):
    m = ModeloLogistico("ocasiao", C=1.0, com_curvas=True,
                        interacoes=INTERACOES["ocasiao"]).fit(X, y, w)
    soma = m.contribuicoes(X).sum(axis=1).to_numpy() + m.base_
    np.testing.assert_allclose(soma, logit(m.predict_proba(X)), atol=1e-8)

def test_categoria_ausente_no_treino_da_previsao_finita(base_sem_uf_13):
    m = ModeloLogistico("ocasiao", C=1.0).fit(X_sem_am, y, w)
    p = m.predict_proba(X_so_am)
    assert np.isfinite(p).all()

def test_graus_de_liberdade_batem_com_a_especificacao():
    m = ModeloLogistico("preferencia", C=1.0, com_curvas=True,
                        interacoes=INTERACOES["preferencia"]).fit(X2, y2, w2)
    assert m.n_parametros == graus_de_liberdade("preferencia", "C")

def test_tabela_antiga_tem_variaveis(tmp_path):
    t = TabelaDeTaxas(["local"]).fit(X, y, w); joblib.dump(t, tmp_path / "t.joblib")
    assert joblib.load(tmp_path / "t.joblib").variaveis == ["local"]

def test_top_fatores_nao_mostra_tem_pizza(artefato_logistico): ...
def test_fit_sem_peso(): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** Consultar no `context7`: `LogisticRegression` (regularização L2 e `sample_weight` no 1.9), `SplineTransformer`, `OneHotEncoder(categories=...)`, `newton-cholesky` com matriz esparsa.
- [ ] **Passo 4: rodar `make teste`.** Os testes antigos de inferência e de paridade continuam passando. Commit.

---

### Tarefa 8: LightGBM (degrau D), calibração e a fábrica da escada

**Arquivos:**
- Criar: `src/occasion/models/arvores.py`, `src/occasion/models/calibracao.py`, `src/occasion/models/escada.py`, `tests/models/test_arvores.py`, `tests/models/test_calibracao.py`, `tests/models/test_escada.py`

**Interfaces:**
- Produz em `occasion.models.arvores`: `ModeloLightGBM(parte: ParteModelo, **params)`. Chama `checar_dominios`; categóricas como `pd.Categorical` com as categorias de `DOMINIOS` (valor do domínio ausente no treino é aceito); numéricas sem transformação. `contribuicoes` por `booster_.predict(X, pred_contrib=True)`: as colunas de variáveis são as contribuições em log-chances e a última é a base. `base_` é a média ponderada dessa última coluna no treino (ela é a mesma em todas as linhas).
- Produz em `occasion.models.calibracao`: `ModeloCalibrado(base: ModeloParte, inclinacao: float, intercepto: float)`: `variaveis` delega ao base; `predict_proba = sigmoide(inclinacao · logit(p_base) + intercepto)`; `contribuicoes = inclinacao · base.contribuicoes`; `base_ = inclinacao · base.base_ + intercepto`.
- Produz em `occasion.models.escada`:
  - `Degrau = Literal["A", "B", "C", "D"]`
  - `construir(degrau: Degrau, parte: ParteModelo, params: dict) -> ModeloParte`: A é `TabelaDeTaxas` com as dimensões da parte (direto usa as da ocasião) e `FORCA_SUAVIZACAO_A`; B é `ModeloLogistico(parte, **params)`; C é `ModeloLogistico(parte, com_curvas=True, interacoes=INTERACOES[parte], **params)`; D é `ModeloLightGBM(parte, **params)`.
  - `espaco(degrau: Degrau) -> list[dict]`: `[{}]` para A; `[{"C": c} for c in GRADE_C_LOGISTICA]` para B e C; `espaco_lightgbm()` para D.

- [ ] **Passo 1: testes que falham.**

```python
def test_contribuicoes_somam_o_logit(base_sintetica_p1):
    m = ModeloLightGBM("ocasiao", **espaco_lightgbm()[0]).fit(X, y, w)
    soma = m.contribuicoes(X).sum(axis=1).to_numpy() + m.base_
    np.testing.assert_allclose(soma, logit(m.predict_proba(X)), atol=1e-6)

def test_lightgbm_reprodutivel(): ...           # dois fits iguais, previsões idênticas
def test_lightgbm_categoria_ausente_no_treino(): ...
def test_calibrado_identidade_quando_a1_b0(): ...
def test_calibrado_mantem_a_soma_das_contribuicoes(): ...   # com base B e com base D, não com a tabela
def test_calibrado_sobre_tabela_tem_base(): ...             # tabela vencedora e calibrada não quebra
def test_calibrado_com_p_extremo_e_finito(): ...  # p_base 0 e 1
def test_construir_cobre_os_quatro_degraus_e_as_tres_partes(): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** Consultar no `context7`: `lightgbm.LGBMClassifier` (parâmetros, `categorical_feature`, determinismo, `predict(..., pred_contrib=True)` com categóricas).
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 9: validação aninhada, seleção final e registro

**Arquivos:**
- Criar: `src/occasion/experimentos/aninhada.py`, `src/occasion/experimentos/registro.py`, `tests/experimentos/test_aninhada.py`, `tests/experimentos/test_registro.py`

**Interfaces:**
- Produz em `occasion.experimentos.aninhada`:
  - `@dataclass class PrevisaoInterna`: `indices: np.ndarray` (posições, na base, das linhas de treino da dobra externa), `p: np.ndarray` (previsão fora da dobra interna, da configuração escolhida), `extras: dict[str, tuple[np.ndarray, np.ndarray]]` (idem para cada base extra: posições e previsões).
  - `@dataclass class ResultadoAninhado`: `oof: np.ndarray`; `oof_extras: dict[str, np.ndarray]`; `parametros: list[dict]` (escolhidos em cada dobra externa); `interno: list[PrevisaoInterna]`; `modelos: list[ModeloParte]` (um por dobra externa).
  - `validacao_aninhada(base: pd.DataFrame, alvo: str, degrau: Degrau, parte: ParteModelo, dobras: pd.DataFrame, *, usar_peso: bool, bases_extras: Mapping[str, pd.DataFrame] = {}, construtor: Callable[[Degrau, ParteModelo, dict], ModeloParte] = construir, semente: int = SEMENTE) -> ResultadoAninhado`
  - `previsao_fora_da_dobra(base, alvo, degrau, parte, params: dict, dobras, *, usar_peso, bases_extras={}, construtor=construir) -> tuple[np.ndarray, dict[str, np.ndarray]]`: validação simples nas dobras externas, com parâmetros fixos.
  - `selecao_final(base, alvo, degrau, parte, dobras, *, usar_peso, bases_extras={}, construtor=construir) -> tuple[dict, np.ndarray, dict[str, np.ndarray]]`: escolhe a configuração pelo log loss ponderado de `previsao_fora_da_dobra` nas 5 dobras externas e devolve os parâmetros e as previsões fora da dobra deles (base e extras). É o que fixa os hiperparâmetros do modelo final, treinado depois em 2008-2009 inteira; as previsões são salvas e reaproveitadas na calibração do artefato (tarefa 12), sem recalcular.

**Algoritmo de `validacao_aninhada`:** para cada dobra externa `k`: treino são as linhas cujo setor tem `dobra != k`. As dobras internas vêm de `dobras_internas(dobras, k)`, isto é, da tabela de setores, e por isso são as mesmas para qualquer base que use a mesma tabela. Para cada configuração de `espaco(degrau)`, treina em cada dobra interna e prevê as linhas de validação interna da base, guardando os três modelos internos; escolhe a configuração de menor log loss ponderado na base (empate: a primeira da lista); só então prevê as bases extras com os três modelos internos da escolhida e descarta os outros. Retreina a escolhida no treino externo inteiro e prevê as linhas da dobra `k` da base e das bases extras. Bases extras são previstas com `tem_pizza = 1` quando a parte é a preferência (é o uso do produto: a parte 2 prevista nas células). Com `usar_peso`, o treino recebe `peso_final` normalizado para média 1; a seleção sempre usa a métrica ponderada.

- Produz em `occasion.experimentos.registro`:
  - `registrar(nome: str, natureza: Literal["confirmatório", "exploratório"], configuracao: dict, metricas: dict, dados: dict, pasta: Path = RELATORIOS / "experimentos") -> Path`: grava `<nome>.json` com essas chaves mais `commit` (`commit_da_receita`), `semente`, `data_hora` e `versoes` (python, pandas, scikit-learn, lightgbm). Converte `Intervalo` e tipos do NumPy para JSON.
  - `salvar_resultado(nome: str, objeto: Any) -> Path` e `carregar_resultado(nome: str) -> Any`, em `DADOS_EXPERIMENTOS / f"{nome}.joblib"` (só arquivos gerados pelo próprio pipeline, como o artefato do Plano 1).
  - **Natureza (seção 6.5):** confirmatórios são H1 a H10 e as comparações da escada; dobras, avaliação por segmento, erros, explicabilidade, região fora e diagnóstico são exploratórios.

- [ ] **Passo 1: testes que falham,** com um construtor espião que guarda os `cod_upa` de cada `fit` e de cada `predict_proba`:

```python
def test_setor_nunca_dos_dois_lados(base_sintetica, dobras):
    espiao = Espiao()
    validacao_aninhada(base_sintetica, "y", "B", "ocasiao", dobras,
                       usar_peso=True, construtor=espiao.construir)
    for treino, previsto in espiao.pares():
        assert not (treino & previsto)

def test_dobras_internas_iguais_na_base_e_nas_extras(base_sintetica, extra, dobras): ...
def test_oof_cobre_cada_linha_uma_vez_sem_nan(): ...
def test_mesma_semente_mesmo_resultado_logistica_e_lightgbm(): ...
def test_rotulos_embaralhados_nao_vencem_a_constante(base_sintetica_com_sinal, dobras):
    embaralhada = base.assign(y=np.random.default_rng(SEMENTE).permutation(base["y"]))
    r = validacao_aninhada(embaralhada, "y", "B", "ocasiao", dobras, usar_peso=True)
    constante = np.full(len(base), np.average(embaralhada["y"], weights=embaralhada["peso_final"]))
    ic = intervalo_ganho(log_loss_ponderado, False, embaralhada["y"], r.oof, constante, ...)
    assert superioridade(ic) == "refutada"

def test_registro_grava_campos_obrigatorios(tmp_path): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 10: peso no treino (H5), escada (H1 a H3), H4 e rótulos embaralhados

Experimento: sem teste antes; verificação por script reprodutível e JSON gravado. A tarefa entrega também a linha de comando.

**Arquivos:**
- Criar: `src/occasion/experimentos/etapas.py`, `src/occasion/experimentos/__main__.py`, `tests/experimentos/test_etapas.py` (marca `dados` para o teste de rótulos embaralhados na parte 2)
- Modificar: `Makefile`

**Interfaces:**
- Produz em `occasion.experimentos.etapas`, cada função lendo de `carregar_base`/`carregar_resultado` e gravando com `registrar`/`salvar_resultado`:
  - `etapa_dobras() -> None`: `escolher_dobras` sobre os setores de `celulas` 2008-2009 com os quatro alvos da tarefa 2; grava `data/experimentos/dobras.parquet` e `reports/dobras.json` (semente usada, tentativas, positivos por dobra).
  - `etapa_peso() -> None` (H5): degrau B de cada parte, com e sem peso; ganho da versão sem peso por `intervalo_ganho` no log loss ponderado; `padrao` decide. Grava `h5_peso.json` com `usar_peso` por parte.
  - `etapa_escada(parte: Literal["ocasiao", "preferencia"]) -> None`: A, B, C e D por `validacao_aninhada` com o peso decidido em H5 (o B de H5 é reaproveitado, não refeito). Na preferência, `bases_extras = {"celulas": celulas}`. Grava `escada_<parte>.json` com, por degrau, `resumo` das métricas fora da dobra (na preferência, também no recorte `tem_pizza == 1`), os parâmetros por dobra; as comparações H1 (B×A), H2 (C×B) e H3 (D contra `melhor_logistico` de H2) com intervalo e veredito; e o vencedor por `vencedor_escada`. Salva cada `ResultadoAninhado`.
  - `etapa_h4() -> None`: `selecao_final` do B da preferência; coeficiente de `tem_pizza` do modelo em 2008-2009 inteira; intervalo por `intervalo_estatistica`, retreinando com o `C` escolhido fixo e pesos `(peso_final se usar_peso senão 1) · multiplicador`, renormalizados para média 1 em cada réplica (o uso de peso é o decidido em H5 para a preferência); `sinal_negativo` decide. Grava `h4_tem_pizza.json` com o `C` usado e o aviso de que o coeficiente fica encolhido pela regularização.
  - `etapa_embaralhado() -> None`: B das duas partes com o alvo permutado (semente fixa) contra a constante igual à prevalência ponderada do treino de cada dobra. Grava `rotulos_embaralhados.json`.
- Produz `uv run python -m occasion.experimentos <etapa>` e os alvos `make dobras`, `make peso`, `make escada` (as duas partes), `make h4`, `make embaralhado`.

- [ ] **Passo 1: teste de integração das etapas em base sintética,** que falha: com `DADOS_PROCESSADOS`, `DADOS_EXPERIMENTOS` e `RELATORIOS` apontados para `tmp_path` (monkeypatch), `etapa_dobras`, `etapa_peso` e `etapa_escada("preferencia")` rodam em uma base sintética pequena e gravam JSONs com as chaves `configuracao`, `metricas`, `commit`. O espaço do D é reduzido a duas configurações por monkeypatch, só neste teste.
- [ ] **Passo 2: teste `dados`:** `etapa_embaralhado` restrita à preferência em 2008-2009 dá `superioridade == "refutada"`.
- [ ] **Passo 3: implementar e rodar os testes.**
- [ ] **Passo 4: rodar nos dados, na ordem:** `make dobras`, `make peso`, `make escada`, `make h4`, `make embaralhado`. A escada da ocasião com o LightGBM é a etapa mais longa (centenas de ajustes em ~535 mil linhas); rodar em segundo plano. Se a conferência das dobras trocar a semente, a troca fica em `reports/dobras.json`.
- [ ] **Passo 5: ler os JSONs** e conferir: rótulos embaralhados não vencem a constante nas duas partes; nenhuma métrica `None` inesperada. Se o embaralhado vencer, parar e usar `superpowers:systematic-debugging` (é sinal de vazamento).
- [ ] **Passo 6: commit** do código e de `reports/dobras.json` e `reports/experimentos/*.json`.

---

### Tarefa 11: calibração, produto (H6 e H7), cortes e a receita

**Arquivos:**
- Criar: `src/occasion/receita.py`, `tests/test_receita.py`
- Modificar: `src/occasion/experimentos/etapas.py`, `src/occasion/experimentos/__main__.py`, `Makefile`

**Interfaces:**
- Produz em `occasion.receita` (Pydantic):
  - `ReceitaParte`: `degrau: Degrau`, `parametros: dict[str, Any]`, `usar_peso: bool`, `calibrar: bool`
  - `Receita`: `faixas_horario: dict[str, list[int]]`, `variaveis: dict[ParteModelo, list[str]]`, `interacoes: dict[ParteModelo, list[tuple[str, str]]]`, `ocasiao: ReceitaParte`, `preferencia: ReceitaParte`, `score: Literal["produto", "direto"]`, `direto: ReceitaParte | None`, `calibrar_score: bool`, `cortes: dict[str, float]` (`alta`, `media`, `ocasiao`, `preferencia`), `experimentos: list[str]` (os JSON que sustentam cada escolha)
  - `salvar_receita(receita: Receita, caminho: Path = RELATORIOS / "receita.json") -> Path` e `carregar_receita(caminho: Path = RELATORIOS / "receita.json") -> Receita`
  - A receita guarda escolhas, não valores ajustados: os coeficientes da sigmoide são do artefato, porque o modelo servido os reajusta.
- Produz em `etapas`:
  - `etapa_calibracao() -> None`: para o vencedor de cada parte, a sigmoide é ajustada em cada dobra externa com `ajustar_sigmoide` sobre `interno[k].p` e aplicada a `oof` da dobra `k`; o ganho da versão calibrada no log loss ponderado decide por `superioridade` (seção 8.2: "comparada como qualquer outra opção"). Grava `calibracao.json` com os três níveis de calibração antes e depois.
  - `etapa_produto() -> None`: o produto fora da dobra é `oof` da ocasião nas células vezes `oof_extras["celulas"]` da preferência (cada um com a calibração decidida), contra `y_pizza_guarana`. **H6:** modelo direto do mesmo degrau do vencedor da ocasião, por `validacao_aninhada` com alvo `y_pizza_guarana`, o uso de peso decidido em H5 para a ocasião e a calibração decidida para a ocasião; ganho do direto sobre o produto no log loss ponderado; `padrao` decide; a PR-AUC ponderada é relatada ao lado. **H7:** sobre o que calcula o score (produto, ou direto se H6 o promover), a sigmoide é ajustada em cada dobra externa com o produto interno (`interno[k]` da ocasião vezes `interno[k].extras["celulas"]` da preferência, que estão nas mesmas linhas porque as dobras internas são as mesmas) e aplicada fora da dobra; entra se `superioridade` confirmar o ganho no log loss; a inclinação é relatada com intervalo (`intervalo_estatistica` reajustando `ajustar_sigmoide` em cada réplica), sem decidir. O produto interno usa as partes já com a sigmoide da etapa de calibração quando ela entrou, ajustada nas mesmas previsões internas; a avaliação continua só fora da dobra externa. Grava `h6_h7_produto.json`.
  - `etapa_cortes() -> None`: com a previsão fora da dobra do score final, posições por `posicoes_ponderadas`; `resumo` com o corte de 80; e a sensibilidade da seção 11.1: parcela ponderada das células que muda de quadrante quando cada corte do Decision Engine se move 10 pontos para cima e para baixo. Os cortes ficam os do pré-registro; o JSON só os relata. Grava `cortes.json`.
  - `etapa_receita() -> None`: monta a `Receita` a partir de `h5_peso.json`, `escada_*.json`, `calibracao.json`, `h6_h7_produto.json` e `cortes.json`; os `parametros` de cada parte (e do direto, se H6 o promover) vêm de `selecao_final` do degrau vencedor, com `bases_extras={"celulas": celulas}` na preferência; salva essas previsões em `data/experimentos/selecao_<parte>.joblib`; grava `reports/receita.json`.
- Alvos `make calibracao`, `make produto`, `make cortes`, `make receita`, e `make experimentos` (de `dobras` até `receita`, em ordem).

- [ ] **Passo 1: testes de `receita.py`,** que falham: ida e volta por JSON preserva tudo; `score="direto"` sem `direto` é recusado; `degrau` fora de A a D é recusado.
- [ ] **Passo 2: estender o teste de integração da tarefa 10** até `etapa_receita` na base sintética, conferindo que `reports/receita.json` é gerado e carregado.
- [ ] **Passo 3: implementar e rodar os testes.**
- [ ] **Passo 4: rodar nos dados:** `make calibracao produto cortes receita`. Ler `reports/receita.json` e conferir que cada escolha aponta para o JSON que a sustenta.
- [ ] **Passo 5: commit** do código, de `reports/receita.json` e dos JSONs novos.

---

### Tarefa 12: artefato avaliado montado a partir da receita

**Arquivos:**
- Criar: `src/occasion/inference/treino.py`, `tests/inference/test_treino.py`
- Modificar: `src/occasion/inference/predict.py`, `src/occasion/score/score.py`, `src/occasion/pipeline.py`, `Makefile`, `tests/inference/test_paridade.py`

**Interfaces:**
- Modifica em `occasion.score.score`: `probabilidades(X, modelo_p1, modelo_p2, calibracao_score: tuple[float, float] | None = None, modelo_direto: ModeloProba | None = None) -> pd.DataFrame` e `construir_referencia(celulas, modelo_p1, modelo_p2, calibracao_score=None, modelo_direto=None)`. `p_produto` é `p1 · p2` (ou a previsão do direto), passando pela sigmoide `calibracao_score` quando houver. Chamadas antigas continuam valendo.
- Modifica `ArtefatoModelo`: campos novos com padrão, `calibracao_score: tuple[float, float] | None = None` e `modelo_direto: Any = None`; artefatos antigos, sem esses atributos, continuam carregando (o padrão do dataclass é atributo de classe). `pontuar` e `_top_fatores` passam a usá-los; com `modelo_direto`, os fatores vêm dele, com `parte="direto"`.
- Produz em `occasion.inference.treino`:
  - `ajustar_parte(base: pd.DataFrame, alvo: str, parte: ParteModelo, receita_parte: ReceitaParte, previsao_fora_da_dobra: np.ndarray) -> ModeloParte`: treina com os parâmetros da receita na base inteira; com `calibrar`, ajusta a sigmoide (com `peso_final`) na previsão fora da dobra salva pela tarefa 11 e devolve `ModeloCalibrado`.
  - `treinar_avaliado(receita: Receita, celulas: pd.DataFrame, ocasioes_refri: pd.DataFrame, selecoes: dict[str, Any], commit: str, assinatura: str) -> ArtefatoModelo`: só 2008-2009 (reusa `_exigir_so_treino`); `selecoes` são os `selecao_<parte>.joblib`; `assinatura` vem de `assinatura_da_receita` (tarefa 14) e fica em `ArtefatoModelo.assinatura_receita: str | None = None`; `calibracao_score` ajustada no produto fora da dobra (com as partes já calibradas pela sigmoide ajustada nessas mesmas previsões) quando `receita.calibrar_score`; referência e `ufs_midia_regional` como no Plano 1; `metadados.degrau` descreve o degrau de cada parte (por exemplo, "ocasião: D (LightGBM); preferência: B (logística)").
- Produz `make avaliado`, que grava `models/avaliado/artefato.joblib` a partir de `reports/receita.json`. O `make baseline` continua gravando em `models/baseline/`.
- `assinatura_da_receita` só existe na tarefa 14; até lá, `make avaliado` grava `assinatura_receita=None` e a tarefa 14 completa a ligação.

- [ ] **Passo 1: testes que falham.**

```python
def test_paridade_lote_e_prever_com_logistica_e_lightgbm(artefato_sintetico): ...
    # pontuar(linhas) e prever(contexto) dão o mesmo score, para B e para D

@given(contextos_validos())
def test_saida_respeita_o_contrato(artefato_sintetico, contexto):
    Resultado.model_validate(prever(contexto, artefato_sintetico).model_dump())

def test_score_entre_0_e_100_e_classificacao_coerente(): ...
def test_coluna_que_nao_e_entrada_nao_muda_o_score(): ...       # id_celula, peso_final, edicao
def test_ramo_direto_usa_fatores_do_direto(): ...
def test_artefato_do_plano_1_continua_carregando(tmp_path): ...  # objeto sem os campos novos
def test_calibracao_do_score_nao_muda_as_posicoes(): ...         # transformação monótona
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar `make teste`, `make avaliado` e `make teste-dados`.** Passam.
- [ ] **Passo 5: abrir a tela** com a skill `run` (`make app`), calcular um contexto e conferir que score, fatores (sem `tem_pizza`) e recomendação aparecem e que o painel mostra o degrau novo. Commit.

---

### Tarefa 13: avaliação em 2008-2009: segmentos, erros, explicabilidade e região fora

**Arquivos:**
- Criar: `src/occasion/avaliacao/__init__.py`, `segmentos.py`, `erros.py`, `explicacao.py`, `tests/avaliacao/test_segmentos.py`, `test_erros.py`, `test_explicacao.py`
- Modificar: `src/occasion/experimentos/etapas.py`, `src/occasion/experimentos/__main__.py`, `Makefile`

**Interfaces:**
- `occasion.avaliacao.segmentos`:
  - `MINIMO_POSITIVOS = 10`
  - `colunas_de_segmento(base: pd.DataFrame) -> dict[str, pd.Series]`: `dia_semana`, `faixa_horario`, `regiao`, `faixa_etaria`, `sexo`, `faixa_renda`, `local`.
  - `metricas_por_segmento(y, p, w, segmentos: dict[str, pd.Series]) -> list[dict]`: por segmento e valor, `n`, `positivos`, `log_loss`, `pr_auc`, `o_e`; com menos de `MINIMO_POSITIVOS`, as métricas saem `None` e `motivo = "positivos insuficientes"`.
- `occasion.avaliacao.erros`:
  - `contextos_com_mais_erro(base: pd.DataFrame, y, p, por: list[str], n: int = 10, minimo_linhas: int = 500) -> dict[str, list[dict]]`: agrupa pelo contexto em `por` (padrão: `regiao`, `dia_semana`, `faixa_horario`, `local`), calcula, com peso, `O = Σ w·y`, `E = Σ w·p`, `z = (O − E) / sqrt(Σ w² p(1 − p))` (`w` é `peso_final` normalizado para média 1) e devolve os `n` com maior `z` (`subestimados`) e os `n` com menor (`superestimados`), com as taxas ponderadas observada e prevista.
- `occasion.avaliacao.explicacao`:
  - `importancia_global(modelo: ModeloParte, X, w) -> dict[str, float]`: média ponderada de `|contribuição|` por variável.
  - `estabilidade(modelos: Sequence[ModeloParte], X, w) -> list[dict]`: para cada variável e nível (idade e renda pelas faixas da tarefa 5), a contribuição média em cada modelo de dobra; `instavel = True` se os sinais divergem entre dobras e o maior valor absoluto passa de 0,05 log-chances.
- Em `etapas`: `etapa_avaliacao_2008() -> None` grava `reports/avaliacao_2008.json` com, para a ocasião, a preferência (todas e recorte de pizza) e o score: `resumo`, segmentos, erros, importância global (no modelo final de 2008-2009 inteira) e estabilidade (nos `modelos` de cada dobra do vencedor). Inclui a rodada de região fora (seção 6.4): para cada grande região, a receita treinada nas outras quatro com os parâmetros finais e avaliada nela, ao lado das métricas fora da dobra da mesma região. É análise secundária e não muda a receita. Alvo `make avaliacao-2008`.

- [ ] **Passo 1: testes que falham.**

```python
def test_segmento_sem_positivos_sai_none_sem_excecao():
    linhas = metricas_por_segmento(np.zeros(20), np.full(20, .1), np.ones(20),
                                   {"local": pd.Series(["em_casa"] * 20)})
    assert linhas[0]["pr_auc"] is None and linhas[0]["motivo"] == "positivos insuficientes"

def test_erros_apontam_o_contexto_subestimado(): ...  # contexto com y=1 e p baixo vem primeiro
def test_estabilidade_marca_sinal_trocado(): ...      # dois modelos falsos com sinais opostos
def test_importancia_global_ordena_pela_contribuicao(): ...
```

- [ ] **Passo 2: rodar, ver falhar, implementar, rodar.** Passam.
- [ ] **Passo 3: rodar nos dados:** `make avaliacao-2008`. Commit do código e do JSON.

---

### Tarefa 14: código do teste final, antes da tag

O teste final precisa existir, revisado e testado, antes de a receita ser congelada: a tag cobre também o código que mede.

**Arquivos:**
- Criar: `src/occasion/experimentos/teste_final.py`, `tests/experimentos/test_teste_final.py`
- Modificar: `src/occasion/bases/acesso.py`, `tests/bases/test_acesso.py`, `src/occasion/inference/treino.py`, `src/occasion/pipeline.py`, `src/occasion/experimentos/__main__.py`, `Makefile`

**Interfaces:**
- Produz em `occasion.bases.acesso`:
  - `ARQUIVOS_DA_RECEITA: tuple[str, ...]`: o que define a receita e o modelo avaliado, `src/occasion/config.py`, `src/occasion/features/`, `src/occasion/bases/modelagem.py`, `src/occasion/models/`, `src/occasion/score/`, `src/occasion/validacao/`, `src/occasion/experimentos/especificacao.py`, `src/occasion/experimentos/teste_final.py`, `src/occasion/inference/treino.py`, `src/occasion/inference/predict.py`, `reports/receita.json`, `pyproject.toml` e `uv.lock`. O resto de `src/` (linha de comando, pós-teste, servido, Trends, hipóteses) pode mudar depois da tag.
  - `assinatura_da_receita(raiz: Path = RAIZ, ref: str | None = None) -> str`: SHA-256 do conteúdo dos arquivos rastreados sob `ARQUIVOS_DA_RECEITA`, em ordem de caminho; com `ref`, lê o conteúdo nessa referência do git (`git ls-tree` e `git show`); sem `ref`, lê a árvore de trabalho e levanta `TesteBloqueado` se `git status --porcelain` mostrar qualquer alteração ou arquivo não rastreado sob esses caminhos.
  - `exigir_receita_intacta(raiz: Path = RAIZ) -> str`: exige a tag e que `assinatura_da_receita(raiz) == assinatura_da_receita(raiz, TAG_RECEITA)`; devolve a assinatura. Senão `TesteBloqueado` com o motivo. Comparar a assinatura, e não o commit, deixa o trabalho das tarefas 17 a 20 continuar depois da tag sem abrir brecha na receita.
- Modifica `inference/treino.py` e o comando `make avaliado`: o artefato avaliado guarda `assinatura_receita`.
- Produz em `occasion.experimentos.teste_final`:
  - `rodar_teste_final(raiz: Path = RAIZ) -> Path`: chama `exigir_receita_intacta`; carrega explicitamente `MODELOS / "avaliado" / "artefato.joblib"` (não `carregar_artefato`, que prefere o servido) e confere que `assinatura_receita` do artefato é a assinatura da tag e que `metadados.degrau` corresponde aos degraus de `reports/receita.json`; lê `celulas` e `ocasioes_refri` de 2017-2018 por `carregar_base`; calcula, com posições em relação à referência de 2008-2009 do artefato:
    - **H8:** PR-AUC ponderada do score contra `y_pizza_guarana` nas células, comparada ao produto das duas tabelas de taxas do degrau A da escada, ajustadas aqui mesmo em 2008-2009 com `construir("A", parte, {})` (dimensões do pré-registro) e o uso de peso decidido em H5; intervalo por `intervalo_ganho` com setores de 2017-2018 dentro da UF; veredito por `superioridade`. Se a receita for a tabela nas duas partes, compara com a constante igual à prevalência ponderada de 2008-2009 (seção 7, H8).
    - relatadas à parte, sem entrar na regra (seção 7, H8): a mesma comparação em cada parte separada, com intervalo, e a concentração no décimo superior;
    - `resumo` do score, de cada parte (ocasião contra `y_pizza_refri`; preferência contra `y_guarana` em todas as ocasiões e no recorte de pizza) para a receita, para o degrau A e para a constante;
    - a queda em relação à validação: as mesmas métricas lidas de `escada_*.json`, `h6_h7_produto.json` e `cortes.json`, lado a lado, e por segmento lendo `avaliacao_2008.json`;
    - segmentos e erros com as funções da tarefa 13.
  - Grava `reports/teste_final.json`. Se o arquivo já existir, grava `reports/teste_final_reexecucao.json` e acrescenta `identico_ao_original: bool`, comparando tudo menos `data_hora` e `commit`, sem sobrescrever o original (seção 6.7: rodar de novo é permitido e deve reproduzir).
- Alvo `make teste-final`.

- [ ] **Passo 1: testes que falham,** em repositório git de `tmp_path` com bases e artefato sintéticos:

```python
def test_bloqueia_sem_tag(repo_sem_tag): ...
def test_bloqueia_com_receita_alterada_depois_da_tag(repo_com_tag_e_alteracao_em_models): ...
def test_aceita_alteracao_fora_da_receita_depois_da_tag(repo_com_tag_e_alteracao_no_trends): ...
def test_bloqueia_arquivo_nao_rastreado_na_receita(repo_com_tag_e_arquivo_novo_em_models): ...
def test_bloqueia_artefato_com_outra_assinatura(repo_com_tag, artefato_de_outra_receita): ...
def test_bloqueia_artefato_do_degrau_a_no_lugar_do_avaliado(repo_com_tag, artefato_baseline): ...
def test_segunda_execucao_nao_sobrescreve_e_confere(repo_com_tag, artefato_da_tag):
    rodar_teste_final(repo); rodar_teste_final(repo)
    reexec = json.loads((relatorios / "teste_final_reexecucao.json").read_text())
    assert reexec["identico_ao_original"] is True
```

- [ ] **Passo 2: rodar, ver falhar, implementar, rodar.** Passam. Nenhum dado real de 2017-2018 é lido nesta tarefa.
- [ ] **Passo 3: commit.**

---

### Tarefa 15: congelamento da receita (ponto de parada para o autor)

**Arquivos:**
- Modificar: nenhum código; só a tag.

- [ ] **Passo 1: reprodução.** Apagar `data/experimentos/` e `models/avaliado/`, rodar `make experimentos avaliacao-2008 avaliado` e conferir que `reports/receita.json` e os JSONs saem iguais aos commitados (diferença só em `data_hora` e `commit`). Se algo mudar, investigar antes de seguir. Depois, `git checkout -- reports/` para devolver a árvore limpa.
- [ ] **Passo 2: `make teste-dados` e `make qualidade`.** Passam.
- [ ] **Passo 3: revisão** com `superpowers:requesting-code-review` sobre as tarefas 1 a 14, com o checklist de método do `CLAUDE.md` (vazamento, pesos, target, linguagem, hipóteses commitadas antes das etapas).
- [ ] **Passo 4: parar e mostrar ao autor** a receita, as hipóteses H1 a H7 com veredito e os cortes. Esperar o "aprovado". Ajustes pedidos aqui ainda são permitidos; depois da tag, não.
- [ ] **Passo 5: retreinar o artefato no commit final,** com a árvore limpa (`make avaliado`; `assinatura_da_receita` recusa árvore suja), criar a tag anotada `git tag -a receita-congelada -m "Receita congelada antes do teste final"` e conferir que `assinatura_da_receita()` é igual a `assinatura_da_receita(ref="receita-congelada")` e à do artefato.

---

### Tarefa 16: teste final (H8)

- [ ] **Passo 1: rodar uma vez:** `make teste-final`. Grava `reports/teste_final.json`.
- [ ] **Passo 2: ler e relatar como veio,** incluindo a queda em relação à validação por segmento. A queda é descrita como efeito conjunto do tempo e da mudança de método de coleta (seção 6.7), sem separar os dois.
- [ ] **Passo 3: commit imediato** de `reports/teste_final.json`, com a mensagem "Teste final da receita congelada". Nada da receita muda depois deste ponto.

---

### Tarefa 17: diagnóstico depois do teste e H9

**Arquivos:**
- Criar: `src/occasion/experimentos/pos_teste.py`, `tests/experimentos/test_pos_teste.py`
- Modificar: `src/occasion/bases/acesso.py`, `tests/bases/test_acesso.py`, `src/occasion/experimentos/__main__.py`, `Makefile`

**Interfaces:**
- Produz em `occasion.bases.acesso`: `exigir_teste_commitado(raiz: Path = RAIZ) -> None`: `reports/teste_final.json` está no git (`git ls-files --error-unmatch`) e sem alteração (`git diff --quiet HEAD -- reports/teste_final.json`); senão `TesteBloqueado`. Fica ao lado das outras travas porque as tarefas 18 e 19 também a usam.
- Produz em `occasion.experimentos.pos_teste`:
  - `rodar_pos_teste() -> Path`: dobras de 2017-2018 por `escolher_dobras` (mesmo critério da tarefa 2). Para cada dobra de 2017-2018 e cada parte, a receita com os parâmetros congelados (sem nova busca) é treinada em dois braços: **recente** (só as dobras de treino de 2017-2018) e **base inteira** (as mesmas mais toda 2008-2009). A chave das dobras é `(edicao, cod_upa)`; as linhas de 2008-2009 ficam sempre no treino, nas dobras externas e nas internas, e nunca são avaliadas. Se H6 promoveu o modelo direto, ele passa pelos mesmos dois braços. A calibração, quando a receita a usa, é ajustada nas previsões fora da dobra interna só das linhas de 2017-2018 (mesma regra do modelo servido, seção 12). Avalia na dobra de 2017-2018 que ficou fora. Grava `reports/pos_teste.json` com:
    - **Diagnóstico 6.8:** as métricas do braço recente ao lado das do teste final; a diferença indica quanto se perde ao atravessar dez anos e a troca de método. Não altera nenhum número relatado.
    - **H9:** ganho do braço recente sobre a base inteira no log loss ponderado, nas duas partes e no recorte de pizza; `padrao` decide cada comparação (a base inteira é o padrão) e a regra pré-registrada as junta: se qualquer uma das três recusar o padrão, `treino_servido` fica `"2017-2018"` para as duas partes; senão, `"base_inteira"`.
- Alvo `make pos-teste`.

- [ ] **Passo 1: testes que falham:** `exigir_teste_commitado` bloqueia sem o arquivo e com o arquivo alterado; em base sintética, o braço recente nunca vê linhas de 2008-2009, e a calibração só usa linhas de 2017-2018 (construtor espião).
- [ ] **Passo 2: implementar, rodar os testes, rodar `make pos-teste`.** Commit do código e do JSON.

---

### Tarefa 18: modelo servido e a tela com ele

**Arquivos:**
- Criar: `src/occasion/inference/servido.py`, `tests/inference/test_servido.py`
- Modificar: `src/occasion/pipeline.py`, `Makefile`, e `app/streamlit_app.py` só se o painel não mostrar as métricas. `inference/treino.py` não muda: ele é parte da receita congelada.

**Interfaces:**
- Produz em `occasion.inference.servido`: `treinar_servido(receita: Receita, bases: dict[str, pd.DataFrame], pos_teste: dict, teste_final: dict, commit: str) -> ArtefatoModelo`, com `bases` contendo `celulas` e `ocasioes_refri` das duas edições (chaves `"celulas_2008-2009"` e assim por diante), reusando `ajustar_parte` e `construir_referencia`:
  - chama `exigir_teste_commitado`;
  - treino em 2008-2009 mais 2017-2018 ou só 2017-2018, conforme `pos_teste["h9"]["treino_servido"]`, sem indicador de edição;
  - calibração de cada parte (quando a receita usa) refeita por validação cruzada com o mesmo método, ajustada só nas linhas de 2017-2018 (dobras de 2017-2018 de `pos_teste`; linhas de 2008-2009 sempre no treino, como na tarefa 17);
  - `calibracao_score` (H7) refeita do mesmo jeito, no produto fora da dobra das linhas de 2017-2018; e o modelo direto, se H6 o promoveu, reajustado no mesmo treino;
  - população de referência e `ufs_midia_regional` em `celulas` de 2017-2018;
  - `metadados`: `modelo="servido"`, `treinado_em` conforme H9, `metricas_da_receita` lidas de `teste_final.json` (PR-AUC, log loss, ROC-AUC e Brier do score no teste, e a PR-AUC do degrau A), `descricao_metricas` dizendo que são as métricas da receita treinada em 2008-2009 e testada uma vez em 2017-2018, e que o modelo servido não tem teste próprio.
- Alvo `make servido`, que grava `models/servido/artefato.joblib`. `carregar_artefato` já prefere `servido`.

- [ ] **Passo 1: testes que falham:** sem `teste_final.json` commitado, `treinar_servido` levanta `TesteBloqueado`; com H9 dizendo `"2017-2018"`, nenhuma linha de 2008-2009 entra no treino; `metricas_da_receita` é igual aos valores do JSON de teste; a referência tem só linhas de 2017-2018.
- [ ] **Passo 2: implementar, rodar os testes, `make servido`.**
- [ ] **Passo 3: abrir a tela de verdade** com a skill `run`: calcular dois contextos (em casa e fora de casa), conferir que o painel diz "servido", mostra em que dados foi treinado e as métricas do teste temporal, e que os números da tela batem com `reports/teste_final.json`. Guardar uma captura em `reports/capturas/`.
- [ ] **Passo 4: commit.**

---

### Tarefa 19: Google Trends (H10)

**Ação do autor, que pode ser feita a qualquer momento antes desta tarefa:** exportar do site do Google Trends três CSVs de interesse por sub-região (estado), Brasil, pesquisa na web, todas as categorias, com os tópicos "Guaraná Antarctica" e "Coca-Cola" na mesma consulta: de 20/05/2008 a 22/05/2009, de 01/06/2017 a 31/07/2018 e os últimos 12 meses. Guardar em `dados_externos/google_trends/` com nomes `trends_2008-2009.csv`, `trends_2017-2018.csv`, `trends_recente.csv`, e anotar em `dados_externos/google_trends/LEIAME.md` a data da extração, os termos, os filtros e qualquer aviso que o site mostrar.

**Arquivos:**
- Criar: `src/occasion/experimentos/trends.py`, `tests/experimentos/test_trends.py`, `tests/exemplos/trends_exemplo.csv`
- Modificar: `src/occasion/experimentos/__main__.py`, `Makefile`

**Interfaces:**
- Produz em `occasion.experimentos.trends`:
  - `ler_trends(caminho: Path) -> pd.DataFrame`: colunas `uf` (sigla), `guarana`, `cola`. Pula as linhas de cabeçalho do arquivo até a linha de títulos; aceita nomes de estado em português e em inglês, com e sem acento (`"São Paulo"`, `"State of Sao Paulo"`, `"Federal District"`, `"Distrito Federal"`); aceita valores com `%` (a comparação por sub-região do site divide cada estado em porcentagens que somam 100) e sem `%`; `"<1"` vira 0,5; vazio vira `NaN`.
  - `participacao_trends(df) -> pd.Series`: `guarana / (guarana + cola)` por UF; UF sem dado fica `NaN` e é contada.
  - `participacao_pof(ocasioes_refri: pd.DataFrame) -> pd.Series`: participação ponderada de `y_guarana` por UF.
  - `correlacao_por_uf(trends: pd.Series, pof: pd.Series) -> Intervalo`: correlação de Spearman nas UFs com dado nas duas fontes; intervalo por `intervalo_por_unidades` sobre as UFs.
  - `rodar_trends() -> Path`: as duas edições (a de 2017-2018 só roda se `exigir_teste_commitado` passar; antes disso grava só 2008-2009); `correlacao_positiva` decide H10; a janela recente entra só como participação de interesse de busca por UF, sem comparação com a POF. Grava `reports/trends.json` com a frase fixa "interesse de busca, não consumo".
- Alvo `make trends`.

- [ ] **Passo 1: testes que falham:** o CSV de exemplo é montado a partir do primeiro arquivo real do autor (mesmo cabeçalho e formato, valores trocados), com `"<1"` e uma UF vazia, e é lido com 27 UFs; valores com `%` e sem `%` dão a mesma participação; nomes em inglês e português dão a mesma sigla; Spearman de duas séries iguais dá 1; sem o teste commitado, `rodar_trends` grava só 2008-2009.
- [ ] **Passo 2: implementar e rodar os testes.**
- [ ] **Passo 3: com os CSVs do autor, `make trends`.** Commit do código, dos CSVs, do LEIAME e do JSON.

---

### Tarefa 20: relatório de hipóteses e fechamento do plano

**Arquivos:**
- Criar: `src/occasion/experimentos/hipoteses.py`, `tests/experimentos/test_hipoteses.py`
- Modificar: `src/occasion/experimentos/__main__.py`, `Makefile`, `CLAUDE.md`

**Interfaces:**
- Produz em `occasion.experimentos.hipoteses`: `consolidar(pasta: Path = RELATORIOS) -> dict` e `rodar_hipoteses() -> Path`, que grava `reports/hipoteses.json` com uma entrada por hipótese H1 a H10: `id`, `enunciado`, `teste`, `metrica`, `estimativa`, `ic_inf`, `ic_sup`, `largura`, `veredito`, `consequencia` (o que mudou na receita) e `fonte` (o JSON de origem). O veredito é copiado do JSON da etapa, nunca recalculado aqui. H1 a H3 aparecem uma vez por parte. Hipótese cujo JSON não existe aparece com `veredito = "não executada"` e o motivo.
- Alvo `make hipoteses`.

- [ ] **Passo 1: testes que falham,** com JSONs de etapa falsos em `tmp_path`: cada hipótese aponta para o arquivo de origem; hipótese ausente sai como "não executada"; H5, H6 e H9 usam "sem evidência contra o padrão" e nunca "confirmada".
- [ ] **Passo 2: implementar, rodar os testes, `make hipoteses`.**
- [ ] **Passo 3: rodar tudo do zero.** Apagar `data/interim`, `data/processed`, `data/experimentos` e `models/`; rodar `make tudo experimentos avaliacao-2008 avaliado teste-final pos-teste servido trends hipoteses`. O teste final grava `teste_final_reexecucao.json` com `identico_ao_original: true`. Depois `make teste-dados` e `make qualidade`. Tudo passa.
- [ ] **Passo 4: `CLAUDE.md`** com `/revise-claude-md`: comandos novos do `Makefile`, a tag, as travas da edição de teste, armadilhas encontradas nas bibliotecas.
- [ ] **Passo 5: revisão** com `superpowers:requesting-code-review` sobre o plano inteiro, com o checklist de método.
- [ ] **Passo 6: commit.**

---

## Ordem de corte

O autor disse que há tempo; esta seção só existe para o caso de o prazo apertar. Vale a ordem da seção 18.2 do design, sem gatilho de data: degrau C (a escada vai de B a D, e H2 fica "não executada"), depois o bloco do Trends (tarefa 19), depois H9 e o reajuste (tarefas 17 e 18). Se H9 for cortada, vale o plano B da seção 12: o modelo servido é o próprio avaliado, copiado para `models/servido/` com `model_metadata` dizendo que não houve reajuste. O diagnóstico 6.8 sai junto com H9. Os entregáveis obrigatórios do TAPI não entram na lista.

## Ordem e paralelismo

```
1 ─┬─ 2 ─┐
   ├─ 3 ─┼─ 4 ─ 5 ─ 6 ─┬─ 7 ─ 8 ─ 9 ─ 10 ─ 11 ─ 12 ─ 13 ─ 14 ─ 15 ─ 16 ─ 17 ─ 18 ─ 20
   └─────┘             └─ (autor exporta os CSV do Trends) ─────────────────── 19 ─┘
```

- As tarefas 2 e 3 podem rodar em paralelo (arquivos diferentes). O resto da espinha é sequencial: cada etapa usa o resultado gravado da anterior, na ordem da seção 8.3 do design.
- As tarefas 6 e 15 são pontos de parada: o executor espera a aprovação do autor.
- A tarefa 19 roda depois da 17 para ter as duas edições; a metade de 2008-2009 pode rodar antes, se os CSVs chegarem cedo.
- Worktrees só nas tarefas paralelas.
