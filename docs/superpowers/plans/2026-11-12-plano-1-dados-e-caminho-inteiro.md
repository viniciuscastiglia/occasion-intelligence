# Plano 1: dados e caminho inteiro com o baseline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** ter, até 15/11/2026, o caminho inteiro funcionando com o degrau A (tabela de taxas): dos microdados da POF até uma tela que recebe um contexto e devolve score, fatores e recomendação.

**Arquitetura:** um pacote Python (`occasion`) com camadas separadas: ingestão, bases, variáveis, modelos, score, decisão e inferência. A tela Streamlit chama a função de inferência direto. Os modelos melhores entram no Plano 2 sem mudar as interfaces deste plano.

**Stack:** Python 3.12 fixado pelo `uv`, pandas, pyarrow, pandera, scikit-learn, pydantic, streamlit, pytest, hypothesis, ruff e pyright.

**Spec:** `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md`. O `CLAUDE.md`, seção "Fonte de dados: POF/IBGE", faz parte da spec.

**Planos seguintes,** escritos depois que este rodar, porque dependem do que os dados mostrarem:

- **Plano 2:** EDA em 2008-2009, hipóteses commitadas, escada de modelos, calibração, teste final, diagnóstico, modelo servido e bloco do Trends.
- **Plano 3:** mapa de calor, briefing por LLM, API, Data Card, README e vídeo.

## Restrições globais

- Código, comentários, nomes de colunas e documentação em português. Funções públicas com anotações de tipo. `pyright` e `ruff` sem erros antes de fechar cada tarefa.
- Antes de usar uma biblioteca (pandas, pandera, scikit-learn, pydantic, streamlit, hypothesis), consultar a documentação da versão instalada pelo servidor `context7`. Enviar só o nome da biblioteca e a dúvida técnica.
- O target é uma medida observada na POF. Nada do target de 2017-2018 é calculado neste plano, exceto as contagens de referência usadas como teste do parser.
- Nada de 2017-2018 entra no ajuste de modelo ou de pré-processamento. O degrau A é ajustado só em 2008-2009.
- Estimativas populacionais usam `peso_final`.
- Nenhum texto afirma causalidade nem apresenta "guaraná" como a marca Guaraná Antarctica.
- Arquivos de largura fixa em codificação latin1. Os zips do IBGE são extraídos com `zipfile` do Python.
- É proibido teste que afirma um limiar de métrica ajustado para passar.
- Os microdados ficam em `data/`, fora do git. Testes que precisam deles levam a marca `dados` e são pulados quando `data/raw` não existe.
- Semente fixa: `SEMENTE = 20261112`.
- Um commit por tarefa, com a mensagem em português e a linha final `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>`.

## Foco de revisão

Situações que a spec implica e que mais podem morder quem usar o sistema. Cada uma tem um teste na tarefa dona do código.

1. **Ocasião com hora ausente ou local "não determinado"** (código 9 em 2017-2018): não pode virar negativo em silêncio. É contada e relatada. Teste na tarefa 7.
2. **Pessoa do consumo sem linha no registro de moradores:** a junção falha com erro claro, sem descartar linhas. Teste na tarefa 6.
3. **Contexto que cai em célula da tabela sem nenhum caso no treino:** a previsão volta para a taxa geral, sem `NaN` nem exceção. Teste na tarefa 10.
4. **Guaraná e cola na mesma ocasião, ou o mesmo alimento registrado duas vezes na mesma hora:** as marcações são sim ou não por ocasião, e o target não é contado em dobro. Teste na tarefa 6.
5. **Artefato de modelo ausente quando a tela abre:** a tela mostra uma instrução de como gerá-lo, sem rastro de erro. Teste na tarefa 14.

---

## Estrutura de arquivos

```
pyproject.toml  .python-version  Makefile
src/occasion/
  config.py              constantes: caminhos, edições, códigos de alimento, faixas, cortes
  ingest/download.py     baixar, conferir checksums e extrair
  ingest/layouts.py      posições dos campos das duas edições
  ingest/parse.py        leitura dos arquivos de largura fixa para o esquema comum
  ingest/schema.py       esquemas pandera do esquema comum
  bases/ocasioes.py      ocasiões e base pessoa x dia
  bases/modelagem.py     base da parte 1 (células) e base da parte 2
  features/variaveis.py  renda em salários mínimos, região e colunas de modelo
  audit/auditoria.py     disponibilidade, tamanho de amostra e validação adversarial
  models/taxas.py        degrau A
  score/score.py         produto, posição e classificação
  decision/engine.py     regras do Decision Engine
  inference/contract.py  contrato de entrada e saída
  inference/predict.py   artefato e função de inferência
  pipeline.py            comandos de linha: bases, auditoria, baseline
app/streamlit_app.py
tests/                   espelha src/occasion
docs/ficha-de-vazamento.md
reports/                 saídas geradas, fora do git exceto os JSON de auditoria
```

---

### Tarefa 1: esqueleto do projeto

**Arquivos:**
- Criar: `pyproject.toml`, `.python-version`, `Makefile`, `src/occasion/__init__.py`, `src/occasion/config.py`, `tests/test_config.py`
- Modificar: `.gitignore`

**Interfaces:**
- Produz, em `occasion.config`:
  - `RAIZ: Path`, `DADOS_BRUTOS: Path` (`data/raw`), `DADOS_INTERIM: Path` (`data/interim`), `DADOS_PROCESSADOS: Path` (`data/processed`), `MODELOS: Path` (`models`), `RELATORIOS: Path` (`reports`)
  - `EDICOES: tuple[str, str] = ("2008-2009", "2017-2018")`, `EDICAO_TREINO = "2008-2009"`, `EDICAO_TESTE = "2017-2018"`
  - `SEMENTE: int = 20261112`
  - `FAIXAS_HORARIO: dict[str, range]` = `madrugada` 0 a 5, `manha` 6 a 10, `almoco` 11 a 14, `tarde` 15 a 17, `noite` 18 a 23
  - `LOCAIS: tuple[str, str] = ("em_casa", "fora_de_casa")`
  - `SALARIO_MINIMO: dict[str, float] = {"2008-2009": 415.0, "2017-2018": 954.0}`
  - `faixa_da_hora(hora: int) -> str`
- Comandos do `Makefile`: `ambiente`, `teste`, `teste-dados`, `qualidade`.

- [ ] **Passo 1: criar o ambiente**

`.python-version` com `3.12`. `pyproject.toml` com o pacote em `src/occasion`, dependências `pandas`, `pyarrow`, `pandera`, `scikit-learn`, `pydantic`, `streamlit`, `joblib` e grupo de desenvolvimento `pytest`, `hypothesis`, `ruff`, `pyright`. Instalar com `uv sync`; o `uv.lock` é commitado. Registrar em `[tool.pytest.ini_options]` a marca `dados`.

- [ ] **Passo 2: escrever o teste que falha**

```python
from occasion import config

def test_faixas_cobrem_as_24_horas_sem_sobreposicao():
    horas = [h for faixa in config.FAIXAS_HORARIO.values() for h in faixa]
    assert sorted(horas) == list(range(24))

def test_faixa_da_hora():
    assert config.faixa_da_hora(0) == "madrugada"
    assert config.faixa_da_hora(20) == "noite"
    assert config.faixa_da_hora(11) == "almoco"
```

- [ ] **Passo 3: rodar e ver falhar.** `uv run pytest tests/test_config.py -v` falha com `ModuleNotFoundError`.
- [ ] **Passo 4: implementar `config.py`** com as constantes acima.
- [ ] **Passo 5: `Makefile`.** `ambiente` roda `uv sync`; `teste` roda `uv run pytest -m "not dados"`; `teste-dados` roda `uv run pytest`; `qualidade` roda `uv run ruff check . && uv run pyright`.
- [ ] **Passo 6: rodar.** `make teste` e `make qualidade` passam.
- [ ] **Passo 7: acrescentar ao `.gitignore`:** `data/`, `models/`, `reports/*` com exceção de `reports/auditoria*.json`. Commit.

**Sobre o salário mínimo:** os valores 415 e 954 são os vigentes em 15/01/2009 e 15/01/2018, datas de referência das edições, citados de memória. Conferir as duas datas na documentação do IBGE e os dois valores em fonte oficial. Se algum divergir, usar o oficial e anotar a fonte em um comentário.

---

### Tarefa 2: download e extração

**Arquivos:**
- Criar: `src/occasion/ingest/download.py`, `tests/ingest/test_download.py`
- Modificar: `Makefile` (alvo `dados-brutos`)

**Interfaces:**
- Produz:
  - `URLS: dict[str, dict[str, str]]`, com as chaves `dados` e `documentacao` por edição, usando os endereços do `CLAUDE.md`
  - `extrair_zip(caminho_zip: Path, destino: Path) -> list[Path]`
  - `baixar_edicao(edicao: str, destino: Path = DADOS_BRUTOS) -> Path`, que devolve `destino/<edicao>/`
  - arquivo `data/raw/checksums.json` com o SHA-256 de cada zip

- [ ] **Passo 1: teste que falha**, com um zip criado no próprio teste contendo um arquivo de nome acentuado:

```python
def test_extrair_zip_preserva_nomes_com_acento(tmp_path):
    zip_path = tmp_path / "doc.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("Dicionários de váriaveis.xls", b"x")
    arquivos = extrair_zip(zip_path, tmp_path / "saida")
    assert [a.name for a in arquivos] == ["Dicionários de váriaveis.xls"]

def test_extrair_zip_recusa_caminho_fora_do_destino(tmp_path):
    zip_path = tmp_path / "mal.zip"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("../fora.txt", b"x")
    with pytest.raises(ValueError):
        extrair_zip(zip_path, tmp_path / "saida")
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** Os nomes nos zips do IBGE vêm em cp850 marcados como cp437; decodificar com `nome.encode("cp437").decode("cp850")` quando o bit de UTF-8 não estiver ligado. `baixar_edicao` pula o download se o zip já existe com o checksum registrado.
- [ ] **Passo 4: rodar os testes.** Passam.
- [ ] **Passo 5: baixar de verdade.** `make dados-brutos` baixa as duas edições (cerca de 250 MB). Conferir que existem `data/raw/2017-2018/CONSUMO_ALIMENTAR.txt` e `MORADOR.txt`, e `data/raw/2008-2009/T_CONSUMO_S.txt` e o arquivo de moradores da edição. Se os nomes forem outros, registrar os nomes reais em `ingest/layouts.py` na tarefa 4 ou 5.
- [ ] **Passo 6: commit.**

---

### Tarefa 3: esquema comum

**Arquivos:**
- Criar: `src/occasion/ingest/schema.py`, `tests/ingest/test_schema.py`

**Interfaces:**
- Produz dois esquemas pandera, `ESQUEMA_CONSUMO` e `ESQUEMA_MORADOR`, e as funções `validar_consumo(df) -> DataFrame` e `validar_morador(df) -> DataFrame`.

Colunas de `consumo`, uma linha por alimento consumido:

| Coluna | Tipo | Regra |
|---|---|---|
| `edicao` | str | uma das duas edições |
| `uf` | int | código IBGE, de 11 a 53 |
| `cod_upa` | str | setor; não vazio |
| `estrato` | str | `"<uf>-<ESTRATO_POF>"` |
| `num_dom`, `num_uc`, `cod_informante` | int | maiores que zero |
| `id_pessoa` | str | `"<edicao>|<cod_upa>|<num_dom>|<num_uc>|<cod_informante>"` |
| `quadro` | int | 71 ou 72 em 2008-2009; 72 ou 73 em 2017-2018 |
| `id_dia` | str | `"<id_pessoa>|<quadro>"` |
| `cod_alimento` | int | sete dígitos |
| `hora` | Int64, aceita nulo | de 0 a 23 |
| `local` | str | `em_casa`, `fora_de_casa` ou `indeterminado` |
| `dia_semana` | int | 1 segunda a 7 domingo |
| `peso_final` | float | maior que zero |
| `renda_total_uc` | float | maior ou igual a zero |

Colunas de `morador`, uma linha por pessoa:

| Coluna | Tipo | Regra |
|---|---|---|
| `edicao`, `uf`, `cod_upa`, `estrato`, `num_dom`, `num_uc`, `cod_informante`, `id_pessoa` | como acima | `id_pessoa` é única |
| `situacao` | str | `urbano` ou `rural` |
| `idade` | int | de 0 a 130 |
| `sexo` | str | `masculino` ou `feminino` |
| `n_moradores_uc` | int | número de linhas de morador na mesma unidade de consumo; maior que zero |

- [ ] **Passo 1: testes que falham.** Um quadro mínimo válido passa em cada esquema; `dia_semana = 8`, `local = "rua"`, `peso_final = 0` e `id_pessoa` repetida no morador falham com erro de esquema.
- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar** os dois esquemas com validação estrita de colunas.
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 4: parser da POF 2017-2018

Pode rodar em paralelo com a tarefa 5.

**Arquivos:**
- Criar: `src/occasion/ingest/layouts.py` (parte de 2017-2018), `src/occasion/ingest/parse.py`, `tests/ingest/test_parse_2017.py`

**Interfaces:**
- Consome: `validar_consumo`, `validar_morador` (tarefa 3).
- Produz:
  - `ler_largura_fixa(caminho: Path, campos: dict[str, tuple[int, int]]) -> DataFrame`, com posição inicial em base 1 e tamanho, tudo como texto, latin1
  - `parse_consumo_2017(caminho: Path) -> DataFrame` e `parse_morador_2017(caminho: Path) -> DataFrame`, no esquema comum
  - arquivos `data/interim/consumo_2017-2018.parquet` e `morador_2017-2018.parquet`

**Posições (base 1, tamanho):**

- `CONSUMO_ALIMENTAR.txt`: as da tabela do `CLAUDE.md`.
- `MORADOR.txt`: `UF` 1 (2), `ESTRATO_POF` 3 (4), `TIPO_SITUACAO_REG` 7 (1), `COD_UPA` 8 (9), `NUM_DOM` 17 (2), `NUM_UC` 19 (1), `COD_INFORMANTE` 20 (2), `V0403` idade 33 (3), `V0404` sexo 36 (1).

**Conversões:**

- `hora`: `V9015` vem como `20H`; retirar o `H`. Vazio vira nulo.
- `local`: `V9018` igual a 1 vira `em_casa`; de 2 a 7, `fora_de_casa`; 9 ou vazio, `indeterminado`.
- `dia_semana`: texto por extenso; segunda-feira vira 1 e domingo vira 7. Comparar sem acento e sem diferenciar maiúsculas.
- `situacao`: 1 urbano, 2 rural.
- `sexo`: conferir as categorias de `V0404` no dicionário da edição antes de mapear.

- [ ] **Passo 1: testes unitários que falham,** com duas linhas de largura fixa montadas no teste:

```python
def test_parse_consumo_2017_converte_campos(tmp_path):
    # linha montada com UF=35, quadro=72, V9001=8500903, V9015="20H", V9018="1", DIA_SEMANA="Sábado"
    df = parse_consumo_2017(arquivo_de_teste)
    linha = df.iloc[0]
    assert (linha.uf, linha.quadro, linha.cod_alimento) == (35, 72, 8500903)
    assert (linha.hora, linha.local, linha.dia_semana) == (20, "em_casa", 6)

def test_local_9_vira_indeterminado(tmp_path): ...
def test_hora_vazia_vira_nulo(tmp_path): ...
```

- [ ] **Passo 2: teste com os dados reais que falha:**

```python
@pytest.mark.dados
def test_dias_pessoa_2017():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet")
    assert df["id_dia"].nunique() == 85_018

@pytest.mark.dados
def test_soma_dos_pesos_2017_e_plausivel():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet")
    total = df.drop_duplicates("id_pessoa")["peso_final"].sum()
    assert 160e6 < total < 200e6

@pytest.mark.dados
def test_domingos_e_sabados_2017():
    dias = pd.read_parquet(DADOS_INTERIM / "consumo_2017-2018.parquet").drop_duplicates("id_dia")
    assert (dias["dia_semana"] == 7).sum() == 16_585
    assert (dias["dia_semana"] == 6).sum() == 3_862
```

- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar.** `n_moradores_uc` é a contagem de linhas do `MORADOR.txt` por `cod_upa`, `num_dom` e `num_uc`. A faixa de plausibilidade da soma dos pesos serve para pegar erro de casas decimais.
- [ ] **Passo 5: gerar os parquets e rodar todos os testes.** Passam. Se uma contagem divergir, usar `superpowers:systematic-debugging`; os primeiros suspeitos são posição de campo e codificação.
- [ ] **Passo 6: commit.**

---

### Tarefa 5: parser da POF 2008-2009

Pode rodar em paralelo com a tarefa 4.

**Arquivos:**
- Criar ou estender: `src/occasion/ingest/layouts.py` (parte de 2008-2009), `src/occasion/ingest/parse.py`, `tests/ingest/test_parse_2008.py`

**Interfaces:**
- Consome: `ler_largura_fixa` (se a tarefa 4 ainda não a criou, criar com a mesma assinatura), `validar_consumo`, `validar_morador`.
- Produz: `parse_consumo_2008(caminho: Path) -> DataFrame`, `parse_morador_2008(caminho: Path) -> DataFrame` e os arquivos `data/interim/consumo_2008-2009.parquet` e `morador_2008-2009.parquet`.

**Posições, lidas do dicionário da edição (base 1, tamanho):**

| Campo | `T_CONSUMO_S.txt` | Registro de moradores |
|---|---|---|
| `COD_UF` | 3 (2) | 3 (2) |
| `NUM_SEQ` | 5 (3) | 5 (3) |
| `NUM_DV` | 8 (1) | 8 (1) |
| `NUM_DOM` | 9 (2) | 9 (2) |
| `NUM_UC` | 11 (1) | 11 (1) |
| `COD_INFORMANTE` | 12 (2) | 12 (2) |
| `ESTRATO_POF` | 14 (2) | 14 (2) |
| `PESO_FINAL` | 31 (15), 8 decimais | não usado |
| `NUM_QUADRO` | 46 (2) | |
| `V9018` fonte do alimento | 48 (1) | |
| `V9015` horário | 49 (2) | |
| `V9001` alimento | 64 (7) | |
| `RENDA_TOTAL` | 115 (16), 2 decimais | |
| `DIA_SEMANA` | 152 (1) | |
| `IDADE_ANOS` | | 60 (3) |
| `V0405` sexo | | 76 (2) |
| `COD_UPA` | | 247 (6) |
| `TIPO_SITUACAO_REG` | | 253 (1) |

**Conversões:**

- `cod_upa`: `COD_UF + NUM_SEQ + NUM_DV`, com zeros à esquerda. Deve ser igual ao `COD_UPA` do registro de moradores.
- `hora`: inteiro de 0 a 23. Aceitar também o formato com `H`.
- `local`: 1 vira `em_casa`, 2 vira `fora_de_casa`.
- `dia_semana`: já é 1 segunda a 7 domingo.
- `sexo`: 1 masculino, 2 feminino. `situacao`: 1 urbano, 2 rural, vinda do registro de moradores.
- **Decimais:** se o texto do campo tiver ponto, converter direto; se não tiver, dividir por 10 elevado ao número de decimais. O teste da soma dos pesos confirma.

**Limite a registrar:** em 2008-2009 o campo é "fonte do alimento" (dentro ou fora do domicílio); em 2017-2018 é "local de refeição". Anotar essa diferença em um comentário do layout; ela vai para a auditoria e para o Data Card.

- [ ] **Passo 1: testes unitários que falham,** no mesmo formato da tarefa 4, com uma linha montada: `COD_UF=35`, `NUM_SEQ=012`, `NUM_DV=3`, quadro 71, `V9001=8500903`, `V9015=20`, `V9018=2`, `DIA_SEMANA=6`. Afirmar `cod_upa == "350123"`, `hora == 20`, `local == "fora_de_casa"`, `dia_semana == 6`.
- [ ] **Passo 2: testes com os dados reais que falham:**

```python
@pytest.mark.dados
def test_dias_pessoa_2008():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    assert df["id_dia"].nunique() == 66_903

@pytest.mark.dados
def test_soma_dos_pesos_2008_e_plausivel():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    assert 140e6 < df.drop_duplicates("id_pessoa")["peso_final"].sum() < 180e6

@pytest.mark.dados
def test_quadros_2008():
    df = pd.read_parquet(DADOS_INTERIM / "consumo_2008-2009.parquet")
    assert set(df["quadro"].unique()) == {71, 72}
```

- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar e gerar os parquets.**
- [ ] **Passo 5: rodar.** Passam. Commit.

---

### Tarefa 6: ocasiões e base pessoa x dia

**Arquivos:**
- Criar: `src/occasion/bases/ocasioes.py`, `tests/bases/test_ocasioes.py`
- Modificar: `src/occasion/config.py` (códigos de alimento)

**Interfaces:**
- Consome: os quatro parquets de `data/interim`.
- Produz, em `config`: `CODIGOS: dict[str, dict[str, frozenset[int]]]`, com as chaves `pizza`, `guarana`, `cola` e `outros_refri` por edição. `refri` é a união das três últimas.
- Produz, em `bases.ocasioes`:
  - `montar_ocasioes(consumo: DataFrame, edicao: str) -> DataFrame`, uma linha por `id_dia` e `hora` não nula, com `id_ocasiao` (`"<id_dia>|<hora>"`), `id_dia`, `id_pessoa`, `hora`, `local`, `tem_pizza`, `tem_refri`, `tem_guarana`, `tem_cola` (booleanos)
  - `montar_pessoa_dia(consumo: DataFrame, morador: DataFrame, ocasioes: DataFrame) -> DataFrame`, uma linha por `id_dia`, com as colunas do morador, `dia_semana`, `peso_final`, `renda_total_uc`, `tem_pizza_dia`, `tem_pizza_refri`, `tem_pizza_guarana`
  - arquivos `data/processed/ocasioes_<edicao>.parquet` e `pessoa_dia_<edicao>.parquet`

**Regras:**

- **Local da ocasião:** o do item de pizza, se houver pizza; senão o do primeiro item de refrigerante; senão o do primeiro item.
- **Itens com hora nula** não formam ocasião. `montar_ocasioes` devolve em `attrs["itens_sem_hora"]` quantos eram.
- **Junção com o morador:** se alguma `id_pessoa` do consumo não existir no morador, levantar `ValueError` com a contagem. Nada é descartado em silêncio.

**Códigos de 2017-2018:** os do `CLAUDE.md`.

**Códigos de 2008-2009,** candidatos lidos do cadastro de produtos da edição:

- **Pizza:** `8500903`, `8500914` a `8500917`, `8508003`, `8508014` a `8508017`. Em dúvida: `6502403` e `6506003` (mini pizza semipronta).
- **Guaraná:** `8200301`, `8200302`, `8201601`, `8201602`, `8207401`, `8207402`.
- **Cola:** `8200101`, `8200102`, `8200701`, `8200702`, `8200801`.
- **Outros refrigerantes:** `8200407`, `8203501`, `8205402`, `8207601`, `8209102`, `8212906`, mais o que o cadastro listar como refrigerante na faixa de `8200101` a `8212999` sem a palavra "refrigerante" no nome.

A lista final de 2008-2009 é a que reproduz as três contagens de referência. Registrar em comentário quais códigos em dúvida entraram e por quê.

- [ ] **Passo 1: testes unitários que falham:**

```python
def test_pizza_e_guarana_na_mesma_hora_marcam_a_ocasiao():
    consumo = quadro_de_consumo([  # auxiliar do teste: (hora, cod_alimento, local)
        (20, 8500903, "em_casa"), (20, 8200301, "em_casa"), (12, 8200101, "fora_de_casa")])
    oc = montar_ocasioes(consumo, "2017-2018").set_index("hora")
    assert oc.loc[20, ["tem_pizza", "tem_refri", "tem_guarana", "tem_cola"]].tolist() == [True, True, True, False]
    assert oc.loc[12, ["tem_pizza", "tem_refri", "tem_cola"]].tolist() == [False, True, True]

def test_guarana_e_cola_juntos_e_item_repetido_nao_contam_em_dobro():
    consumo = quadro_de_consumo([(20, 8200301, "em_casa"), (20, 8200301, "em_casa"), (20, 8200101, "em_casa")])
    oc = montar_ocasioes(consumo, "2017-2018")
    assert len(oc) == 1
    assert oc.iloc[0][["tem_guarana", "tem_cola"]].tolist() == [True, True]

def test_pizza_e_refri_em_horas_diferentes_nao_e_ocasiao_conjunta(): ...
def test_local_da_ocasiao_e_o_do_item_de_pizza(): ...
def test_pessoa_sem_morador_levanta_erro(): ...   # pytest.raises(ValueError)
def test_itens_sem_hora_sao_contados_e_nao_formam_ocasiao():
    oc = montar_ocasioes(quadro_de_consumo([(None, 8500903, "em_casa"), (20, 8200301, "em_casa")]), "2017-2018")
    assert len(oc) == 1 and oc.attrs["itens_sem_hora"] == 1
```

- [ ] **Passo 2: testes com os números de referência, que falham:**

```python
REFERENCIA = {"2017-2018": (85_018, 1_681, 783, 176), "2008-2009": (66_903, 1_149, 635, 101)}

@pytest.mark.dados
@pytest.mark.parametrize("edicao", ["2017-2018", "2008-2009"])
def test_numeros_de_referencia(edicao):
    pd_ = pd.read_parquet(DADOS_PROCESSADOS / f"pessoa_dia_{edicao}.parquet")
    obtido = (len(pd_), pd_.tem_pizza_dia.sum(), pd_.tem_pizza_refri.sum(), pd_.tem_pizza_guarana.sum())
    assert obtido == REFERENCIA[edicao]

@pytest.mark.dados
def test_ocasioes_com_refrigerante_2017():
    oc = pd.read_parquet(DADOS_PROCESSADOS / "ocasioes_2017-2018.parquet")
    assert oc.tem_refri.sum() == 15_863
```

- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar** e gerar os parquets das duas edições.
- [ ] **Passo 5: rodar.** Passam nas duas edições. Divergência em 2008-2009 se resolve ajustando a lista de códigos em dúvida, nunca o número de referência.
- [ ] **Passo 6: commit.**

---

### Tarefa 7: bases de modelagem e variáveis

**Arquivos:**
- Criar: `src/occasion/features/variaveis.py`, `src/occasion/bases/modelagem.py`, `tests/features/test_variaveis.py`, `tests/bases/test_modelagem.py`, `docs/ficha-de-vazamento.md`

**Interfaces:**
- Consome: `pessoa_dia_<edicao>.parquet` e `ocasioes_<edicao>.parquet`.
- Produz, em `features.variaveis`:
  - `COLUNAS_CONTEXTO: list[str] = ["dia_semana", "faixa_horario", "local", "uf", "regiao", "situacao", "idade", "sexo", "renda_pc_sm", "n_moradores_uc"]`
  - `regiao_da_uf(uf: int) -> str`, pelo primeiro dígito: 1 `norte`, 2 `nordeste`, 3 `sudeste`, 4 `sul`, 5 `centro_oeste`
  - `renda_pc_sm(renda_total_uc: Series, n_moradores_uc: Series, edicao: str) -> Series`
- Produz, em `bases.modelagem`:
  - `montar_celulas(pessoa_dia: DataFrame, ocasioes: DataFrame) -> DataFrame`: uma linha por `id_dia`, faixa e local, com `id_celula`, `id_dia`, `id_pessoa`, `edicao`, `cod_upa`, `estrato`, `peso_final`, as `COLUNAS_CONTEXTO`, `y_pizza_refri` e `y_pizza_guarana` (0 ou 1). Em `attrs["dias_positivos_excluidos"]`, quantos dias com pizza e refrigerante ficaram sem célula por local indeterminado.
  - `montar_ocasioes_refri(pessoa_dia: DataFrame, ocasioes: DataFrame) -> DataFrame`: uma linha por ocasião com refrigerante e local determinado, com `id_ocasiao`, as mesmas colunas de identificação e de contexto, `tem_pizza` e `y_guarana` (0 ou 1). Em `attrs["ocasioes_excluidas"]`, quantas saíram.
  - arquivos `data/processed/celulas_<edicao>.parquet` e `ocasioes_refri_<edicao>.parquet`

- [ ] **Passo 1: testes unitários que falham:**

```python
def test_cada_dia_tem_dez_celulas():
    celulas = montar_celulas(pessoa_dia_minimo(n_dias=3), ocasioes_vazias())
    assert len(celulas) == 3 * 10
    assert celulas.groupby("id_dia").size().eq(10).all()

def test_ocasiao_positiva_marca_uma_unica_celula():
    # pizza e refrigerante às 20h, em casa
    celulas = montar_celulas(pd_1_dia, ocasiao(hora=20, local="em_casa", pizza=True, refri=True, guarana=False))
    positivas = celulas[celulas.y_pizza_refri == 1]
    assert positivas[["faixa_horario", "local"]].values.tolist() == [["noite", "em_casa"]]
    assert celulas.y_pizza_guarana.sum() == 0

def test_ocasiao_positiva_com_local_indeterminado_e_contada_e_nao_some():
    celulas = montar_celulas(pd_1_dia, ocasiao(hora=20, local="indeterminado", pizza=True, refri=True))
    assert celulas.y_pizza_refri.sum() == 0
    assert celulas.attrs["dias_positivos_excluidos"] == 1

def test_renda_pc_sm():
    r = renda_pc_sm(pd.Series([1660.0]), pd.Series([4]), "2008-2009")
    assert r.iloc[0] == pytest.approx(1.0)

def test_regiao_da_uf():
    assert [regiao_da_uf(u) for u in (11, 29, 35, 43, 53)] == ["norte", "nordeste", "sudeste", "sul", "centro_oeste"]
```

- [ ] **Passo 2: testes com os dados reais que falham:**

```python
@pytest.mark.dados
@pytest.mark.parametrize("edicao,dias,positivos", [("2008-2009", 66_903, 635), ("2017-2018", 85_018, 783)])
def test_celulas_fecham_com_a_referencia(edicao, dias, positivos):
    c = pd.read_parquet(DADOS_PROCESSADOS / f"celulas_{edicao}.parquet")
    assert len(c) == dias * 10
    dias_pos = c.loc[c.y_pizza_refri == 1, "id_dia"].nunique()
    assert dias_pos + ler_excluidos(edicao) == positivos   # ler_excluidos lê reports/bases.json
```

- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar.** As contagens de exclusão são gravadas em `reports/bases.json`.
- [ ] **Passo 5: escrever `docs/ficha-de-vazamento.md`:** uma linha por coluna de `COLUNAS_CONTEXTO` e por `tem_pizza`, com o argumento de por que é conhecida no instante de previsão. Para `tem_pizza`, usar o argumento da seção 5.1 da spec. Para `renda_pc_sm`, o da seção 5.4.
- [ ] **Passo 6: gerar as bases e rodar.** Passam. Commit.

---

### Tarefa 8: auditoria dos dados

**Arquivos:**
- Criar: `src/occasion/audit/auditoria.py`, `tests/audit/test_auditoria.py`
- Modificar: `src/occasion/pipeline.py` (comando `auditoria`), `Makefile`

**Interfaces:**
- Consome: as bases de `data/processed`.
- Produz:
  - `r2_cs_maximo(n: int, eventos: int) -> float`, o maior R² de Cox-Snell possível para a prevalência observada
  - `parametros_sustentados(n: int, r2_cs: float, encolhimento: float = 0.9) -> int`, critério de encolhimento de Riley e colegas
  - `validacao_adversarial(treino: DataFrame, teste: DataFrame, colunas: list[str], semente: int) -> dict`, com as chaves `auc` e `importancias` (variável e valor)
  - `rodar_auditoria() -> dict`, que grava `reports/auditoria.json`

**Conteúdo de `reports/auditoria.json`:**

1. Para cada coluna de contexto, a presença e a fração de nulos em cada edição.
2. Contagens de exclusão das tarefas 6 e 7, por edição.
3. Positivos de 2008-2009 em cada parte e no recorte de pizza da parte 2, com o número de parâmetros sustentados.
4. AUC da validação adversarial e as variáveis que mais separam as edições, na base da parte 1 e na da parte 2.
5. Resposta à pergunta "algum registro informa o período da coleta?", com o nome do campo, se existir.
6. A diferença de definição do campo de local entre as edições.

**Regras:**

- A validação adversarial usa só `COLUNAS_CONTEXTO`. Nenhuma coluna `y_` nem `tem_pizza` entra.
- Nenhuma estatística do target de 2017-2018 é calculada. O item 3 usa só 2008-2009.
- A auditoria é relato. Nada nela altera variáveis ou bases.

**Fórmulas,** citadas de memória; conferir no artigo de Riley e colegas antes de implementar e citar a fonte no código:

- Critério de encolhimento: `n = p / ((S - 1) * ln(1 - r2_cs / S))`, com `S` igual ao encolhimento. Resolvida para `p`: `p = n * (S - 1) * ln(1 - r2_cs / S)`, arredondada para baixo.
- R² máximo: `1 - exp(2 * lnL0 / n)`, com `lnL0 = E * ln(E / n) + (n - E) * ln(1 - E / n)` e `E` igual ao número de eventos.
- Como o R² do modelo não é conhecido de antemão, a auditoria usa 15% do R² máximo, que é a sugestão dos autores quando não há estudo anterior.

- [ ] **Passo 1: testes que falham:**

```python
def test_parametros_sustentados_valor_calculado_a_mao():
    # 10000 * (0.9 - 1) * ln(1 - 0.02 / 0.9) = 22,47
    assert parametros_sustentados(10_000, 0.02) == 22

def test_parametros_sustentados_cresce_com_n():
    assert parametros_sustentados(20_000, 0.02) > parametros_sustentados(10_000, 0.02)

def test_r2_cs_maximo_com_prevalencia_de_meio():
    # lnL0 = n * ln(0.5); 1 - exp(2 * ln(0.5)) = 0.75
    assert r2_cs_maximo(1_000, 500) == pytest.approx(0.75)

def test_validacao_adversarial_distingue_bases_diferentes_e_nao_bases_iguais():
    iguais = validacao_adversarial(amostra_a, amostra_a_outra_metade, ["idade"], SEMENTE)
    diferentes = validacao_adversarial(jovens, idosos, ["idade"], SEMENTE)
    assert diferentes["auc"] > 0.95
    assert abs(iguais["auc"] - 0.5) < 0.1

def test_validacao_adversarial_recusa_coluna_de_target():
    with pytest.raises(ValueError):
        validacao_adversarial(a, b, ["idade", "y_pizza_refri"], SEMENTE)
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** Classificador da validação adversarial: `HistGradientBoostingClassifier` do scikit-learn, AUC por validação cruzada de 5 dobras, importância por permutação.
- [ ] **Passo 4: rodar os testes e `make auditoria`.** O arquivo `reports/auditoria.json` é gerado e commitado.
- [ ] **Passo 5: commit.**

---

### Tarefa 9: contrato de entrada e saída

Depende só da tarefa 1. Pode rodar em paralelo com as tarefas 2 a 8.

**Arquivos:**
- Criar: `src/occasion/inference/contract.py`, `tests/inference/test_contract.py`, `tests/exemplos/resultado_exemplo.json`

**Interfaces:**
- Produz, em Pydantic:

```python
UF = Literal["RO","AC","AM","RR","PA","AP","TO","MA","PI","CE","RN","PB","PE","AL","SE","BA",
             "MG","ES","RJ","SP","PR","SC","RS","MS","MT","GO","DF"]
DiaSemana = Literal["segunda","terca","quarta","quinta","sexta","sabado","domingo"]
FaixaHorario = Literal["madrugada","manha","almoco","tarde","noite"]
Local = Literal["em_casa","fora_de_casa"]
Situacao = Literal["urbano","rural"]
Sexo = Literal["masculino","feminino"]
FaixaRenda = Literal["ate_meio_sm","meio_a_1_sm","de_1_a_2_sm","de_2_a_5_sm","mais_de_5_sm"]
Classificacao = Literal["alta","média","baixa"]

class Contexto(BaseModel):
    uf: UF; situacao: Situacao; dia_semana: DiaSemana; faixa_horario: FaixaHorario; local: Local
    idade: int = Field(ge=18, le=110); sexo: Sexo; faixa_renda: FaixaRenda
    numero_moradores: int = Field(ge=1, le=20); versao_contrato: str = "1.0"

class ComponenteScore(BaseModel): posicao: float; probabilidade: float
class Fator(BaseModel):
    parte: Literal["ocasiao","preferencia","direto"]; variavel: str; valor: str
    direcao: Literal["aumenta","diminui"]; contribuicao: float
class MetadadosModelo(BaseModel):
    modelo: Literal["avaliado","servido"]; degrau: str; treinado_em: list[str]; receita_commit: str
    metricas_da_receita: dict[str, float] | None; descricao_metricas: str
    instante_de_previsao: str; aviso: str; versao_contrato: str
class Resultado(BaseModel):
    occasion_score: float = Field(ge=0, le=100); classification: Classificacao
    occasion_probability: float = Field(ge=0, le=1)
    score_components: dict[Literal["ocasiao","preferencia"], ComponenteScore]
    top_factors: list[Fator]; recommendation_category: str; recommendation_reason: str
    model_metadata: MetadadosModelo
```

- `PONTO_MEDIO_RENDA: dict[FaixaRenda, float]` = 0,25; 0,75; 1,5; 3,5; 7,5 salários mínimos por pessoa.
- `CODIGO_UF: dict[UF, int]` com os códigos do IBGE.

As faixas de renda são um ponto de partida com classes usuais do IBGE; o Plano 2 pode revê-las na EDA de 2008-2009.

- [ ] **Passo 1: testes que falham:**

```python
def test_idade_abaixo_de_18_e_recusada():
    with pytest.raises(ValidationError):
        Contexto(**{**CONTEXTO_VALIDO, "idade": 17})

def test_resultado_exemplo_respeita_o_contrato():
    Resultado.model_validate_json((EXEMPLOS / "resultado_exemplo.json").read_text())

def test_saida_tem_os_seis_campos_do_tapi():
    campos = set(Resultado.model_fields)
    assert {"occasion_score", "classification", "top_factors", "recommendation_category",
            "recommendation_reason", "model_metadata"} <= campos

def test_codigo_uf_cobre_as_27():
    assert len(CODIGO_UF) == 27 and CODIGO_UF["SP"] == 35 and CODIGO_UF["DF"] == 53
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar** e escrever `resultado_exemplo.json` com valores ilustrativos. Esse arquivo é o exemplo fixo contra o qual as tarefas 12 e 14 podem trabalhar em paralelo.
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 10: degrau A, a tabela de taxas

**Arquivos:**
- Criar: `src/occasion/models/taxas.py`, `tests/models/test_taxas.py`

**Interfaces:**
- Produz:

```python
class TabelaDeTaxas:
    def __init__(self, dimensoes: list[str], forca_suavizacao: float = 50.0) -> None: ...
    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight: pd.Series) -> "TabelaDeTaxas": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...        # vetor de probabilidades, uma por linha
    def contribuicoes(self, X: pd.DataFrame) -> pd.DataFrame: ...      # uma coluna por dimensão, em log-chances
    taxa_geral_: float
```

- `DIMENSOES_PARTE_1 = ["dia_semana", "faixa_horario", "local"]`
- `DIMENSOES_PARTE_2 = ["regiao", "local"]`

**Regras:**

- Os pesos são normalizados pela média antes de tudo.
- Taxa de uma combinação: `(soma dos pesos dos positivos + m × taxa geral) / (soma dos pesos + m)`, com `m = forca_suavizacao`.
- Combinação que não apareceu no treino recebe a taxa geral.
- `contribuicoes`: para cada dimensão, log-chances da taxa marginal suavizada daquele valor menos log-chances da taxa geral.
- A parte 2 não usa `tem_pizza` na tabela; com poucos casos de pizza, a tabela ficaria vazia. Esse é um limite do degrau A, anotado no relatório.

- [ ] **Passo 1: testes que falham:**

```python
def test_taxa_ponderada_sem_suavizacao():
    X = pd.DataFrame({"local": ["em_casa"] * 4 + ["fora_de_casa"] * 4})
    y = pd.Series([1, 0, 0, 0, 1, 1, 0, 0]); w = pd.Series([1.0] * 8)
    t = TabelaDeTaxas(["local"], forca_suavizacao=0.0).fit(X, y, w)
    p = t.predict_proba(pd.DataFrame({"local": ["em_casa", "fora_de_casa"]}))
    assert p.tolist() == pytest.approx([0.25, 0.5])

def test_peso_amostral_muda_a_taxa():
    X = pd.DataFrame({"local": ["em_casa", "em_casa"]}); y = pd.Series([1, 0])
    t = TabelaDeTaxas(["local"], 0.0).fit(X, y, pd.Series([3.0, 1.0]))
    assert t.predict_proba(X)[0] == pytest.approx(0.75)

def test_combinacao_nao_vista_recebe_a_taxa_geral():
    t = TabelaDeTaxas(["local"], 50.0).fit(X_so_em_casa, y, w)
    p = t.predict_proba(pd.DataFrame({"local": ["fora_de_casa"]}))
    assert p[0] == pytest.approx(t.taxa_geral_) and np.isfinite(p).all()

def test_suavizacao_puxa_celula_pequena_para_a_taxa_geral(): ...
def test_contribuicoes_tem_uma_coluna_por_dimensao_e_sinal_coerente(): ...
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 11: score

**Arquivos:**
- Criar: `src/occasion/score/score.py`, `tests/score/test_score.py`

**Interfaces:**
- Consome: qualquer modelo com `predict_proba(X) -> np.ndarray` (tarefa 10) e o tipo `Classificacao` (tarefa 9).
- Produz:

```python
CORTE_ALTA = 80.0
CORTE_MEDIA = 50.0

@dataclass(frozen=True)
class ReferenciaScore:
    ocasiao: np.ndarray; preferencia: np.ndarray; produto: np.ndarray   # valores ordenados
    peso_acumulado: dict[str, np.ndarray]                               # mesma ordem, de 0 a 1
    def posicao(self, valor: float, qual: Literal["ocasiao", "preferencia", "produto"]) -> float: ...

def probabilidades(X: pd.DataFrame, modelo_p1, modelo_p2) -> pd.DataFrame: ...   # colunas p_ocasiao, p_preferencia, p_produto
def construir_referencia(celulas: pd.DataFrame, modelo_p1, modelo_p2) -> ReferenciaScore: ...
def classificar(score: float) -> Classificacao: ...
```

**Regras:**

- `probabilidades` calcula `p_preferencia` com `tem_pizza = 1` em todas as linhas e `p_produto = p_ocasiao × p_preferencia`.
- `posicao` devolve, de 0 a 100, a fração ponderada da população de referência com valor menor ou igual ao informado.
- `construir_referencia` usa todas as linhas da base de células e `peso_final`.
- `classificar`: `alta` a partir de 80; `média` de 50 até abaixo de 80; `baixa` abaixo de 50.

- [ ] **Passo 1: testes que falham:**

```python
def test_posicao_e_fracao_ponderada():
    ref = referencia_de(valores=[0.1, 0.2, 0.3, 0.4], pesos=[1, 1, 1, 1])
    assert ref.posicao(0.2, "produto") == pytest.approx(50.0)
    assert ref.posicao(0.05, "produto") == 0.0
    assert ref.posicao(0.9, "produto") == 100.0

def test_posicao_respeita_o_peso():
    ref = referencia_de(valores=[0.1, 0.2], pesos=[3, 1])
    assert ref.posicao(0.1, "produto") == pytest.approx(75.0)

def test_produto_e_a_multiplicacao_e_usa_tem_pizza_igual_a_um():
    p = probabilidades(X, ModeloFixo(0.02), ModeloQueExigeTemPizza())
    assert (p.p_produto == p.p_ocasiao * p.p_preferencia).all()

@pytest.mark.parametrize("score,esperado", [(80.0, "alta"), (79.9, "média"), (50.0, "média"), (49.9, "baixa")])
def test_classificar(score, esperado):
    assert classificar(score) == esperado
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.**
- [ ] **Passo 4: rodar.** Passam. Commit.

---

### Tarefa 12: Decision Engine

Depende só da tarefa 9. Pode rodar em paralelo com as tarefas 10 e 11.

**Arquivos:**
- Criar: `src/occasion/decision/engine.py`, `tests/decision/test_engine.py`

**Interfaces:**
- Consome: `Local`, `Situacao`, `UF`, `Fator` (tarefa 9).
- Produz:

```python
CORTE_OCASIAO = 75.0
CORTE_PREFERENCIA = 50.0
IDADE_JOVEM = 30

Objetivo = Literal["Sustentar", "Converter", "Criar ocasião", "Não priorizar"]
Canal = Literal["influencers e conteúdo digital", "ação na pizzaria",
                "combo em app de delivery", "pizza e guaraná juntos no varejo"]

@dataclass(frozen=True)
class Decisao:
    objetivo: Objetivo; canal_principal: Canal | None; canais_complementares: tuple[str, ...]
    categoria: str; regra: str

def decidir(posicao_ocasiao: float, posicao_preferencia: float, local: Local, situacao: Situacao,
            idade: int, uf: UF, ufs_midia_regional: frozenset[str]) -> Decisao: ...
def justificar(decisao: Decisao, fatores: list[Fator]) -> str: ...
```

**Regras, na ordem da seção 11 da spec:**

- **Objetivo:** ocasião alta se `posicao_ocasiao >= 75`; preferência alta se `posicao_preferencia >= 50`. Alta e alta: Sustentar. Alta e baixa: Converter. Baixa e alta: Criar ocasião. Baixa e baixa: Não priorizar.
- **Canal principal,** primeira regra que se aplicar:
  1. Não priorizar: nenhum.
  2. Converter e `idade < 30`: influencers e conteúdo digital.
  3. `fora_de_casa`: ação na pizzaria.
  4. `em_casa` e `urbano`: combo em app de delivery.
  5. `em_casa` e `rural`: pizza e guaraná juntos no varejo.
- **Canal complementar:** `"mídia regional"` quando a UF está em `ufs_midia_regional` e o objetivo não é Não priorizar.
- **`categoria`:** `"<objetivo>, via <canal>"`, ou `"Não priorizar"`.
- **`regra`:** identificador da regra que disparou, como `"canal-4"`.
- **`justificar`:** texto por modelo fixo que cita o objetivo, a regra e até três fatores. Não usa as palavras "aumenta as vendas", "vende mais", "causa" nem "Guaraná Antarctica". A frase final é fixa: "A recomendação indica que a ativação combina com o contexto; ela não estima o efeito da ação."

- [ ] **Passo 1: testes de exemplo que falham:**

```python
def test_converter_jovem_vai_para_influencers():
    d = decidir(90, 20, "em_casa", "urbano", 24, "SP", frozenset())
    assert (d.objetivo, d.canal_principal, d.regra) == ("Converter", "influencers e conteúdo digital", "canal-2")
    assert d.categoria == "Converter, via influencers e conteúdo digital"

def test_sustentar_em_casa_urbano_vai_para_delivery():
    d = decidir(90, 80, "em_casa", "urbano", 24, "SP", frozenset())
    assert (d.objetivo, d.canal_principal) == ("Sustentar", "combo em app de delivery")

def test_rural_em_casa_vai_para_varejo(): ...
def test_fora_de_casa_vai_para_pizzaria(): ...
def test_nao_priorizar_nao_tem_canal_nem_midia_regional():
    d = decidir(10, 10, "em_casa", "urbano", 40, "BA", frozenset({"BA"}))
    assert (d.canal_principal, d.canais_complementares, d.categoria) == (None, (), "Não priorizar")

@pytest.mark.parametrize("oc,pref,obj", [(75, 50, "Sustentar"), (74.9, 50, "Criar ocasião"),
                                         (75, 49.9, "Converter"), (74.9, 49.9, "Não priorizar")])
def test_cortes_sao_inclusivos(oc, pref, obj): ...
```

- [ ] **Passo 2: testes gerados com Hypothesis, que falham:**

```python
@given(oc=st.floats(0, 100), pref=st.floats(0, 100), local=st.sampled_from(LOCAIS),
       situacao=st.sampled_from(["urbano", "rural"]), idade=st.integers(18, 110), uf=st.sampled_from(UFS))
def test_toda_entrada_valida_cai_em_exatamente_uma_categoria(oc, pref, local, situacao, idade, uf):
    d = decidir(oc, pref, local, situacao, idade, uf, frozenset({"BA", "PE"}))
    assert d.categoria in CATEGORIAS_POSSIVEIS          # as onze categorias, listadas no teste
    assert (d.canal_principal is None) == (d.objetivo == "Não priorizar")

@given(...)
def test_justificativa_nao_usa_termos_proibidos(...):
    texto = justificar(d, fatores)
    assert not any(t in texto.lower() for t in ["vende mais", "aumenta as vendas", "causa", "guaraná antarctica"])
```

- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar.**
- [ ] **Passo 5: rodar.** Passam. Commit.

---

### Tarefa 13: artefato e função de inferência

**Arquivos:**
- Criar: `src/occasion/inference/predict.py`, `tests/inference/test_predict.py`
- Modificar: `src/occasion/pipeline.py` (comando `baseline`), `Makefile` (alvos `bases`, `baseline`, `tudo`)

**Interfaces:**
- Consome: tarefas 9 a 12 e as bases de 2008-2009.
- Produz:

```python
N_UFS_MIDIA_REGIONAL = 7

@dataclass
class ArtefatoModelo:
    modelo_p1: Any; modelo_p2: Any; referencia: ReferenciaScore
    ufs_midia_regional: frozenset[str]; metadados: MetadadosModelo

def treinar_baseline(celulas: pd.DataFrame, ocasioes_refri: pd.DataFrame, commit: str) -> ArtefatoModelo: ...
def salvar_artefato(artefato: ArtefatoModelo, pasta: Path) -> Path: ...     # pasta/artefato.joblib
def carregar_artefato(pasta: Path | None = None) -> ArtefatoModelo: ...     # models/servido, senão models/avaliado
def contexto_para_linha(contexto: Contexto) -> pd.DataFrame: ...            # uma linha com COLUNAS_CONTEXTO
def pontuar(X: pd.DataFrame, artefato: ArtefatoModelo) -> pd.DataFrame: ... # p_ocasiao, p_preferencia, p_produto, posicao_*, score
def prever(contexto: Contexto, artefato: ArtefatoModelo) -> Resultado: ...

class ArtefatoAusente(FileNotFoundError): ...
```

**Regras:**

- `treinar_baseline` ajusta uma `TabelaDeTaxas(DIMENSOES_PARTE_1)` em `celulas` com o target `y_pizza_refri` e uma `TabelaDeTaxas(DIMENSOES_PARTE_2)` em `ocasioes_refri` com o target `y_guarana`, as duas com `peso_final`. Depois constrói a referência com `construir_referencia(celulas, ...)`.
- `treinar_baseline` recebe só dados de 2008-2009 e levanta `ValueError` se encontrar outra edição na coluna `edicao`.
- `contexto_para_linha` leva `numero_moradores` para a coluna `n_moradores_uc`.
- `ufs_midia_regional`: as 7 UFs com maior média ponderada de `p_preferencia` na população de referência.
- `metadados` do baseline: `modelo="avaliado"`, `degrau="A (tabela de taxas)"`, `treinado_em=["2008-2009"]`, `metricas_da_receita=None`, `descricao_metricas="Ainda não medidas: o teste temporal acontece depois que a receita for congelada."`, `instante_de_previsao="planejamento, antes da ocasião"`, `aviso="Guaraná é o sabor registrado na POF, não uma marca."`.
- `contexto_para_linha` converte a sigla da UF em código, o dia em número, a faixa de renda no ponto médio e calcula a região.
- `top_factors`: a partir de `contribuicoes` dos dois modelos, os cinco de maior valor absoluto, com `parte` `ocasiao` ou `preferencia`.
- `prever` e o processamento em lote passam os dois por `pontuar`.
- `carregar_artefato` levanta `ArtefatoAusente` com a mensagem "Modelo não encontrado. Rode `make baseline`." quando nenhuma pasta existe.

- [ ] **Passo 1: testes que falham,** com bases sintéticas pequenas geradas no teste:

```python
def test_prever_devolve_resultado_valido(artefato_sintetico):
    r = prever(Contexto(**CONTEXTO_VALIDO), artefato_sintetico)
    assert 0 <= r.occasion_score <= 100
    assert r.classification == classificar(r.occasion_score)
    assert r.occasion_probability == pytest.approx(
        r.score_components["ocasiao"].probabilidade * r.score_components["preferencia"].probabilidade)

def test_inferencia_e_lote_dao_o_mesmo_score(artefato_sintetico):
    ctx = Contexto(**CONTEXTO_VALIDO)
    lote = pontuar(pd.concat([contexto_para_linha(ctx)] * 3, ignore_index=True), artefato_sintetico)
    assert lote["score"].nunique() == 1
    assert prever(ctx, artefato_sintetico).occasion_score == pytest.approx(lote["score"].iloc[0])

def test_campo_que_nao_e_entrada_do_modelo_nao_altera_o_score(artefato_sintetico):
    a = prever(Contexto(**CONTEXTO_VALIDO), artefato_sintetico)
    b = prever(Contexto(**{**CONTEXTO_VALIDO, "versao_contrato": "9.9"}), artefato_sintetico)
    assert a.occasion_score == b.occasion_score

def test_treinar_baseline_recusa_edicao_de_teste(celulas_com_2017, ocasioes_refri_2008):
    with pytest.raises(ValueError):
        treinar_baseline(celulas_com_2017, ocasioes_refri_2008, "abc")

def test_artefato_ausente_tem_mensagem_clara(tmp_path):
    with pytest.raises(ArtefatoAusente, match="make baseline"):
        carregar_artefato(tmp_path / "nao_existe")

def test_treinar_duas_vezes_da_o_mesmo_artefato(celulas_2008_sintetica, ocasioes_refri_2008):
    a = treinar_baseline(celulas_2008_sintetica, ocasioes_refri_2008, "abc")
    b = treinar_baseline(celulas_2008_sintetica, ocasioes_refri_2008, "abc")
    assert np.array_equal(a.referencia.produto, b.referencia.produto)
```

- [ ] **Passo 2: teste gerado com Hypothesis, que falha:** para qualquer `Contexto` válido, `prever` devolve um `Resultado` que passa em `Resultado.model_validate`.
- [ ] **Passo 3: rodar e ver falhar.**
- [ ] **Passo 4: implementar.**
- [ ] **Passo 5: comandos.** `make bases` roda as tarefas 4 a 7 na ordem; `make baseline` treina em 2008-2009 e grava `models/avaliado/artefato.joblib`; `make tudo` roda `dados-brutos`, `bases`, `auditoria` e `baseline`.
- [ ] **Passo 6: rodar os testes e `make baseline`.** Passam, e o artefato é gerado. Commit.

---

### Tarefa 14: tela

**Arquivos:**
- Criar: `app/streamlit_app.py`, `tests/app/test_app.py`
- Modificar: `Makefile` (alvo `app`)

**Interfaces:**
- Consome: `Contexto`, `prever`, `carregar_artefato`, `ArtefatoAusente`.
- Produz: `make app`, que roda `uv run streamlit run app/streamlit_app.py`.

**Conteúdo da tela:**

1. **Formulário,** na barra lateral: os campos de `Contexto`, com rótulos em português, e o botão "Calcular".
2. **Resultado:** o score em destaque, a classificação, a posição nas duas partes, os fatores em barras horizontais com a direção indicada, a categoria recomendada e a justificativa.
3. **Sobre o modelo:** o conteúdo de `model_metadata` e os limites da fonte: sabor e não marca, sem data, sem canal e base de 2008-2009 e 2017-2018.
4. **JSON do contrato,** em uma área recolhível.

Carregar a skill `dataviz` antes de escrever o gráfico de fatores. O mapa de calor e o briefing ficam para o Plano 3.

- [ ] **Passo 1: testes que falham,** com a ferramenta de teste de aplicações do Streamlit (conferir a interface no `context7`):

```python
def test_tela_mostra_score_e_recomendacao(artefato_em_disco):
    at = AppTest.from_file("app/streamlit_app.py").run()
    at.sidebar.button[0].click().run()
    assert not at.exception
    texto = " ".join(m.value for m in at.markdown)
    assert "Recomendação" in texto and "Sobre o modelo" in texto

def test_tela_sem_artefato_mostra_instrucao_e_nao_quebra(sem_artefato):
    at = AppTest.from_file("app/streamlit_app.py").run()
    assert not at.exception
    assert any("make baseline" in e.value for e in at.error)
```

- [ ] **Passo 2: rodar e ver falhar.**
- [ ] **Passo 3: implementar.** A pasta do artefato vem da variável de ambiente `OCCASION_MODELOS`, com `models/` como padrão, para os testes apontarem para uma pasta temporária.
- [ ] **Passo 4: rodar os testes.** Passam.
- [ ] **Passo 5: abrir e usar de verdade,** com a skill `run`: `make app`, preencher um contexto, calcular, e conferir score, fatores, recomendação e o painel do modelo. Trocar o local de "em casa" para "fora de casa" e conferir que a recomendação muda de canal. Guardar uma captura de tela em `reports/`.
- [ ] **Passo 6: commit.**

---

### Tarefa 15: fechamento do plano

**Arquivos:**
- Modificar: `CLAUDE.md`

- [ ] **Passo 1: rodar tudo do zero.** Apagar `data/interim`, `data/processed` e `models/`, rodar `make tudo` e depois `make teste-dados` e `make qualidade`. Tudo passa, e os números de referência são reproduzidos.
- [ ] **Passo 2: registrar no `CLAUDE.md`** os comandos reais do `Makefile`, a lista final de códigos de alimento de 2008-2009 e as armadilhas encontradas, com `/revise-claude-md`.
- [ ] **Passo 3: revisão.** Usar `superpowers:requesting-code-review` com o checklist de método do `CLAUDE.md`: vazamento, pesos, target, linguagem e LLM.
- [ ] **Passo 4: commit.**

---

## Ordem e paralelismo

```
1 ─┬─ 2 ─ 3 ─┬─ 4 ─┬─ 6 ─ 7 ─ 8 ─────────┐
   │         └─ 5 ─┘                     ├─ 13 ─ 14 ─ 15
   └─ 9 ─┬─ 10 ─ 11 ─────────────────────┤
         └─ 12 ──────────────────────────┘
```

- As tarefas 4 e 5 rodam em paralelo.
- A tarefa 9 pode começar logo depois da 1. As tarefas 10 e 11 e a tarefa 12 rodam em paralelo depois dela.
- Worktrees só nas tarefas paralelas.
