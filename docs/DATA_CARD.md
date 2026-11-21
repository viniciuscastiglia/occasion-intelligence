# Data Card

Occasion Intelligence: pizza with guaraná soda. Modelling bases derived from the personal food intake block of the Brazilian Household Budget Survey (Pesquisa de Orçamentos Familiares, POF), run by IBGE, editions 2008-2009 and 2017-2018.

This card describes the data the models are trained and tested on: where it comes from, how it is built, what each column means, how leakage is prevented, who is covered and what the data cannot say. Model results and their discussion are in the [README](../README.md); the only results repeated here are the out-of-fold metrics of the training edition, for reference.

The count and metric tables are generated from the JSON files in `reports/` and sit between `BEGIN`/`END` markers. `make docs` rewrites them, and `tests/relatorios/test_docs_atualizados.py` fails if they are stale or if the prose cites a decimal or percentage that is not in one of them. Counts quoted in the prose name their source where they are cited.

## At a glance

| Item | Answer |
|---|---|
| Ground truth | Observed food intake reported in the POF personal food intake block. Not a proxy. |
| Unit of analysis | Part 1 (occasion): person x day x cell, where a cell is a time band combined with a place (at home, away from home). Part 2 (preference): soda occasion, that is, person x day x hour with at least one soda. |
| Targets | Part 1: pizza and soda in the same hour within the cell. Part 2: the soda occasion includes guaraná-flavored soda. Score check: pizza and guaraná soda in the same hour within the cell. |
| Prediction instant | When an activation is planned, before the occasion happens. |
| Training edition | 2008-2009 (development: EDA, feature choice, models, cutoffs). |
| Test edition | 2017-2018 (one final test with the frozen recipe; afterwards, the served model is refitted on both editions). |
| Weights | `PESO_FINAL`, the calibrated survey weight, in every population estimate and metric, and in model training. |
| Population | Residents aged 10 or older who answered the personal food intake block, all 27 federative units, urban and rural. |
| Not in the data | Brand, calendar date (in the consumption record), purchase channel, delivery. |

## 1. Sources and versions

Both editions come from IBGE's public FTP server. `make dados-brutos` downloads the zips, records their SHA-256 in `data/raw/checksums.json`, and extracts them with Python's `zipfile` (the documentation zips have file names in a legacy encoding that the macOS `unzip` cannot read). Nothing under `data/` is committed.

| Edition | Role | Data zip | Documentation zip | Folder |
|---|---|---|---|---|
| 2008-2009 | Training | `Dados_20231009.zip` | `Documentacao_20231009.zip` | <https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2008_2009/Microdados/> |
| 2017-2018 | Test, then served-model refit | `Dados_20230713.zip` | `Documentacao_20230713.zip` | <https://ftp.ibge.gov.br/Orcamentos_Familiares/Pesquisa_de_Orcamentos_Familiares_2017_2018/Microdados/> |

Files read from each edition:

| Content | 2008-2009 | 2017-2018 |
|---|---|---|
| Personal food intake (one row per food item eaten by a person on a day) | `T_CONSUMO_S.txt` | `CONSUMO_ALIMENTAR.txt` |
| Residents (age, sex, urban or rural) | `T_MORADOR_S.txt` | `MORADOR.txt` |
| Field layout (spelled this way in both zips) | `Dicionários de váriaveis.xls` | `Dicionários de váriaveis.xls` |
| Food codes | `Cadastro de Produtos do Consumo Alimentar.xls` | `Cadastro de Produtos do Consumo Alimentar.xls` |

Both are fixed-width text files in latin1. Field positions, checked against each edition's dictionary, are in `src/occasion/ingest/layouts.py`. The two editions have different layouts; the parsers map both to one common schema validated with pandera (`src/occasion/ingest/schema.py`).

Collection periods: in 2008-2009 the recorded consumption days run from 20 May 2008 to 22 May 2009 (`reports/investigacao_datas.json`); in 2017-2018 collection ran from June 2017 to July 2018 (IBGE). Only these two editions measure individual food intake: the 1987-1988, 1995-1996 and 2002-2003 editions measured household food purchases, and the 2024-2025 edition had no published microdata when this project was built.

One external constant enters the data: the nominal minimum wage on each edition's reference date, R$ 415 on 15 January 2009 and R$ 954 on 15 January 2018, as cited in IBGE's publications for each edition (`SALARIO_MINIMO` in `src/occasion/config.py`).

## 2. Unit of analysis and target

### Occasion

An occasion is all items eaten by the same person, on the same recorded day, in the same whole hour (field `V9015`). "Pizza with soda" means both in the same occasion. This key, day + hour, is the same in both editions (decision R12 in `docs/registro-de-decisoes.md`, section 25): the 2017-2018 file also has a meal-type field (`V9017`), but 2008-2009 does not, so it is not part of the key. Flags are yes/no per occasion, so an item repeated within the hour does not count twice.

The place of an occasion is the place of its pizza item if there is one; otherwise that of its first soda item; otherwise that of its first item (`src/occasion/bases/ocasioes.py`).

### What counts as pizza, soda and guaraná

Soda is defined by product, generic or branded, with the same rule in both editions (decision R13): flavors recorded under a brand name count as that flavor or as other soda. "Guaraná" always means **soda of guaraná flavor as recorded by the survey**. For guaraná the POF records the flavor ("refrigerante de guaraná"), not a brand.

| Group | 2008-2009 codes | 2017-2018 codes |
|---|---|---|
| Pizza | 8500903, 8508003, 8500914 to 8500917, 8508014 to 8508017 | 8500903, 8500914 to 8500917, 8500924, 8500927 to 8500930, 8500932, 8500934 to 8500939, 8508017 |
| Guaraná soda | 8200301, 8200302, 8201601, 8201602, 8207401, 8207402 | same |
| Cola | 8200101, 8200102, 8200701, 8200702, 8200801 | same |
| Other soda | Orange, grape, lemon and unspecified sodas, regional sodas, tonic water and quinine soda, regular, light and diet (29 codes) | Same rule, plus the "H2O"-type soft drink (25 codes) |

Soda is the union of guaraná, cola and other soda. The full lists, with each code's catalogue name, are in `CODIGOS` in `src/occasion/config.py`. Left out in both editions: natural guaraná refresco (`8200308`, `8500608`), mate (a tea), powdered drink mixes, energy and isotonic drinks, waters and teas. The semi-prepared mini pizza (`6502403`, `6506003`) exists only in the 2008-2009 catalogue and is left out to keep the two editions consistent.

### Part 1: occasion

- **Unit:** person x day x cell. A cell is one of five time bands crossed with one of two places, so every person-day has ten cells, whether or not the person ate anything in them. Time and place exist only when someone eats; with one row per person-day they could not be inputs without leaking the outcome. With cells they are legitimate inputs, because the cell exists before anything is eaten in it.
- **Target `y_pizza_refri`:** 1 if any occasion of that day, in that time band and place, has pizza and soda; 0 otherwise. A day can have more than one positive cell, so positive cells slightly outnumber positive days.
- **Target `y_pizza_guarana`:** 1 if any occasion in the cell has pizza and guaraná soda. It is the event the final score is evaluated against.

### Part 2: preference

- **Unit:** an occasion with at least one soda and a known place.
- **Target `y_guarana`:** 1 if the occasion includes guaraná soda.
- **Training population:** all soda occasions, not only those with pizza, because pizza with guaraná soda alone has few cases. The column `tem_pizza` says whether the occasion also had pizza; it is observed in training and fixed to 1 at inference, so the model answers "if there is pizza with soda in this context, how likely is the soda to be guaraná".

The score is the product of the two parts, which is the chain rule for "pizza with guaraná soda in this cell", and is shown as a weighted rank from 0 to 100 among the contexts of a reference population. The activation type (promotion, influencer, channel) is chosen by explicit rules downstream and is never a target.

### Counts

<!-- BEGIN:contagens -->
| Count | 2008-2009 | 2017-2018 |
|---|---|---|
| Person-days | 66,903 | 85,018 |
| Cells (person-day x time band x place) | 669,030 | 850,180 |
| Days with pizza and soda in the same hour | 651 | 796 |
| Days with pizza and guaraná soda in the same hour | 101 | 180 |
| Soda occasions | 19,182 | 16,112 |

| Person-days by weekday | 2008-2009 | 2017-2018 |
|---|---|---|
| Monday | 5,013 | 13,179 |
| Tuesday | 14,699 | 14,668 |
| Wednesday | 8,686 | 14,281 |
| Thursday | 17,383 | 14,799 |
| Friday | 10,314 | 7,644 |
| Saturday | 6,430 | 3,862 |
| Sunday | 4,378 | 16,585 |

Counts of the modelling bases (cells and soda occasions) after exclusions, unweighted. In 2017-2018 a few soda occasions are dropped for undetermined place, so the totals differ slightly from the parser reference table. Days with pizza and no soda are not listed: counting them would require reading the 2017-2018 person-day base outside the gated loader.
<!-- END:contagens -->

The parser itself reproduces the raw reference counts of each edition (tested in `make teste-dados`). The modelling bases differ from them only in 2017-2018, and only through place: 10 soda occasions have an undetermined place (`V9018` equal to 9 or blank) and belong to no cell, so the soda-occasion base has 16,112 rows instead of 16,122, and one pizza-with-soda day is left without any positive cell, so the cell base has 796 positive days instead of the 797 of the raw reference table. In 2008-2009 nothing is excluded, and in both editions every item has an hour (`reports/bases.json`).

## 3. Variable dictionary

### Context variables (model inputs)

These are `COLUNAS_CONTEXTO` in `src/occasion/features/variaveis.py`, identical in both parts and both editions. Categorical domains come from constants (`DOMINIOS`), never from the data, so no base decides which levels exist; a value outside the domain raises an error.

| Column | Type | Domain | Source field, 2008-2009 | Source field, 2017-2018 | Notes |
|---|---|---|---|---|---|
| `dia_semana` | int | 1 (Monday) to 7 (Sunday) | `DIA_SEMANA`, numeric code | `DIA_SEMANA`, weekday written out | The 2008-2009 coding (1 is Monday) was checked against an external signal: beer intake peaks on codes 6 and 7. |
| `faixa_horario` | category | `madrugada` (hours 0 to 5), `manha` (6 to 10), `almoco` (11 to 14), `tarde` (15 to 17), `noite` (18 to 23) | `V9015` | `V9015` | Whole hour of the occasion. Bands fixed on 2008-2009 and frozen before any model. |
| `local` | category | `em_casa`, `fora_de_casa` | `V9018`, **food source**: 1 inside, 2 outside the household | `V9018`, **meal place**: 1 at home (or brought from home), 2 to 7 away from home | Same labels, different measurements (see section 6). Undetermined values are excluded. |
| `uf` | int | IBGE code of the 27 federative units, 11 to 53 | `COD_UF` | `UF` | |
| `regiao` | category | `norte`, `nordeste`, `sudeste`, `sul`, `centro_oeste` | derived | derived | First digit of the UF code. |
| `situacao` | category | `urbano`, `rural` | `TIPO_SITUACAO_REG` (resident record) | `TIPO_SITUACAO_REG` (resident record) | 1 urban, 2 rural. |
| `idade` | int | years; 10 or older in the data | `IDADE_ANOS` | `V0403` | From the resident record. |
| `sexo` | category | `masculino`, `feminino` | `V0405` | `V0404` | As recorded by the survey. |
| `renda_pc_sm` | float | at least 0 | `RENDA_TOTAL` | `RENDA_TOTAL` | Monthly income of the consumption unit, divided by its number of residents and by the edition's minimum wage. |
| `n_moradores_uc` | int | 1 or more | count of rows | count of rows | Number of residents of the consumption unit in the resident file. |

Inside the models, `adicionar_derivadas` adds `fim_de_semana` (1 for Friday to Sunday). It is computed from `dia_semana` the same way in training and inference.

### Target and design columns

| Column | Base | Type | Domain | Role |
|---|---|---|---|---|
| `y_pizza_refri` | cells | int8 | 0, 1 | Part 1 target. |
| `y_pizza_guarana` | cells | int8 | 0, 1 | Event for evaluating the score. |
| `y_guarana` | soda occasions | int8 | 0, 1 | Part 2 target. |
| `tem_pizza` | soda occasions | int8 | 0, 1 | Part 2 input: observed in training, fixed to 1 at inference. |
| `peso_final` | both | float | greater than 0 | Survey weight `PESO_FINAL` from the consumption record. Not a model input. |
| `cod_upa`, `estrato` | both | string | survey codes | Primary sampling unit (sector) and stratum. Used to build validation folds and bootstrap intervals, never as inputs. |
| `id_celula`, `id_ocasiao`, `id_dia`, `id_pessoa`, `edicao` | both | string | keys | Row identity. The person key is sector + household + consumption unit + informant; the day key adds the record sheet (`QUADRO`: 71 and 72 in 2008-2009, 72 and 73 in 2017-2018). |

The per-variable argument for why each input is known at the prediction instant is in `docs/ficha-de-vazamento.md` (in Portuguese).

### Application input

The application and the API take a `Contexto` (`src/occasion/inference/contract.py`) and convert it to the columns above with the same feature functions used in training:

| Field | Domain | Becomes |
|---|---|---|
| `uf` | one of the 27 UF abbreviations | `uf` (IBGE code) and `regiao` |
| `situacao` | `urbano`, `rural` | `situacao` |
| `dia_semana` | `segunda` to `domingo` | `dia_semana` (1 to 7) |
| `faixa_horario` | the five bands | `faixa_horario` |
| `local` | `em_casa`, `fora_de_casa` | `local` |
| `idade` | integer, 18 to 110 | `idade` |
| `sexo` | `masculino`, `feminino` | `sexo` |
| `faixa_renda` | `ate_meio_sm`, `meio_a_1_sm`, `de_1_a_2_sm`, `de_2_a_5_sm`, `mais_de_5_sm` | `renda_pc_sm`, at the band midpoint (`0.25`, `0.75`, `1.5`, `3.5`, `7.5` minimum wages) |
| `numero_moradores` | integer, 1 to 20 | `n_moradores_uc` |

The models are trained on everyone aged 10 or older, but the application accepts only ages 18 and over: it does not recommend soda activations aimed at minors.

## 4. Integration rules

- **The POF is the only source of rows and targets.** Nothing else is joined at the person, day or occasion level.
- **Two editions, never the same observation.** Each edition is parsed, validated and built separately, and every row carries its `edicao`. The recipe is chosen and evaluated with training on 2008-2009 and testing on 2017-2018. The served model, refitted after the test, stacks the rows of both editions, each keeping its own edition's weight, with no edition indicator and no reweighting to a common population.
- **Same definitions in both editions.** Occasion key day + hour (R12), soda defined by product (R13), the same pizza rule (the 2008-2009-only mini pizza left out), the same time bands and the same place labels.
- **Income in minimum wages, not in reais.** Ten years of inflation make nominal reais incomparable. Income per person is divided by the edition's minimum wage (R$ 415 and R$ 954), a public constant, so no preprocessing is fitted on 2017-2018.
- **Weather and holidays not joined.** In 2008-2009 the exact calendar date of each consumption day can be rebuilt (birth date plus the resident's age in days gives the start of the household diary, and the weekday fixes the day within that week). In 2017-2018 it cannot: birth date plus age leaves an error of about a month. A variable that exists in training but not in testing would break the temporal test, the served model is refitted on 2017-2018, the POF has no municipality (only UF and stratum), and national holidays hold too few positive cases to estimate anything. Details in section 26 of `docs/registro-de-decisoes.md` and `reports/investigacao_datas.json`.
- **Google Trends not joined.** Search interest is never treated as consumption or as a target. It was planned only as a UF-level cross-check against the POF preference (hypothesis H10) and was cut before any comparison with the POF: the state-level shares come rounded to whole percentages, the topics mix searches unrelated to the drink, and most states have no value in 2008-2009. The exported CSVs stay in `dados_externos/google_trends/` as a record; no number from them enters the score, the recommendation or the results.

## 5. Leakage strategy

**Prediction instant.** The moment an activation is planned, before the occasion. Every input describes the context or the audience and is known in advance; nothing observed about what the person ate that day enters at inference. `tem_pizza` is part of the conditional question in part 2 and is fixed to 1, not information about what someone ate.

**Temporal split.** Everything that is fitted or chosen (EDA, time bands, variables, model ladder, hyperparameters, calibration, score and decision cutoffs) uses 2008-2009 only. Before the test, 2017-2018 was used only to check parser counts, harmonize the schema and run an adversarial check on input columns, without the target. The final test was opened on 21 November 2026 and its result is committed in `reports/teste_final.json`; a later re-execution from the raw files reproduced the same metrics. Random shuffling is not the final evaluation.

**Design-aware validation inside 2008-2009.** Nested cross-validation with folds that keep each sector (`cod_upa`) whole and spread sectors within strata, so the same person, household or sector is never on both sides. Uncertainty comes from a bootstrap of whole sectors within each UF.

**Gated access to the 2017-2018 target.** `carregar_base` (`src/occasion/bases/acesso.py`) is the only reader of the processed bases used by modelling and testing. For 2017-2018 it returns only context columns unless the tag `receita-congelada` exists; the target columns (`y_pizza_refri`, `y_pizza_guarana`, `y_guarana`, `tem_pizza`) are not even loaded into memory. Post-test diagnostics and the served-model refit additionally call `exigir_teste_commitado`, which requires `reports/teste_final.json` to be committed and unchanged.

**Frozen recipe.** The recipe was frozen by the git tag `receita-congelada` (commit `fa700be`) before the test. A SHA-256 signature (`src/occasion/bases/assinatura.py`, `ARQUIVOS_DA_RECEITA`) covers the configuration, ingestion, target definition, features, models, score, validation, evaluation, the recipe file and the test code itself. `make avaliado` and `make teste-final` refuse to run if the signature differs from the tag's, and the test checks the installed versions of numpy, pandas, scipy, scikit-learn, lightgbm and pyarrow against `src/occasion/versoes_da_receita.py`. Hypotheses and decision rules were committed before the stages that test them (`docs/pre-registro.md`).

**What had been seen before.** Before the design was written, the raw reference counts of 2017-2018 and a few descriptive shares of guaraná and cola in that edition had already been looked at. They are recorded in the design document (`docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md`, in Portuguese) and did not guide the models without being redone on 2008-2009.

## 6. Source caveats

- **Flavor, not brand.** For guaraná the POF records the flavor ("refrigerante de guaraná"), not a brand. Every statement in this project is about the flavor.
- **No calendar date in the consumption record.** Only the weekday. No seasonality, weather or holiday analysis is possible across editions (see section 4).
- **No channel.** The survey says at home or away from home; it says nothing about delivery, apps or where the food was bought.
- **Place means different things.** In 2017-2018 `V9018` is where the meal was eaten (1 at home or brought from home, 2 to 7 away); in 2008-2009 it is the source of the food (1 inside, 2 outside the household). Both are mapped to `em_casa` and `fora_de_casa`, but they do not measure the same thing, and part of any difference between editions in this variable may come from that.
- **Unbalanced weekdays, differently in each edition.** See the weekday table above. In 2008-2009 there are far more Tuesdays and Thursdays and few Sundays: the household diary almost always starts on a Tuesday, and the two food-record days fall on day 0 and day 2 of that week. In 2017-2018 there are many more Sundays than Saturdays. Raw counts should not be compared across weekdays; population estimates use `PESO_FINAL`. Weekday is the variable that most separates the two editions in the adversarial check (`reports/auditoria.json`), which reflects collection design, not behavior.
- **Different collection methods.** 2008-2009 used a food record filled in by the resident on two non-consecutive days; 2017-2018 used a 24-hour recall by interview on two non-consecutive days. The food catalogue grew from 1,503 to 1,832 items (Rodrigues and colleagues, Revista de Saúde Pública, 2021). Any change between editions mixes ten years of habit change with the change of method, and this project cannot separate them.
- **No meal type in 2008-2009.** Hence the occasion key without meal type (R12).
- **Income in reais is not comparable.** Hence minimum wages. The real gain of the minimum wage between the two dates is a declared limit of this normalization.

## 7. Population covered and excluded

**Covered.** Residents aged 10 or older of the households that answered the POF personal food intake block, in all 27 federative units, urban and rural, with up to two recorded days per person. Estimates are weighted by `PESO_FINAL` to represent that population in each edition.

**Excluded or not observed.**

- People under 10 (not interviewed in the food intake block).
- Occasions with an undetermined place (10 soda occasions in 2017-2018, none in 2008-2009).
- Items without an hour (none in either edition) would not form occasions.
- The 2008-2009 mini pizza codes, and drinks outside the soda definition (natural refresco, mate, powdered mixes, energy and isotonic drinks, waters, teas).
- In the application: anyone under 18.

## 8. Biases and limitations

- **Self-reported intake.** Both methods depend on memory or diligence; under-reporting of snacks and drinks is a known risk of dietary surveys, and it may differ between a written record and an interview.
- **Rare event.** Pizza with guaraná soda is rare in both editions (see the counts). That is why the score has two parts, and why the preference model is learned from all soda occasions; whether the preference order learned there holds for occasions with pizza is a stated risk, which the final test measures.
- **Two points in time, both before the delivery boom.** The latest edition is from 2017-2018. Nothing here describes current consumption or app-based ordering.
- **Different measurements between editions** (method, place, catalogue size) and a weekday mix driven by collection design.
- **Whole-hour occasion.** Pizza at 20h and soda at 21h are two occasions. Items eaten together across an hour boundary are missed, the same way in both editions.
- **Associations only.** The data are observational. Nothing in them supports a claim that an activation causes consumption, or that one activation sells more than another.
- **The served model is never tested on its own.** It is the frozen recipe refitted on both editions after the test; no clean data is left to test it.
- **Adult-only application on a model trained from age 10.** The training population is wider than the audience the application accepts.

## 9. Out-of-fold metrics, 2008-2009

Reference only; the test results on 2017-2018 and their reading are in the README.

<!-- BEGIN:metricas_2008 -->
| Model | Rows | Positives | Prevalence | PR-AUC | ROC-AUC | Log loss | Brier |
|---|---|---|---|---|---|---|---|
| Occasion (cells) | 669,030 | 654 | 0.001303 | 0.02377 | 0.9177 | 0.007770 | 0.001285 |
| Preference (all soda occasions) | 19,182 | 4,684 | 0.2345 | 0.3189 | 0.6084 | 0.5310 | 0.1746 |
| Preference (pizza only) | 655 | 101 | 0.1392 | 0.1644 | 0.5269 | 0.4134 | 0.1226 |
| Score (product) | 669,030 | 101 | 0.0001815 | 0.002627 | 0.9145 | 0.001462 | 0.0001813 |

Out-of-fold predictions on 2008-2009; prevalence, metrics and the score are weighted by the survey weight, rows and positives are unweighted.
<!-- END:metricas_2008 -->

The occasion row counts positive cells, which can exceed positive days because a day can have more than one positive cell. The score row is evaluated against `y_pizza_guarana` on the cells.

## 10. License and privacy (LGPD)

**Data.** The POF microdata are published openly by IBGE on its public server. They are used here with the source cited (IBGE, Pesquisa de Orçamentos Familiares 2008-2009 and 2017-2018, microdata) and are **not redistributed**: `data/` is excluded from git, and `make dados-brutos` downloads the files from IBGE.

**Code.** The project code is released under the MIT license.

**LGPD.** The microdata are anonymized by IBGE: no names, addresses or documents, only survey codes for sector, household and informant. This project:

- makes no attempt to re-identify anyone and joins no other person-level source;
- publishes only aggregates (counts, metrics, model parameters);
- commits a served model that holds fitted parameters and the weighted distribution of predicted probabilities over the reference population (sorted values with cumulative weights), not survey records or identifiers;
- sends no microdata to the language model that writes the optional briefing: the request carries only the context the user typed in the application and the model's output for it (score, probabilities, factors, recommendation and the flavor-not-brand notice). The provider is configured by whoever runs the application, through environment variables or Streamlit secrets; without a key, the application shows a fallback text built from the same output.

## 11. Reproduce

```bash
make ambiente        # dependencies (uv sync)
make dados-brutos    # download, checksum and extract the IBGE zips into data/raw
make bases           # parquet bases in data/interim and data/processed
make auditoria       # reports/auditoria.json: availability, sample size, adversarial check
make teste-dados     # all tests, including the reference counts
make docs            # regenerate the numeric blocks of this card and the README
```

Each base is rebuilt from the raw files; a full rebuild produced content-identical tables (section 27 of `docs/registro-de-decisoes.md`).
