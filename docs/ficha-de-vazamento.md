# Ficha de vazamento

Instante de previsão: o momento do planejamento da ativação, antes de a ocasião acontecer (spec, seção 5.1). Só entram variáveis que descrevem o contexto e o público e que são conhecidas de antemão. Nada do que foi observado sobre o consumo da pessoa naquele dia entra como informação na inferência. A lista abaixo cobre `COLUNAS_CONTEXTO` (em `src/occasion/features/variaveis.py`) e `tem_pizza`.

| Variável | Parte | Por que é conhecida no instante de previsão |
|---|---|---|
| `dia_semana` | 1 e 2 | O planejador escolhe o dia da ativação. Não depende do consumo. |
| `faixa_horario` | 1 e 2 | Na parte 1 é a própria célula, que existe mesmo quando nada foi consumido nela (spec, 5.2); com uma linha por pessoa e dia, usá-la seria vazamento. Na parte 2 é o horário da ocasião que se quer ativar, escolhido no planejamento. |
| `local` | 1 e 2 | Mesmo argumento da faixa: na parte 1 é a célula (em casa ou fora de casa); na parte 2 é o local da ocasião que se quer ativar. |
| `uf` | 1 e 2 | Atributo do público-alvo da ativação, conhecido de antemão. |
| `regiao` | 1 e 2 | Derivada da UF pelo primeiro dígito do código IBGE (`regiao_da_uf`). Sem informação de consumo. |
| `situacao` | 1 e 2 | Área urbana ou rural: atributo do público, conhecido de antemão. |
| `idade` | 1 e 2 | Atributo do público, do registro de moradores. Não vem do consumo. |
| `sexo` | 1 e 2 | Atributo do público, do registro de moradores. Não vem do consumo. |
| `renda_pc_sm` | 1 e 2 | Renda mensal por pessoa da unidade de consumo dividida pelo salário mínimo vigente na data de referência da edição (spec, 5.4). O salário mínimo é uma constante pública, conhecida no instante de previsão, então nenhum pré-processamento é ajustado nos dados de 2017-2018. O ganho real do salário mínimo no período é um limite declarado da comparação entre edições. |
| `n_moradores_uc` | 1 e 2 | Atributo do domicílio, do registro de moradores. Não vem do consumo. |
| `tem_pizza` | 2 | Só existe na parte 2. Ela é condicional por construção: responde "se houver pizza com refrigerante neste contexto, qual a chance de ser guaraná". A variável faz parte da pergunta e é fixada em "sim" na inferência; não é uma informação sobre o que alguém comeu (spec, 5.1). No treino é observada, e mede diretamente se a preferência muda quando há pizza (H4). |

## Colunas que não são variáveis de modelo

`id_celula`, `id_dia`, `id_pessoa`, `id_ocasiao`, `edicao`, `cod_upa`, `estrato` e `peso_final` identificam, agrupam ou pesam as linhas. Não entram como entrada de modelo. `peso_final` é usado em métricas e estimativas populacionais; se entra no treino é a hipótese H5. `cod_upa` e `estrato` servem a agrupamentos de validação.

## Alvos

`y_pizza_refri` e `y_pizza_guarana` (parte 1) e `y_guarana` (parte 2) são medidas observadas na POF. Nenhum é variável de entrada.
