# Registro de decisões do brainstorming

- **Datas da conversa:** 11 e 12/11/2026
- **Participantes:** o autor do projeto e o Claude
- **Documento irmão:** `docs/superpowers/specs/2026-11-12-occasion-intelligence-design.md` diz o que será construído. Este registro conta como se chegou lá: cada assunto discutido, a decisão, o motivo, as alternativas consideradas e por que cada uma foi descartada.

Os assuntos estão na ordem em que apareceram. Cada um indica quem decidiu: o autor, ou o Claude com o autor podendo vetar.

---

## Resumo das decisões

| # | Assunto | Decisão |
|---|---|---|
| 1 | Ponto de partida | A direção já aprovada não foi reaberta |
| 2 | Como escolher o modelo | Definir o protocolo de avaliação antes de qualquer modelo |
| 3 | Fonte do target | POF mantida, com duas descobertas anotadas |
| 4 | Coletor de dados de delivery | Descartado; vira próximo passo no README |
| 5 | Google Trends | No escopo, só como conferência |
| 6 | Enquadramento | Usuário de marketing ou trade; instante de previsão no planejamento |
| 7 | Protocolo de avaliação | Seis pontos aprovados |
| 8 | Mudanças entre 2008-2009 e 2017-2018 | Divisão temporal mantida, com quatro proteções |
| 9 | Unidade da parte 1 | Pessoa × dia × célula de horário e local |
| 10 | Candidatos a modelo | Escada de quatro degraus |
| 11 | Práticas de referência | Oito práticas adicionadas, hipóteses em seis campos e camadas de teste |
| 12 | Formato do score | Produto das partes, mostrado como posição de 0 a 100 |
| 13 | Este registro | Documento separado do design |
| 14 | Decision Engine | Duas camadas: objetivo e canal |
| 15 | Formato da aplicação | Consulta de contexto, com mapa de calor e briefing em botão |
| 16 | Stack | Python 3.12, uv, scikit-learn, LightGBM, Streamlit e o restante |
| 17 | Provedor de LLM | Groq, NVIDIA NIM ou equivalente, com interface compatível com a da OpenAI |
| 18 | Modelo servido | Reajuste na base inteira depois do teste, com salvaguarda |
| 19 | "Mesa" | Fora do escopo |
| 20 | Ordem de construção e de corte | Caminho inteiro primeiro; cinco itens na ordem de corte |
| 21 | Walk-forward | Não se aplica em vários passos; o desenho já é a versão de um passo |
| 22 | Segunda verificação das edições da POF | Confirmado: só duas edições têm consumo individual |
| 23 | Ajustes feitos ao escrever o design | Três correções e as escolhas do redator |
| 24 | Revisão independente do design | Sete perguntas respondidas e nove inconsistências corrigidas |
| 25 | Decisões durante a execução do Plano 1 | R12 e R13 (target), R9, R11 e R14 (escolhas do executor) |
| 26 | Clima e feriados casados com a POF | Fora do score; data exata existe só em 2008-2009; bloco de clima vira próximo passo |

---

## 1. Ponto de partida

**O que já estava decidido antes desta conversa,** registrado no `CLAUDE.md`:

- A pergunta central: em que contextos o guaraná se sai melhor contra a cola, e quais contextos de pizza se parecem com eles.
- O score em duas partes: ocasião e preferência.
- A POF como fonte do target, com treino em 2008-2009 e teste em 2017-2018.
- Promoção de pizza em app de delivery e influencers como categorias de ativação.
- O briefing por LLM no escopo.

**O que estava em aberto:** a stack, o formato da aplicação, as regras do Decision Engine, a comparabilidade entre as duas edições, a estabilidade da preferência em 2008-2009 e a ideia da "mesa".

**Decisão:** o brainstorming tratou só do que estava em aberto.

---

## 2. Como escolher o modelo

**O que aconteceu.** O autor perguntou como profissionais de ML escolhem modelos. O Claude respondeu com uma lista de perguntas e uma escada de três modelos, e disse que a decisão real era "qual deles vira o score".

**A crítica do autor.** A resposta simplificou demais decisões que não são simples. Ele pediu um fluxo de trabalho profissional, baseado no que um sênior em ML e AI Engineering faz, e disse que a simplificação gerou receio.

**O que o Claude reconheceu.** A escolha entre logística e LightGBM é a parte menos importante. O que tinha ficado de fora:

- a seleção não pode usar o teste;
- as linhas não são independentes, porque cada pessoa tem dois dias e a amostra é sorteada por setor;
- o peso amostral pode entrar no treino, na métrica ou nos dois;
- a parte 2 depende de uma hipótese de transferência para a pizza;
- multiplicar as duas partes precisa de justificativa;
- os candidatos devem sair de hipóteses sobre os dados;
- o método de coleta pode ter mudado entre as edições.

**Decisão (autor):** definir um fluxo profissional antes de falar em modelo. O Claude propôs nove etapas, em ordem: enquadramento, auditoria dos dados, protocolo de avaliação, baselines, candidatos, experimentos registrados, teste final, camada de decisão e empacotamento. O ajuste que o Claude fez à proposta do autor: o primeiro item a fixar é o protocolo de avaliação, porque sem ele qualquer comparação entre modelos é arbitrária.

**Efeito no trabalho.** O autor pediu para ir devagar. A partir daí, cada decisão de método foi apresentada com o que foi considerado e o motivo.

---

## 3. Fonte do target: a POF

**A dúvida do autor.** A POF pareceu fraca: poucos dados e só duas edições, de 2008-2009 e 2017-2018.

**Decisão (autor):** manter a POF.

**Por que ela continua sendo a melhor opção:**

- É a única base pública encontrada que observa a ocasião: quem comeu o quê, em que horário, onde e junto com o quê. O TAPI exige que o target seja uma medida real.
- Duas edições permitem a validação temporal obrigatória, com target real nos dois períodos.

**Onde a crítica do autor acerta:**

- **Poucos casos na interseção.** Os 85 mil dias-pessoa e as 15.863 ocasiões com refrigerante dão volume razoável. O que é pequeno é pizza com guaraná: 176 dias em 2017-2018 e 101 em 2008-2009. Por isso o score tem duas partes.
- **Base antiga,** anterior à expansão do delivery.
- **Só dois pontos no tempo,** sem tendência nem sazonalidade.

**Alternativas consideradas e descartadas:**

| Alternativa | Por que foi descartada |
|---|---|
| Google Trends como target | Mede interesse de busca; o TAPI proíbe |
| Dados do iFood, Nielsen ou Kantar | Não são públicos |
| Olist | É e-commerce, sem comida |
| Dados de recibos da Rappi | São pagos |
| Bases abertas de compras dos Estados Unidos, como a da Instacart | Outro país, sem guaraná |
| Vigitel | Pergunta em quantos dias da semana a pessoa toma "refrigerante ou suco artificial"; sem sabor, sem horário, e não foi encontrado item de pizza |
| Outros inquéritos de consumo individual | O único nacional é o próprio bloco da POF |

**Descoberta 1: a POF 2024-2025.**

- A coleta foi de novembro de 2024 a novembro de 2025, com o módulo de consumo alimentar pessoal.
- Segundo o resumo de um comunicado do IBGE, os primeiros resultados saem em 27/11/2026, dois dias depois da entrega, e tratam de despesas, rendimentos e variação patrimonial.
- Não foi encontrada data para os microdados de consumo alimentar.
- O servidor do IBGE não tem pasta dessa edição.
- **Ressalva:** o site do IBGE recusou o acesso automático em todas as tentativas. A data de 27/11 não foi confirmada na fonte.
- **Efeito:** não dá para usar a edição nova. Ela entra no README como próximo passo e fortalece o argumento de que o projeto é um método a ser rodado de novo.

**Descoberta 2: o método de coleta mudou.**

- 2008-2009 usou registro alimentar; 2017-2018 usou recordatório de 24 horas.
- O cadastro de alimentos cresceu de 1.503 para 1.832 itens.
- **Efeito:** o teste em 2017-2018 mistura a mudança de hábito em dez anos com a mudança de método. Isso é limite declarado.

**Ideia levantada e não verificada:** o bloco de despesas da própria POF registra compras de pizza e refrigerante em mais domicílios. Não foi conferido se ele distingue o sabor do refrigerante. Não entrou no escopo.

---

## 4. Coletor de dados de delivery

**A pergunta do autor.** Se existe alguma base de delivery que dê para extrair por API, programando um coletor.

**Decisão (autor):** não construir. Vira próximo passo no README.

**O que foi apurado:**

| Fonte | O que traz | Por que não serve como target |
|---|---|---|
| API oficial do iFood | Pedidos e catálogo de lojas que autorizaram o desenvolvedor | Exige CNPJ e permissão loja a loja |
| Cardápios do iFood ou Rappi, por coleta não oficial | Itens, preços, combos e notas | Mede oferta, não consumo |
| Google Places | Pizzarias por cidade, nota e número de avaliações | Mede presença de pizzarias |
| Google Trends e YouTube | Buscas e vídeos | Mede interesse |

**Motivos do descarte:**

- Um coletor só traz cardápios. O dado de quem pediu o quê fica com a plataforma e com o restaurante.
- A medida possível, a parcela de combos de pizza com guaraná por cidade, responde a outra pergunta e não cruza com a POF, porque período e unidade são outros.
- A coleta usa a interface interna do aplicativo, que não é oficial. Um relato encontrado diz que um coletor funcionou até ser bloqueado.
- O texto dos termos de uso do iFood não foi encontrado, então não se sabe se há proibição expressa.
- Em duas semanas, com vídeo eliminatório, um coletor que pode ser bloqueado é o maior risco de prazo.

O próprio TAPI, na seção 9, cita coletores de código aberto como referência técnica e avisa que preço de vitrine não comprova venda.

---

## 5. Google Trends

**A pergunta do autor.** Se o Trends não agregaria nada.

**Decisão (autor):** colocar no escopo, só no papel de conferência.

**Por que tem valor.** Ele cobre os três buracos da POF: tem marca, tem data e é atual.

**Usos aprovados:**

1. **Conferir se o padrão da POF ainda vale.** Comparar por UF o interesse de busca com a preferência guaraná × cola da POF, nos mesmos períodos das duas edições. É um bloco pequeno da EDA, com arquivos CSV fixos no repositório.

**Usos que ficaram como extra, se sobrar tempo:**

2. Sazonalidade ao longo do ano, mostrada como contexto de interesse de busca.
3. Conferência do padrão semanal.

**Usos descartados:**

| Uso | Por que foi descartado |
|---|---|
| Variável do modelo | Sem data na POF, o Trends só entraria como um valor por UF, quase igual à própria UF e com mais risco de vazamento |
| Entrada que altera o score ou a recomendação | O proxy passaria a ser tratado como consumo |
| Coleta pela biblioteca não oficial | Costuma ser bloqueada; a exportação de CSV pelo site é reprodutível |

**Limites anotados:** são só 27 pontos; o volume de buscas em 2008-2009 era menor e estados pequenos podem vir zerados; busca por marca tem outros motivos além de consumo.

---

## 6. Enquadramento

**O que aconteceu.** O Claude perguntou duas vezes quem usaria a aplicação e o autor não tinha uma resposta pronta. O Claude então propôs um retrato, marcado como palpite, e o autor aprovou sem alterações.

**Decisão (palpite do Claude, aprovado pelo autor):**

- **Quem usa:** alguém de marketing ou trade da Ambev planejando ativações.
- **O que informa:** UF, dia da semana, horário, em casa ou fora, e perfil do público.
- **O que decide:** para qual contexto direcionar qual tipo de ativação.
- **Instante de previsão:** o momento do planejamento, antes da ocasião. Nada sobre o que a pessoa comeu naquele dia entra como variável.

---

## 7. Protocolo de avaliação

**Decisão (proposta do Claude, aprovada pelo autor):**

1. **Papéis fixos.** 2008-2009 é o desenvolvimento; toda escolha acontece nela. 2017-2018 é o teste, aberto uma vez com a receita congelada.
2. **Validação dentro de 2008-2009:** cruzada em cinco partes, agrupada por setor e estratificada.
3. **Métricas fixadas antes:** log loss ponderado como principal; PR-AUC e curva de calibração como secundárias; todas com peso amostral.
4. **Critério de vitória:** a diferença precisa ser maior que a incerteza, medida por reamostragem de setores inteiros; no empate, fica o mais simples.
5. **Parte 2 avaliada duas vezes:** em todas as ocasiões com refrigerante e só nas de pizza.
6. **Registro:** cada experimento grava configuração, semente, commit e métricas em arquivo.

**Um problema reconhecido.** O teste já tinha sido olhado: os números "guaraná cai de 27% para 23% com pizza" são de 2017-2018. A decisão foi que a EDA passa a olhar só 2008-2009 e que essa observação precisa ser refeita lá antes de influenciar o desenho.

**Por que essas escolhas:**

- **Agrupar por setor:** cada pessoa tem dois dias, e a POF sorteia por setor e domicílio. Sem agrupar, a métrica sai inflada.
- **Log loss:** a saída é uma probabilidade que vira score; a métrica pune ordenar mal e errar o tamanho da probabilidade.
- **PR-AUC:** o evento é raro, e ela mede se os contextos mais prováveis ficam no topo.

---

## 8. Mudanças no mundo real entre as edições

**A dúvida do autor.** Se mudanças grandes entre 2008-2009 e 2017-2018 não atrapalhariam muito o modelo.

**Resposta.** É quase certo que aparecem no resultado, e medir isso é a função do teste temporal.

**O que mudou:** a renda nominal, por dez anos de inflação; o consumo de refrigerante, em queda (um estudo com o Vigitel mostra redução de 32,7% no consumo regular entre 2007 e 2014); a economia; o delivery por aplicativo, que começava em 2017-2018; o método de coleta e o cadastro de alimentos.

**Três tipos de mudança:**

1. A frequência geral muda: o modelo erra o tamanho da probabilidade, mas a ordem dos contextos se mantém.
2. O perfil da população muda: costuma causar pouco dano.
3. A relação entre contexto e consumo muda: é o dano real, e só o teste revela.

**Indício nos números já conhecidos,** sem peso amostral: dias com pizza e refrigerante são 0,95% em 2008-2009 e 0,92% em 2017-2018; o guaraná entre os dias de pizza com refrigerante passa de 16% para 22%. A parte 1 parece estável; o risco se concentra na parte 2.

**Decisão (autor):** manter 2008-2009 como treino e 2017-2018 como teste.

**Proteções adotadas:**

- A renda não entra em reais. Na conversa foram citadas duas formas, posição relativa dentro da edição ou salários mínimos; a revisão do design fechou em salários mínimos (item 24).
- O Decision Engine usa termos relativos, não probabilidades absolutas.
- A queda de desempenho é relatada por segmento.
- Um diagnóstico depois do teste treina a receita só em 2017-2018 para medir quanto se perde ao atravessar as edições.

---

## 9. Unidade da parte 1: células

**O problema, percebido pelo Claude no meio da conversa.** A parte 1 tinha sido pensada com uma linha por pessoa e dia. Mas horário e local só existem quando a pessoa come algo. Usá-los como entrada seria vazamento.

**Decisão (proposta do Claude, aprovada pelo autor):** cada dia é dividido em células, cada uma sendo uma faixa de horário combinada com um local. O target de cada célula é: houve pizza com refrigerante nela, sim ou não.

**Alternativa descartada:** manter a linha por pessoa e dia, com a parte 1 conhecendo só o dia e o perfil. Foi descartada porque deixaria de fora justamente o "sábado à noite em casa", e o local é o que diferencia as ativações.

**Custo aceito:** a base cresce cerca de dez vezes em linhas, com o mesmo número de positivos.

---

## 10. Candidatos a modelo

**Decisão (proposta do Claude, aprovada pelo autor):** uma escada de quatro degraus, em que cada um responde a uma pergunta.

| Degrau | Modelo | Pergunta |
|---|---|---|
| A | Tabela de taxas ponderadas, sem ML | Quanto se consegue só contando? |
| B | Regressão logística regularizada | As variáveis somam informação além da tabela? |
| C | Logística com curvas e poucas interações | Os efeitos são curvos, e combinações importam? |
| D | LightGBM com árvores rasas | Existem combinações em que ninguém pensou? |

**Regras da comparação:** busca de hiperparâmetros pequena e declarada antes; validação aninhada; sem reamostragem do evento raro; calibração conferida em todos, com correção sigmoide se necessário; peso amostral no treino tratado como experimento.

**Específico da parte 2:** o target é "a ocasião inclui guaraná", e "tem pizza" entra como variável.

**Alternativas descartadas:**

| Alternativa | Por que foi descartada |
|---|---|
| Rede neural | Em dado tabular raramente ganha, e o volume é pequeno |
| Modelo hierárquico em biblioteca própria | A regularização da logística cumpre o mesmo papel para a UF, sem dependência extra |
| Técnicas de reamostragem para equilibrar classes | Distorcem a probabilidade, e o score depende dela |
| Correção de calibração isotônica | Com poucos positivos, a sigmoide é mais segura |
| Comparar modelos sem validação aninhada | O modelo com mais hiperparâmetros levaria vantagem artificial |

O autor não pediu nenhum modelo adicional.

---

## 11. Práticas de referência, hipóteses e testes

**O pedido do autor.** Tudo bem testado, hipóteses bem estruturadas, e uma nova conferência das metodologias usadas por profissionais experientes, consultando livros e fontes.

**Decisão (autor):** adotar o pacote inteiro recomendado.

**O que as fontes confirmaram que já estava no protocolo:** baseline, teste usado uma vez, validação por setor, métrica ponderada e calibração.

**Oito práticas que faltavam:**

| Prática | Fonte | Papel no projeto |
|---|---|---|
| Registro prévio de hipóteses | Modelos de pré-registro para modelagem preditiva | Hipóteses commitadas antes do teste, separando confirmatório de exploratório |
| Ficha de vazamento | Kapoor e Narayanan | Argumento escrito por variável |
| Conta de tamanho de amostra | Riley e colegas | Limita o número de parâmetros |
| Validação adversarial | Prática comum em competições e monitoramento | Mostra quais variáveis mudaram entre as edições, sem usar o target |
| Três níveis de calibração | Van Calster e colegas | Relatar média, inclinação e curva |
| Limite de comparações | Guia de Lones | Experimentos declarados antes e todos registrados |
| Dobras que imitam o desenho amostral | Wieczorek e colegas; revisão SaML | Agrupar por setor e estratificar pelo estrato |
| Caminho inteiro primeiro | Rules of ML, do Google | Aplicação funcionando com o baseline desde a primeira semana |

**Formato das hipóteses:** enunciado, motivo, teste, métrica, regra de decisão e o que muda se for refutada.

**Camadas de teste:** dados, código, reprodutibilidade, igualdade entre treino e aplicação, comportamento do modelo, rótulos embaralhados e Decision Engine. O teste de rótulos embaralhados vem da experiência do Claude, não das fontes consultadas.

**Práticas consideradas e deixadas de fora:**

| Prática | Por que ficou de fora |
|---|---|
| Análise de benefício líquido por limiar | Exige atribuir ganho a cada ativação, o que encosta em alegação causal |
| Reponderar o treino para parecer com 2017-2018 | Usa o teste no ajuste |
| Combinação de modelos e redes neurais | Custo alto para o volume de dados |
| Ferramenta de rastreio de experimentos | Arquivos JSON bastam |

**O que não foi lido por inteiro:** o texto completo do ML Test Score, os nove passos do SaML e os itens do TRIPOD+AI. Ficam como referência de apoio, não como checklist.

---

## 12. Formato do score

**Decisão (proposta do Claude, aprovada pelo autor):**

- O produto das duas partes é a chance de pizza com guaraná na célula. Como a parte 2 é calculada com "tem pizza = sim", o produto é a regra da cadeia, não uma suposição de independência.
- O produto é conferido contra o evento observado, que existe na POF.
- Uma hipótese compara a decomposição com um modelo treinado direto no evento raro. Se a decomposição perder, o modelo direto assume.
- O `occasion_score` vai de 0 a 100 e é a posição do contexto entre todos os contextos.
- As posições nas duas partes aparecem separadas, em um campo a mais no contrato.
- A probabilidade calibrada acompanha.

**Alternativa descartada:** mostrar a probabilidade bruta como número principal. Ela é da ordem de centésimos de por cento, difícil de ler, e sensível à mudança de frequência geral entre as edições.

---

## 13. Este registro

**O pedido do autor.** Um documento, além do `CLAUDE.md`, com tudo o que foi decidido e discutido, incluindo o que foi descartado e por quê.

**Decisão (Claude, sem veto do autor):** dois arquivos. O design diz o que será construído; este registro conta a história das decisões.

**Alternativa descartada:** um arquivo só. O design precisa ser lido por quem monta o plano de implementação, e a história da discussão o deixaria longo demais para isso.

---

## 14. Decision Engine

**O que aconteceu.** O Claude perguntou que tipos de ativação o autor tinha em mente. O autor respondeu com dashboard e chatbot. O Claude esclareceu que esses são formatos de aplicação, e que a pergunta era sobre a ação de marketing recomendada.

**Decisão (proposta do Claude, aprovada pelo autor):** a recomendação tem duas camadas.

**Objetivo,** pelo cruzamento das duas partes do score:

| | Preferência alta | Preferência baixa |
|---|---|---|
| **Ocasião alta** | Sustentar | Converter |
| **Ocasião baixa** | Criar ocasião | Não priorizar |

**Canal,** pelo local e pelo perfil: em casa aponta para combo em app de delivery ou para pizza e guaraná juntos no varejo; fora de casa, para ação na pizzaria; público jovem, para influencers e conteúdo digital; concentração regional, para mídia local.

**Limite:** o sistema diz que a ativação combina com o contexto. Ele não afirma que uma ação vende mais que outra.

O autor não cortou nem acrescentou objetivos ou canais.

---

## 15. Formato da aplicação

**Como foi decidido.** O Claude mostrou três esboços no navegador, com números ilustrativos.

| Opção | Descrição | Resultado |
|---|---|---|
| A. Consulta de contexto | Formulário de um lado, resultado do outro | Virou a base |
| B. Painel de contextos | Filtros, mapa de calor e lista dos contextos mais fortes | Descartado como formato principal; o mapa de calor foi aproveitado |
| C. Conversa | Pergunta em texto livre, com um LLM traduzindo e redigindo | Descartado |

**Decisão (autor):** a combinação: opção A como base, o mapa de calor da opção B abaixo do resultado e o briefing por LLM em um botão.

**Motivos dos descartes:**

- **B puro:** mostra mais do modelo, mas pede mais telas, gráficos e tempo.
- **C:** é o mais chamativo e o mais arriscado. O LLM pode entender o contexto errado, e isso apareceria ao vivo no vídeo. Além disso, o LLM passaria a ter um papel na entrada, quando a regra do projeto é que ele só redige.

**Um detalhe da sessão.** O verificador de design apontou texto de 10px nos rótulos do mapa de calor do esboço. O Claude corrigiu para 12px em uma nova versão da tela.

---

## 16. Stack

**Decisão (proposta do Claude, sem veto do autor).** A máquina do autor foi conferida: `uv`, `git`, Pyright, 16 GB de RAM e 56 GB livres.

| Para quê | Escolha | Motivo |
|---|---|---|
| Linguagem | Python 3.12 fixado pelo `uv` | O Python do sistema é o 3.14, recente demais para garantir LightGBM e SHAP |
| Dados | pandas e Parquet | Já previstos no `CLAUDE.md` |
| Esquema | pandera | Validação em cada etapa |
| Modelos | scikit-learn, LightGBM e SHAP | Cobrem a escada e a explicação |
| Testes | pytest e Hypothesis | Hypothesis gera os casos do Decision Engine |
| Qualidade | Pyright e ruff | Tipos e estilo |
| Contrato | Pydantic | Um contrato para função, tela e API |
| Tela | Streamlit, chamando a função direto | O vídeo só precisa de um processo rodando |
| API | FastAPI como camada fina | Poucas linhas; primeiro item a cortar |
| Comandos | `Makefile` | Um comando por etapa do pipeline |

**Briefing:** o LLM recebe só a decisão estruturada. Sem chave ou sem rede, a aplicação mostra um texto por modelo fixo, para o vídeo não depender de internet.

**Alternativas descartadas:**

- **Tela chamando a API por rede:** exigiria dois processos no vídeo.
- **MLflow:** opcional no TAPI; arquivos JSON bastam.
- **Docker:** o `uv` com arquivo de trava já dá a reprodutibilidade.

---

## 17. Provedor de LLM

**O que foi apurado.** Não havia chave da API da Anthropic configurada no terminal do autor.

**Decisão (autor):** usar uma chave do Groq, do NVIDIA NIM ou de outro provedor desse tipo.

**Como fica:** a camada de briefing usa a interface compatível com a da OpenAI, e endereço, modelo e chave vêm de variáveis de ambiente. Trocar de provedor não mexe no código. O modelo exato é escolhido na implementação.

**Efeito no `CLAUDE.md`:** a linha que manda carregar a skill da API do Claude antes do código do briefing precisa ser ajustada.

---

## 18. Qual modelo a aplicação serve

**O dilema.** As métricas são do modelo treinado em 2008-2009, mas servir recomendações com hábitos de 2008 é a parte mais fraca da história.

**As opções:**

| Opção | Vantagem | Custo |
|---|---|---|
| 1. Servir o modelo avaliado | O que está na tela é o que foi medido | A recomendação reflete o dado mais antigo |
| 2. Reajustar a receita só em 2017-2018 | Dado mais recente, um método de coleta só | Menos casos |
| 3. Reajustar na base inteira | Quase triplica os casos de pizza com guaraná em relação ao modelo de 2008-2009 | Mistura dois métodos de coleta |

**O percurso da decisão.** O Claude recomendou primeiro a opção 2 e descartou a 3 pela mistura de métodos. O autor propôs a opção 3 e perguntou se era boa prática. O Claude explicou que o reajuste final é prática reconhecida e, ao pesar o contexto do projeto, mudou a recomendação para a opção 3.

**Decisão (autor):** opção 3 como padrão, com salvaguarda.

**Por que a base inteira neste projeto:**

- O gargalo é a falta de casos: 101 dias de pizza com guaraná em uma edição e 176 na outra.
- O score é uma posição relativa, e a principal diferença esperada entre as edições é de nível.
- A recomendação deixa de refletir só 2008.

**A salvaguarda.** Um experimento declarado antes e rodado depois do teste: validação cruzada dentro de 2017-2018, comparando o treino só com o dado recente contra o treino com 2008-2009 somado. A base inteira é adotada se não for pior por mais que a incerteza. Se for pior, serve-se o modelo só de 2017-2018.

**Travas:** o reajuste só acontece depois do teste gravado; a receita não é reescolhida; os dois artefatos têm nomes distintos; o `model_metadata` informa em que dados o modelo servido foi treinado.

**Limitações aceitas:** o modelo servido nunca é testado; nada pode mudar depois do teste.

**Plano B de prazo:** opção 1.

---

## 19. "Mesa"

**Decisão (proposta do Claude, aprovada pelo autor):** fora do escopo; citada no README como próximo passo.

**Motivo:** exige uma terceira unidade de análise, com poucos casos, e não é obrigatória no TAPI.

---

## 20. Ordem de construção e ordem de corte

**Decisão (proposta do Claude, aprovada pelo autor).**

**Ordem:** design, plano e parsers em 12 e 13/11; bases, auditoria, baseline e o caminho inteiro em 14 e 15/11; EDA, hipóteses e modelos de 16 a 18/11; teste final em 19/11; salvaguarda, modelo servido e Trends em 20/11; tela completa em 21/11; documentação em 22/11; vídeo e folga de 23 a 25/11.

**Ordem de corte:** camada FastAPI; degrau C; bloco do Trends; salvaguarda e reajuste; briefing por LLM. Os entregáveis obrigatórios do TAPI não entram na lista.

**Motivo da ordem:** a partir do dia 15 já existe uma aplicação que roda, então o vídeo eliminatório nunca depende do modelo final.

---

## 21. Walk-forward

**A pergunta do autor.** Ele aplicou a técnica no projeto anterior e queria saber como funciona e se serviria aqui.

**O que é.** Uma validação para dados ordenados no tempo, que repete o ciclo "treinar no passado, avaliar no bloco seguinte" avançando a janela. Serve para imitar o uso real, impedir vazamento do futuro, obter várias medidas ao longo do tempo e ajustar hiperparâmetros sem tocar no teste final.

**Decisão:** não se aplica em vários passos.

**Motivos:**

- Entre as edições só existe um passo, e ele já é o desenho do projeto.
- Dentro de uma edição o registro só tem dia da semana, então as linhas não podem ser ordenadas no tempo.
- Mesmo com data, uma edição cobre doze meses, e o salto que importa é de nove anos.

**Onde ele entra:** quando o consumo alimentar da POF 2024-2025 for publicado, o desenho vira um walk-forward de duas rodadas. A segunda rodada testa o modelo servido.

**Item para a auditoria:** conferir se algum registro traz o trimestre da coleta; se trouxer, serve só para checar sazonalidade.

---

## 22. Segunda verificação das edições da POF

**O pedido do autor.** Confirmar se só existem as edições 2008-2009 e 2017-2018.

**O que a primeira verificação tinha feito:** só a listagem dos nomes das pastas no servidor do IBGE.

**O que a segunda verificação fez:**

- Abriu cada pasta. O servidor tem cinco edições: 1987-1988, 1995-1996, 2002-2003, 2008-2009 e 2017-2018.
- As três mais antigas mediram a compra de alimentos pelo domicílio. A pasta de 2002-2003 só tem "Aquisição alimentar domiciliar"; a publicação "Análise do Consumo Alimentar no Brasil" aparece pela primeira vez em 2008-2009.
- A literatura confirma: o consumo individual entrou em 2008-2009, em uma subamostra de cerca de um quarto dos domicílios, e foi repetido em 2017-2018.
- Os arquivos citados no `CLAUDE.md` continuam sendo os atuais.
- O ENDEF, dos anos 1970, é anterior à POF e não teve continuidade. Não foi verificado se tem microdados utilizáveis.

**Decisão:** fonte confirmada.

---

## 23. Ajustes feitos ao escrever o design

Ao escrever o design, o Claude leu o TAPI por inteiro e fez três correções em relação ao que tinha dito na conversa.

**Correção 1: a auditoria não compara taxas do target entre as edições.** Na conversa, o Claude disse que a auditoria compararia "as taxas brutas das duas edições antes de qualquer modelo". Calcular taxas do target em 2017-2018 antes do teste seria olhar o teste de novo. No design, a auditoria usa só as variáveis das duas edições e o target de 2008-2009. As contagens de referência continuam valendo como teste do parser.

**Correção 2: a metade de 2017-2018 da comparação com o Trends só roda depois do teste,** pelo mesmo motivo.

**Correção 3: métricas que o TAPI pede e que não tinham sido listadas.** Precision, recall, F1 e matriz de confusão, calculadas no corte da classe "alta"; e a justificativa da métrica principal pelo custo dos erros. Como o projeto não tem valores para esses custos, o design explica que não fixa um limiar ótimo.

**Acréscimos a partir do TAPI:**

- Uma rodada de validação deixando cada grande região de fora, porque o TAPI diz que divisões por região medem generalização.
- A análise por segmentos usa o local como substituto de canal, declarado.
- Os itens obrigatórios do Data Card.
- As extensões da seção 9 do TAPI (Trend Radar, Mapa de Oportunidade e Laboratório de Preço) como fora do escopo.

**O que já tinha sido visto do teste.** A memória de sessões anteriores registra, além dos números do `CLAUDE.md`, uma leitura descritiva de 2017-2018: guaraná mais forte no Nordeste, na renda baixa e entre pessoas de 60 anos ou mais. O design lista isso na seção 4.3 e proíbe que oriente o modelo sem ser refeito em 2008-2009.

**Um contraste anotado.** O TAPI diz que os dados da Ambev mostram maior associação de Guaraná Antarctica com a ocasião na classe A. A leitura descritiva da POF de 2017-2018 aponta o sabor guaraná mais forte na renda baixa. As duas afirmações não são diretamente comparáveis: uma é sobre a marca na ocasião de pizza, a outra sobre o sabor em todas as ocasiões com refrigerante. O projeto vai medir isso em 2008-2009 antes de afirmar qualquer coisa.

**Escolhas do redator.** O design tem vinte e quatro decisões tomadas pelo Claude ao escrever e ao responder à revisão, listadas na seção 19 de lá. As que mais merecem conferência do autor:

- **"Preferência alta" significa guaraná acima do contexto mediano,** e não guaraná à frente da cola. Em termos absolutos, a cola lidera na maioria dos contextos.
- **A aplicação só aceita público de 18 anos ou mais.** O modelo treina com todos, mas a aplicação não recomenda ativação de refrigerante para menores.
- **Os cortes numéricos:** ocasião alta a partir da posição 75; preferência alta a partir da posição 50; classificação alta a partir de 80 e média a partir de 50.
- **A ordem das regras de canal,** com influencers para quem tem menos de 30 anos quando o objetivo é converter, e varejo para área rural.

---

## 24. Revisão independente do design

**O que foi feito.** Antes de entregar o design ao autor, um revisor sem acesso à conversa leu o documento no papel de quem teria de montar o plano de implementação. Ele devolveu sete perguntas e nove inconsistências. Nenhuma exigiu decisão do autor; todas foram resolvidas pelo Claude e estão marcadas no design como escolha do redator.

### Perguntas do revisor e como foram respondidas

| # | Pergunta | Resposta incorporada ao design |
|---|---|---|
| 1 | Em que ordem as escolhas encadeadas são feitas, e em que modelo se mede cada hipótese? | Ordem fixa: variáveis; peso no treino, medido no degrau B; escada; calibração só no vencedor; produto; cortes. O efeito de "tem pizza" é o coeficiente da logística do degrau B. A receita congelada é esse conjunto |
| 2 | O que a auditoria com variáveis de 2017-2018 pode mudar na receita? | Nada. A validação adversarial é só relato. A lista de variáveis é fixada em 2008-2009; de 2017-2018 só vem a exigência de a variável existir nas duas edições |
| 3 | "Preferência alta" é comparada com qual média, e em que população? | Passou a ser a posição 50 na parte 2. As posições das duas partes e do produto são calculadas na mesma população de referência, com "tem pizza = sim" na parte 2 |
| 4 | Qual é a regra exata da hipótese sobre o ordenamento no teste? | Vale o score contra o evento observado de pizza com guaraná, com PR-AUC, comparado ao produto das duas tabelas de taxas. As partes separadas e a concentração no topo são relatadas à parte |
| 5 | O modelo servido tem indicador de edição, e a calibração usa quais linhas? | Sem indicador. A calibração usa só as linhas da edição mais recente, para a probabilidade e a população de referência serem da mesma época |
| 6 | Como se calculam os fatores quando o modelo é a tabela de taxas? | Pela diferença, em log-chances, entre a taxa de cada valor e a taxa geral |
| 7 | Quais termos e filtros do Trends, e quando o autor exporta? | Tópicos "Guaraná Antarctica" e "Coca-Cola" na mesma consulta, Brasil, por estado. A exportação pode ser feita a qualquer momento, porque não usa a POF |

### Inconsistências e como foram corrigidas

**1. "Tem pizza" contra o instante de previsão.** O design dizia que nada sobre o que a pessoa comeu entra como variável, mas "tem pizza" é variável da parte 2. Correção: a parte 2 é condicional por construção; a variável faz parte da pergunta e é fixada em "sim" na inferência. O argumento vai para a ficha de vazamento.

**2. Uso de 2017-2018 antes do teste.** A tabela de papéis só permitia contagens do parser, mas a auditoria usa as variáveis dessa edição. Correção: a tabela agora lista o que é permitido antes do teste, sempre sem o target e sem alterar a receita.

**3. Renda relativa como pré-processamento ajustado no teste.** A posição da renda dentro da edição seria calculada com a distribuição de 2017-2018, o que contraria a regra de que nada dessa edição entra no pré-processamento. Correção: a renda passa a ser medida em salários mínimos por pessoa, com o salário mínimo da data de referência de cada edição. É uma constante pública, e nada é ajustado nos dados de teste.

- **Alternativa descartada:** manter a posição relativa e declarar uma exceção à regra. Foi descartada porque a regra fica mais forte sem exceções, e a faixa em salários mínimos é mais natural na tela.
- **Custo aceito:** o salário mínimo teve ganho real no período, então a comparação entre edições é imperfeita. Fica como limite declarado.

**4. Retirar "tem pizza" quebraria a conta do score.** Na conversa, o exemplo de hipótese dizia que, se o efeito não aparecesse, a variável sairia. O revisor mostrou dois problemas: sem a variável, o produto das partes vira a suposição de independência que o design nega; e, com poucos casos de pizza, não achar o efeito indicaria falta de dados, não ausência de efeito. Correção: a variável permanece no modelo em todos os casos, e a hipótese só relata o efeito e o intervalo. Isso muda o que foi dito ao autor na conversa.

**5. Hipóteses "confirmadas" por falta de casos.** Três hipóteses eram confirmadas quando o intervalo de incerteza continha um valor de referência. Com cerca de cem dias positivos, o intervalo é largo, e a confirmação mediria falta de poder. Correções:

- A hipótese de calibração do produto deixou de usar "o intervalo contém 1". A correção sigmoide entra se melhorar o log loss na validação.
- Nas hipóteses do peso no treino, da decomposição e do modelo servido, o design declara qual opção é o padrão, e o relatório escreve "sem evidência contra o padrão" em vez de "confirmada".

**6. Teste de rótulos embaralhados.** O design dizia que, com o target embaralhado, o modelo cairia "para o nível do degrau A". Um degrau A treinado no target real fica acima disso. Correção: a comparação é contra uma previsão constante igual à frequência geral, na mesma rodada.

**7. Hipótese do teste final vazia se a tabela vencer.** Se a tabela de taxas for o modelo escolhido, compará-la com ela mesma não diz nada. Correção: nesse caso relata-se o desempenho da tabela contra a previsão constante.

**8. Trends no limite da regra de proxies.** Na conversa, o Claude disse que, se o Trends acompanhasse a POF, o Trends de hoje seria "um indício de que o padrão se manteve". O revisor apontou que isso trata interesse de busca como evidência de consumo, o que o TAPI proíbe. Correção: a hipótese mede só se o proxy acompanha a medida real no mesmo período e na mesma UF. O Trends recente é mostrado como interesse de busca, e nenhum texto o usa para afirmar algo sobre o consumo atual. Isso reduz o que foi prometido ao autor sobre esse bloco.

**9. Juntar as edições no modelo servido.** O revisor lembrou que o próprio design declara a troca de método de coleta, e que a salvaguarda está na ordem de corte. Correção: o design diz expressamente que o reajuste na base inteira nunca acontece sem a salvaguarda; se ela for cortada, vale o plano B. A junção das edições entra como limite declarado.

**Pendência no `CLAUDE.md`.** O revisor também notou que o `CLAUDE.md` manda carregar a skill da API do Claude para o briefing, enquanto o design usa outro provedor. A atualização do `CLAUDE.md` fica para depois da aprovação do design.

---

## 25. Decisões durante a execução do Plano 1

Duas decisões mudaram a definição do target e foram aprovadas pelo autor em 12/11/2026. Três escolhas menores ficaram com o executor e estão marcadas.

### R12. A ocasião é dia + hora nas duas edições

**Decisão (decidido pelo autor).** A ocasião é a chave do dia mais o horário (`V9015`), nas duas edições.

**Motivo.** Os números antigos de 2017-2018 (783 e 176) vinham da chave dia + hora + refeição (`V9017`), campo que não existe em 2008-2009. Com chaves diferentes, as duas edições não medem a mesma coisa.

**Evidência.** As listas antigas de códigos, com a chave dia + hora, reproduzem 635/101 em 2008-2009 e 793/180 e 15.863 ocasiões com refrigerante em 2017-2018. Isso conferiu o parser.

**Alternativa descartada:** manter a refeição na chave em 2017-2018. A definição do target ficaria diferente entre treino e teste.

### R13. Refrigerante é definido pelo produto, genérico ou de marca

**Decisão (decidido pelo autor).** A mesma regra nas duas edições: conta como refrigerante o produto, genérico ou de marca, e os sabores registrados por marca entram.

**Motivo.** As listas antigas não traziam os sabores registrados por marca, então subcontavam refrigerante e guaraná.

**Evidência.** Números novos, com a chave dia + hora. Em 2017-2018: 85.018 dias-pessoa, 1.681 com pizza, 797 com pizza e refrigerante, 180 com pizza e guaraná, 16.122 ocasiões com refrigerante. Em 2008-2009: 66.903, 1.149, 651, 101 e 19.182 ocasiões.

**Alternativa descartada:** manter as listas antigas, que seriam comparáveis ao `CLAUDE.md` original mas incompletas.

### Escolhas do executor

- **R9 (escolha do executor):** `ufs_midia_regional` inclui as UFs empatadas com a 7ª. No degrau A, que usa região na parte 2, isso dá as 9 UFs do Nordeste. Cortar em 7 seria arbitrário entre UFs com a mesma média.
- **R11 (escolha do executor):** as dobras da validação adversarial são agrupadas por `cod_upa`, para a mesma UPA não cair em treino e validação ao mesmo tempo.
- **R14 (escolha do executor):** `receita_commit` vem de `git describe --always --dirty`, para o artefato dizer de que versão do código saiu e se havia alterações não commitadas.

---

## 26. Clima e feriados casados com a POF

**A pergunta do autor (20/11/2026).** Se haveria uma forma boa, que contribuísse de fato para o case e para a inovação que a Ambev procura, de juntar clima e feriados à POF; e se isso ajudaria ou prejudicaria os resultados. O autor pediu uma resposta pesquisada, não de cabeça.

**O que foi verificado.** Os números estão em `reports/investigacao_datas.json`, gerado por `make investigacao-datas` (`scripts/investigacao_datas.py`). O script só faz contagens: nenhuma taxa do target foi calculada contra data, clima ou feriado, para não contaminar hipóteses futuras. De 2017-2018 só foram lidas data de nascimento e idade.

### O que os dados permitem

**2008-2009 tem data de calendário exata,** o que a auditoria não tinha visto (ela anotou só a semana, `PERIODO_REAL`).

- No dicionário, `IDADE_DIA` (registro de morador) é a idade em dias calculada entre a data de nascimento (`V04041`, `V04042`, `V04043`) e a data de início da caderneta de despesa coletiva (POF3). Nascimento + `IDADE_DIA` dá essa data.
- Nos 55.917 domicílios, todos os moradores dão a mesma data, sem divergência. As datas vão de 20/05/2008 a 18/05/2009 e batem com as semanas de `PERIODO_REAL`.
- O registro alimentar foi feito em dois dias não consecutivos durante a semana da caderneta. Dentro de uma semana cada dia da semana aparece uma vez só, então `DIA_SEMANA` fixa a data de cada dia-pessoa. Conferência: 66.885 dos 66.903 dias-pessoa recebem data; em 99,8% das pessoas o quadro 71 vem antes do 72; só 0,1% têm dias consecutivos; o padrão mais comum é o dia 0 e o dia 2 da semana.
- Isso explica a armadilha dos dias da semana: a caderneta começa quase sempre numa terça (30.677 de 55.917 domicílios), o primeiro dia de registro cai no dia 0 (terça) e o segundo no dia 2 (quinta). Daí o excesso de terças e quintas em 2008-2009.

**2017-2018 não tem data, e a reconstrução é grosseira.**

- A coleta foi de junho de 2017 a julho de 2018, por recordatório de 24 horas em dois dias não consecutivos (IBGE).
- A idade em anos (`V0403`) foi calculada na data da entrevista, não numa data fixa: só 75% dos moradores são compatíveis com 15/01/2018, e quase todos os domicílios (99,95%) têm uma janela de datas comum. Nascimento + idade limitam a data a uma janela de 202 dias de mediana por domicílio.
- Cruzar os moradores do mesmo setor não chega à semana. Em 2008-2009 as datas de um setor se espalham por 75 a 90 dias. O mesmo método, aplicado a 2008-2009 sem `IDADE_DIA` e comparado com a data verdadeira, erra 24 dias na mediana e 50 no 9º decil, e acerta o trimestre em 52% dos domicílios e o mês em 32%.

**Feriados não têm amostra.** Em 2008-2009 caem 12 dias com pizza e refrigerante em feriado nacional e 1 com pizza e guaraná; nas datas comerciais (Namorados, Pais, Crianças, véspera de Natal, Réveillon, Mães), 12 e 5; de 24/12 a 01/01, 11 e 0; no Carnaval, 3 e 0.

### Decisão

**Decisão (decidido pelo autor):** clima e feriados ficam fora do Occasion Score e fora do Plano 2. O achado fica registrado aqui, no Data Card e no README.

**Motivos para ficar fora do score:**

1. A seção 5.4 do design exige que toda variável exista nas duas edições. Clima diário e feriado existiriam no treino e não no teste final.
2. Uma data estimada no teste, com erro de cerca de um mês contra a data exata do treino, mudaria a medida entre treino e teste, e a queda de desempenho seria lida como mudança de hábito.
3. O modelo servido é reajustado com 2017-2018 (seção 12), que não tem data.
4. A POF só informa UF e estrato, sem município. O clima só seria bem localizado no estrato da capital.

**Motivo para ficar fora do Plano 2:** prazo. Em 20/11 o Plano 1 acabou de terminar, e EDA, escada de modelos, teste final e vídeo, todos obrigatórios, ainda estão pela frente.

**Alternativas descartadas:**

| Alternativa | Por que foi descartada |
|---|---|
| Clima diário como variável do score | Não existe em 2017-2018; quebra a seção 5.4 e o teste temporal |
| Estação estimada em 2017-2018 como variável | Erro de cerca de um mês, com medida diferente entre treino e teste |
| Feriados e datas comerciais como variável | 12 e 1 positivos em feriado nacional em 2008-2009 |
| Validação temporal dentro de 2008-2009 com a data exata | Possível, mas o desenho já tem o teste entre edições; fica como ideia |

**O que vira próximo passo no README:** um bloco analítico só em 2008-2009 e só nas capitais, com a pergunta "em dia mais quente que o normal para aquela capital, a participação do guaraná entre as ocasiões com refrigerante muda?". Ele usaria temperatura e chuva diárias do Open-Meteo (reanálise ERA5, desde 1940), a anomalia em relação à média da própria capital para não confundir clima com região, e a parte 2 (19.182 ocasiões com refrigerante). Seria uma associação, nunca efeito causal, registrada como hipótese antes de ser testada. Se entrar no projeto, entra no lugar do bloco do Trends, não somada a ele: o clima é comparado com consumo real, e o Trends só com interesse de busca. Um indício externo de que o tema importa: um estudo da Scanntech divulgado em 2026 associa cada 1 °C a mais a 2,9% a mais de volume de refrigerante no varejo; só a divulgação na imprensa foi encontrada, sem o relatório técnico.

---

## 27. Decisões do Plano 2

Decisões tomadas ao escrever e executar o Plano 2, até o pré-registro (`docs/pre-registro.md`). As escolhas que vieram do plano estão marcadas como **escolha do executor, aprovada pelo autor**; as que o autor tomou em 20/11/2026, depois da EDA, como **decidido pelo autor**. A aprovação acontece no ponto de parada da tarefa 6.

### Bootstrap de setores dentro da UF

**Decisão (escolha do executor, aprovada pelo autor).** Toda incerteza vem de um bootstrap que sorteia setores (`cod_upa`) inteiros, com reposição, dentro de cada UF, com `B = 1000` e intervalo percentil de 95%. As duas previsões de uma comparação usam os mesmos sorteios (bootstrap pareado).

**Motivo.** O setor é a unidade de sorteio da POF, e as linhas de um setor (a mesma pessoa, o mesmo domicílio, os vizinhos) não são independentes. Os 550 estratos de 2008-2009 têm mediana de 5 setores, e 7 têm um só. Com tão poucos setores, o bootstrap dentro do estrato subestima a variância pelo fator (n−1)/n, cerca de 20%. Por UF são dezenas a centenas de setores.

**Alternativas descartadas:** sortear linhas (ignora a dependência dentro do setor e estreita os intervalos); sortear setores dentro do estrato (subestima a variância nos estratos pequenos).

### Regra do campeão e `melhor_logistico`

**Decisão (escolha do executor, aprovada pelo autor).** As hipóteses H1 a H3 são relatadas como a seção 7 do design as define: B contra A, C contra B, D contra o melhor logístico, que é C se H2 for confirmada e B se não for (`melhor_logistico`). O degrau vencedor de cada parte sai de outra regra, a do campeão (`vencedor_escada`): o campeão começa em A, e B, C e D, nessa ordem, só assumem se superarem o campeão do momento.

**Motivo.** Se B não supera A, comparar C com B mede C contra um modelo que já perdeu. Pela regra do campeão, C e D precisam superar a tabela. Quando H1 é confirmada, as duas formas coincidem; quando não, o JSON da escada grava as duas comparações, para o relatório não se contradizer.

**Alternativa descartada:** escolher o vencedor pela cadeia das hipóteses (o último degrau confirmado). Poderia declarar vencedor um modelo que não supera a tabela.

### Calibração ajustada nas previsões internas

**Decisão (escolha do executor, aprovada pelo autor).** A correção sigmoide de cada parte, e a do produto (H7), é ajustada em cada dobra externa nas previsões das dobras internas daquela dobra e aplicada às previsões fora da dobra externa. A avaliação usa só as previsões fora da dobra externa.

**Motivo.** Ajustar a sigmoide nas próprias previsões fora da dobra externa e medir o ganho nelas faria a calibração ver as linhas em que é avaliada, e o ganho sairia otimista. As previsões internas já existem (a busca de hiperparâmetros as produz) e não tocam a dobra externa.

**Alternativa descartada:** ajustar a sigmoide nas previsões fora da dobra externa e avaliar nelas mesmas.

### Interações do degrau C por argumento, e o conjunto escolhido na validação

**Decisão (escolha do executor, aprovada pelo autor, para as interações por argumento; decidido pelo autor, para o conjunto escolhido na validação).** As interações do degrau C saem do enunciado do TAPI ("sexta a domingo, à noite e em casa") e das perguntas do projeto (a pizza muda a preferência dentro e fora de casa, e por região?), não do tamanho do efeito na EDA, embora o design diga "definidas na EDA". Depois da EDA, o autor acrescentou à ocasião o conjunto `fim_de_semana` (o dia agrupado como no TAPI) e decidiu que o conjunto é hiperparâmetro do degrau C, escolhido na validação aninhada junto com a regularização.

**Motivo.** Escolher interações pelo alvo de 2008-2009 inteiro e depois medir C contra B na mesma base deixaria H2 otimista. A EDA mostrou que dia × faixa só tem positivos à noite, e o conjunto original gasta 24 graus de liberdade nisso; o alternativo gasta 4. Como essa alternativa foi pensada depois de ver a EDA, ela não pode ser imposta: é escolhida nas dobras internas, e a comparação confirmatória continua fora da dobra.

**Alternativas descartadas:** escolher as interações pelo desvio na EDA (H2 otimista); trocar o conjunto original pelo `fim_de_semana` sem teste (decisão tomada olhando o alvo, sem validação).

### Critério de 10% a 30% de positivos por dobra

**Decisão (escolha do executor, aprovada pelo autor).** Cada dobra externa precisa ter entre 10% e 30% dos positivos de cada alvo conferido (`y_pizza_refri` e `y_pizza_guarana` nas células; `y_guarana` nas ocasiões com refrigerante e no recorte de pizza). Se não tiver, a semente avança de 100 em 100, e a troca fica registrada.

**Motivo.** O design (6.1) pede "positivos suficientes" sem dizer quanto. Com 5 dobras, a fração esperada é 20%; a faixa de 10% a 30% aceita a variação natural de sortear setores inteiros e recusa a dobra que deixaria uma comparação apoiada em poucas dezenas de casos de pizza com guaraná. O critério foi escrito no código antes de qualquer sorteio nos dados reais.

**Alternativa descartada:** um mínimo absoluto de positivos por dobra. Exigiria um número diferente por alvo, e o de pizza com guaraná (101 no total) ficaria arbitrário.

### Contribuições do LightGBM pelo TreeSHAP do próprio LightGBM

**Decisão (escolha do executor, aprovada pelo autor).** As contribuições do degrau D saem de `predict(..., pred_contrib=True)` do LightGBM, que calcula os valores SHAP de árvore em log-chances. O pacote `shap` não entra.

**Motivo.** O resultado é o mesmo (TreeSHAP exato), já na escala de log-chances que a seção 10 do design pede. O pacote `shap` traz `numba` e tem atrito conhecido com divisões categóricas.

**Alternativa descartada:** o pacote `shap`, citado na stack da seção 16 do design.

### Pasta separada para o degrau A do Plano 1

**Decisão (escolha do executor, aprovada pelo autor).** O artefato do Plano 1 (tabela de taxas) passou para `models/baseline/`. `models/avaliado/` fica só para o modelo da receita, e `carregar_artefato` procura servido, avaliado e baseline, nessa ordem.

**Motivo.** O teste final carrega `models/avaliado/` e confere que ele saiu da receita congelada. Se o degrau A do Plano 1 continuasse ali, um esquecimento faria o teste medir a tabela no lugar da receita.

**Alternativa descartada:** deixar a tabela do Plano 1 em `models/avaliado/` até a tarefa 12 sobrescrevê-la. Se a tarefa 12 falhasse ou não rodasse, o arquivo no lugar do modelo avaliado seria a tabela, sem nada no nome que avisasse.

### Degrau A escolhido na validação aninhada

**Decisão (decidido pelo autor).** O baseline deve ser o mais forte possível, escolhido por teste. O degrau A tem 12 configurações por parte (4 conjuntos de dimensões vezes 3 forças de suavização, 10, 100 e 1000), escolhidas nas dobras internas como os hiperparâmetros dos outros degraus. A configuração que `selecao_final` escolher em 2008-2009 é a referência de H8.

**Motivo.** A tabela do Plano 1 tem 21 das 70 combinações sem nenhum positivo e 50 com menos de 5 (`reports/eda.json`). Comparar B, C e D com uma tabela que perde por ruído faria H1 e H8 dizerem pouco.

**Alternativa descartada:** manter a tabela do Plano 1 como degrau A fixo, ou trocá-la por outra escolhida olhando a EDA.

### Faixas de horário e cortes mantidos

**Decisão (decidido pelo autor).** As faixas de horário ficam como no Plano 1, e os cortes de `classification` (80 e 50) e do Decision Engine (75 e 50) ficam como no design, com a análise de sensibilidade da tarefa 11.

**Motivo.** A EDA não mostrou fronteira cortando pico no alvo da parte 1: a noite reúne 87,5% das ocasiões de pizza com refrigerante, com pico às 20h. O único pico dividido é o das 15h na série de todas as ocasiões com refrigerante, que não é o alvo da parte 1. Os cortes não têm um custo de erro que justifique outro valor (seção 6.2 do design).

**Alternativa descartada:** mover a fronteira entre almoço e tarde para juntar 14h e 15h. Mudaria as bases, a auditoria e o contrato por uma série que não é o alvo.

### Corte do bloco do Google Trends (H10)

**Decisão (decidido pelo autor).** O bloco do Google Trends foi cortado em 20/11/2026, antes de qualquer cálculo com a POF. H10 sai como "não executada" no relatório de hipóteses. Os três CSV exportados e o `LEIAME.md` ficam em `dados_externos/google_trends/` como registro; nenhum número deles entra no score, na recomendação ou nos resultados.

**Motivo.** A extração mostrou que a medida não sustenta H10:

1. **Resolução.** Na comparação por estado, o site dá a participação de cada tópico em porcentagem inteira. A do guaraná fica entre 1% e 11% em 2008-2009 e quase toda em 2% em 2017-2018, o que não permite ordenar os estados.
2. **Ruído de marca.** Os tópicos juntam buscas que não são sobre a bebida, como bonés e coleções de maquiagem licenciadas, e o público delas tem geografia própria.
3. **Volume em 2008-2009.** Na comparação, 15 dos 27 estados vêm sem valor para o guaraná.

**Mudança de medida que chegou a ser cogitada antes do corte.** Para contornar o arredondamento, foram consideradas exportações individuais por tópico (índice de 0 a 100 por estado, um arquivo para guaraná e outro para cola em cada janela). A correlação de postos da razão guaraná/cola não depende da escala de cada tópico, então a comparação seguiria válida. Também foi testado o filtro de categoria "Alimentos e bebidas" no lugar de "Todas as categorias" (seção 13 do design), para tirar o ruído de marca. O filtro não removeu o ruído: as buscas de boné e maquiagem continuaram entre as mais frequentes. Com o ruído mantido, e com 15 estados vazios na comparação de 2008-2009, o autor preferiu cortar o bloco a trocar a medida.

**Alternativas descartadas:** rodar H10 com a participação arredondada (sem variação para ordenar os estados); trocar a medida pelas exportações individuais com o filtro "Alimentos e bebidas" (o ruído de marca continuaria, e o volume de busca de 2008-2009 seguiria baixo); substituir o Trends pelo bloco de clima nas capitais (seção 26), que fica como próximo passo no README.

---

## Fontes consultadas

**Dados e POF**

- [Servidor do IBGE: Orçamentos Familiares](https://ftp.ibge.gov.br/Orcamentos_Familiares/)
- [Comunicado do IBGE sobre os primeiros resultados da POF 2024-2025](https://www.ibge.gov.br/comunicados/48392-ibge-divulgara-em-27-de-novembro-a-pesquisa-de-orcamentos-familiares-pof-2024-2025-primeiros-resultados) (não foi possível abrir; a informação vem do resumo da busca)
- [Página da POF 2024-2025 no IBGE](https://www.ibge.gov.br/pof2024/) (não foi possível abrir)
- [IBGE prevê início da divulgação da POF em novembro (Canal Rural)](https://www.canalrural.com.br/economia/ibge-diz-que-microdados-do-censo-2022-seguem-sem-data-e-preve-inicio-da-pof-em-novembro/)
- [Limitations in the comparison of the Brazilian National Dietary Surveys of 2008–2009 and 2017–2018](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9586432/)
- [Fatos e perspectivas do primeiro Inquérito Nacional de Alimentação](https://www.scielo.br/j/rsp/a/3XsQD7ph8jzXN6kDG9XQ8Vk/?lang=pt)
- [Catálogo da FAO: POF, consumo alimentar pessoal](https://microdata.fao.org/index.php/catalog/2869)
- [Estudo com microdados do Vigitel sobre refrigerante](https://scielosp.org/pdf/csc/2020.v25n7/2529-2540/pt)
- [IBGE: POF 2008-2009, consumo alimentar](https://www.ibge.gov.br/estatisticas/sociais/saude/24786)
- [Bezerra e colegas, Revista de Nutrição: registro alimentar da POF 2008-2009](https://periodicos.puc-campinas.edu.br/nutricao/article/download/9304/6682/35393)
- [Release do IBGE sobre a POF 2017-2018](https://agenciadenoticias.ibge.gov.br/agencia-sala-de-imprensa/2013-agencia-de-noticias/releases/28646-pof-2017-2018)
- [Recordatório de 24 horas da POF 2017-2018 (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8969488)

**Clima e feriados**

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [ERA5 no Copernicus](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels)
- [Estudo da Scanntech sobre temperatura e bebidas (Mundo do Marketing)](https://mundodomarketing.com.br/cada-1-c-a-mais-eleva-em-4-1-vendas-de-bebidas-sensiveis-ao-calor-aponta-estudo)
- [python-holidays: Brasil](https://python-holidays.readthedocs.io/en/latest/_autosummary/holidays.countries.brazil.html)

**Delivery**

- [Como integrar a API do iFood (TabNews)](https://www.tabnews.com.br/dhpn/como-integrar-a-api-do-ifood-pra-consumir-somente-o-endpoint-de-tracking)
- [Open iFood SDK](https://openifood.gitbook.io/open-ifood-sdk/)
- [Scraping de iFood funcionou até parar de funcionar (TabNews)](https://www.tabnews.com.br/dataminer/scraping-de-ifood-funcionou-ate-parar-de-funcionar)
- [Brazilian E-Commerce Public Dataset by Olist](https://github.com/spdrio/Brazilian-E-Commerce-Public-Dataset-by-Olist)
- [Rappi E-Receipt Data (Datarade)](https://datarade.ai/data-products/rappi-e-receipt-data-food-delivery-transactions-alternativ-measurable-ai)

**Método**

- [How to avoid machine learning pitfalls (Lones)](https://arxiv.org/html/2108.02497v5)
- [Leakage and the Reproducibility Crisis in ML-based Science (Kapoor e Narayanan)](https://reproducible.cs.princeton.edu/)
- [REFORMS: Reporting Standards for Machine Learning Based Science](https://arxiv.deeppaper.ai/papers/2308.07832v2)
- [Survey-aware Machine Learning (SaML)](https://arxiv.org/abs/2605.08963)
- [Cross-validation in complex survey data](https://notstatschat.rbind.io/2024/05/21/crossvalidation-in-complex-survey-data/)
- [surveyCV (Wieczorek e colegas)](https://packages.oit.ncsu.edu/cran/web/packages/surveyCV/readme/README.html)
- [pmsampsize, critérios de Riley e colegas](https://rdrr.io/pkg/pmsampsize/man/pmsampsize.html)
- [Calibration: the Achilles heel of predictive analytics (Van Calster e colegas)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912996)
- [TRIPOD+AI statement](https://research.birmingham.ac.uk/en/publications/tripodai-statement-updated-guidance-for-reporting-clinical-predic/)
- [The ML Test Score (Breck e colegas)](https://research.google/pubs/pub46555/?authuser=2)
- [Rules of Machine Learning (Zinkevich)](https://developers.google.com/machine-learning/guides/rules-of-ml/)
- [Adversarial validation](https://apxml.com/courses/monitoring-managing-ml-models-production/chapter-2-advanced-drift-detection/adversarial-validation-drift)
- [Exploring pre-registration for predictive modeling](https://statmodeling.stat.columbia.edu/2023/12/06/exploring-pre-registration-for-predictive-modeling/)
- [Beyond Accuracy: Behavioral Testing with CheckList](https://arxiv.org/pdf/2005.04118)
- [Validating and testing ML pipelines with pandera](https://flyte.readthedocs.io/en/latest/flytesnacks/examples/pandera_plugin/validating_and_testing_ml_pipelines.html)
