# Interface aprimorada (React + TypeScript)

Fase 10. As mesmas funcionalidades da interface simples, com design próprio, animações
suaves e treino em tempo real. Fala com o backend só através da API (HTTP para pedidos,
WebSocket para o treino) e nunca importa código Python.

Tecnologias: React 19, TypeScript 6, Vite 8, Tailwind CSS 4, shadcn/ui (com o Radix) e
Motion; TanStack Router e TanStack Query; o cliente gerado pelo `@hey-api/openapi-ts`; D3 para
os gráficos e KaTeX para as fórmulas. As razões estão na ADR 0027; o sistema visual, na
ADR 0029.

Este ficheiro tem duas partes: como correr a interface e o **guia do contrato com o
backend**, tudo o que o frontend precisa de saber para falar com a API sem adivinhar (as
razões estão nas ADRs 0019, 0022 e 0026).

## Como correr

Precisa do Node 24 (traz o npm). Na primeira vez, nesta pasta:

```bash
npm install
```

Depois, a API e a interface, cada uma no seu terminal (a interface abre em
<http://localhost:5173>):

```bash
uv run python -m ml_lab.api
npm run dev
```

Os pedidos da interface vão para `/api`, e o Vite passa-os à API no endereço de
`ML_LAB_API_HOST` e `ML_LAB_API_PORT` (os mesmos do `.env` da raiz do projeto).

| Comando | O que faz |
|---|---|
| `npm run api` | Gera o cliente (`src/api/generated/`) a partir do `openapi.json`. Corre sozinho antes dos outros |
| `npm run dev` | O servidor de desenvolvimento, que recarrega ao gravar |
| `npm run typecheck` | Os tipos (o "mypy" do frontend) |
| `npm run lint` | O ESLint, sem avisos (o "ruff") |
| `npm run format` | O Prettier, que também ordena as classes do Tailwind |
| `npm test` | Os testes do Vitest (o "pytest") |
| `npm run e2e` | Os testes de ponta a ponta, no Chromium, com a API numa pasta temporária |
| `npm run build` | O build final, em `dist/` |

Depois de mudar um serviço do backend: `uv run python scripts/export_openapi.py` na raiz e
`npm run api` aqui. Os tipos que deixarem de bater certo aparecem no `npm run typecheck`.

## Organização

```
src/
├── api/         o cliente (gerado em generated/), os erros, a cache, o vocabulário, os temas,
│                a demonstração de um modelo e o acompanhamento pelo WebSocket (stream.ts)
├── components/  ui/ (os do shadcn, adaptados), shell/ (a moldura), charts/ (os gráficos, em
│                D3 e SVG), forms/ (o formulário gerado de um JSON Schema), markdown/ (o texto
│                com fórmulas), data/ (a tabela de dados), notices/ (os avisos)
├── features/    um ecrã por pasta: home, data, explore, prepare, models, training, agents
├── hooks/       hooks partilhados (o tamanho de um elemento, um valor com atraso)
├── lib/         format.ts (os números à portuguesa), colors.ts e classes.ts (as cores dos
│                dados), utils.ts
├── routes/      um ficheiro por rota, que só aponta para o ecrã (o TanStack Router gera
│                routeTree.gen.ts)
├── state/       o pouco estado do cliente (session.ts)
└── styles/      globals.css: os tokens do sistema visual
```

## A API

```bash
uv run python -m ml_lab.api
```

- Responde em <http://127.0.0.1:8000>, com a documentação interativa em `/docs`.
- Por omissão, o CORS deixa entrar o servidor do Vite (`http://localhost:5173` e
  `http://127.0.0.1:5173`). Para outras origens: `ML_LAB_API_CORS_ORIGINS` (ver
  `.env.example`).
- `GET /health` diz se a API está a correr, e em que versão.

## O contrato: `openapi.json`

`ui/web/openapi.json` é o OpenAPI da API, guardado no Git. Gera-se com
`uv run python scripts/export_openapi.py`, e um teste falha quando fica desatualizado. É daqui
que saem os tipos TypeScript (por exemplo, com o `openapi-typescript`), sem a API a correr.

O que o esquema garante:

- **cada operação chama-se `<serviço>_<método>`** (`experiments_start`,
  `automl_updates`): é o nome do método do serviço que a rota chama;
- **cada tipo tem um só esquema, com um nome simples** (`ExperimentConfig`, nunca
  `ExperimentConfig-Input`);
- **nas respostas, todos os campos vêm sempre** (todos obrigatórios no esquema). Nos tipos
  que também se enviam (`ExperimentConfig`, `PreparationConfig`, `AutoMLRequest`…), um
  campo com valor por omissão pode faltar no pedido;
- **um número nunca chega `null` onde o esquema diz `number`**: os NaN passam a `null` só
  nos campos que o esquema já declara que podem ser `null` (um teste confirma-o com
  resultados reais);
- **os corpos dos pedidos recusam campos que não conhecem** (422, `invalid_request`): um
  nome mal escrito não passa em silêncio.

## Ao arrancar: os textos, uma vez

Os textos escrevem-se uma vez, no backend (regra 5 do `CONTRIBUTING.md`). Pede-se e guarda-se:

- `GET /vocabulary` tem duas partes:
  - `labels`: o nome a mostrar de cada valor de cada enum, pelo nome do tipo. Mostra-se
    sempre `labels.TaskType["binary_classification"]` ("Classificação binária"), nunca o
    valor cru. Um teste garante que cada enum que a API devolve está lá, com todos os
    valores;
  - `tasks`: o que muda com cada tarefa (o paradigma, se precisa de alvo, a métrica
    principal, o modelo sugerido, se os agentes trabalham nela);
  - `step_metrics`: as métricas que os modelos reportam em cada passo do treino, além da
    loss (`StepRecord.metrics`: `accuracy`, `val_loss`, `oob_score`), com o nome a mostrar,
    se são percentagem e se são uma loss (a de validação desenha-se com a do treino).
- `GET /explanations` dá todos os temas (`{tema: {title, text}}`), e
  `GET /explanations/{tema}` um só. São os do painel "Porquê?" e também o "como ler" de cada
  gráfico (`correlation_matrix`, `confusion_matrix`, `lineage`, `train_vs_test`…): a interface
  mostra-os por baixo dos gráficos. Pedem-se ao arrancar, como o vocabulário, e a interface
  pede cada um pelo nome; `tests/contract/test_topics_in_use.py` procura esses nomes no código
  das duas interfaces e falha se algum não existir. Os temas que explicam a página de cada
  modelo vêm no cartão (`topics`), e o "como ler" do campo de decisão vem na própria animação
  (`TrainingAnimation.how_to_read`), porque depende do que os valores são.
- `GET /catalog/metrics` dá cada métrica com o que mede, como se lê, a **fórmula em LaTeX**
  e o significado de cada símbolo. Não muda de run para run: pede-se uma vez. Os números de
  um run vêm nos relatórios: cada `MetricResult` traz os `terms` da fórmula (os VP e os FP da
  precisão, as somas do R²…), com o mesmo `symbol` que aparece na fórmula; na multiclasse,
  cada termo tem o `group` (o número da classe), porque a métrica é a média das classes.

Os textos vêm em Markdown simples (negrito, listas). As fichas didáticas (`doc` nos cartões do
catálogo, dos agentes de RL e nas opções dos agentes) têm fórmulas em LaTeX (`$...$` e
`$$...$$`): mostram-se com o KaTeX.

## Trabalho em segundo plano

O treino de um modelo, o de um agente de RL e a procura dos agentes funcionam da mesma
maneira:

| | Começar (202) | Acompanhar | Controlar | Resultado |
|---|---|---|---|---|
| Modelo | `POST /experiments` | `GET /experiments/{run_id}/updates` | `.../pause`, `.../resume`, `.../cancel` | `GET /experiments/{run_id}/result` |
| Agente de RL | `POST /reinforcement` | `GET /reinforcement/{run_id}/updates` | idem | `GET /reinforcement/{run_id}/result` |
| Agentes (AutoML) | `POST /automl` | `GET /automl/{search_id}/updates` | idem | `GET /automl/{search_id}/result` |

- A resposta de começar é um `JobState`: o `run_id` (na procura dos agentes, o id da procura,
  `search-...`), o estado e o progresso (`planned_steps` dá a barra de progresso).
- `updates` devolve só o que é novo, a partir de dois cursores que o frontend guarda:
  - modelos e agentes de RL: `after_step` (o último passo que já tem) e `after_snapshot` (o
    passo da última fotografia);
  - procura dos agentes: `after_step` (o último candidato) e `after_message` (a última
    mensagem da conversa).
- **WebSocket** (não aparece no OpenAPI): `/experiments/{run_id}/stream`,
  `/reinforcement/{run_id}/stream` e `/automl/{search_id}/stream` enviam o mesmo que
  `updates`, sozinhos, de 0,25 em 0,25 s, em mensagens `{"kind": "update", "data": ...}`
  (ou `{"kind": "error", "data": <ApiError>}`). A ligação fecha quando o trabalho termina;
  o resultado pede-se então por HTTP.
- Na procura dos agentes, cada `updates` traz também o `working`: o candidato que um agente
  está a medir agora, o que muda nele e o fold em que vai (`fold` de `folds`). É `null` entre
  candidatos e fora das rondas. É o que acende o agente certo no fluxograma.
- Pedir o resultado antes do fim responde 409 (`training_in_progress`).
- `POST /automl/{search_id}/candidates/{number}/train` treina um candidato à mão: devolve o
  `JobState` de um treino normal, que se acompanha nas rotas de `/experiments`.

## Erros

O corpo de um erro é sempre um `ApiError`: `{code, message, problems}`.

- `message` está em português e diz o que fazer: mostra-se como está.
- `code` é estável (o nome do erro, em `snake_case`) e serve para o frontend decidir.
- `problems` só vem num pedido inválido, com uma frase por problema.

| HTTP | Quando | Códigos |
|---|---|---|
| 404 | O identificador não existe | `run_not_found`, `dataset_not_found`, `job_not_found`, `search_not_found`, `candidate_not_found`, `run_not_in_memory`, `component_not_found` |
| 409 | O trabalho ainda decorre, ou falhou | `training_in_progress`, `training_failed` |
| 422 | Os valores do pedido não servem | `invalid_request`, `hyperparameter`, `invalid_parameter`, `column_not_found`, `data`, `data_source`, `data_validation`, `invalid_seed` |
| 400 | O pedido não se pode cumprir | `model`, `capability`, `not_fitted` |
| 500 | O registo falhou, ou um erro inesperado | `tracking`, `unexpected` |

`run_not_in_memory` quer dizer que o run saiu da memória (só os últimos ficam): abre-se
outra vez com `GET /history/runs/{run_id}/result`.

## Números

Os números vêm em bruto, com o que é preciso para os mostrar; a formatação é do frontend, à
portuguesa (`Intl.NumberFormat("pt-PT")`), como em `ml_lab/formatting.py`:

- `as_percent: true`: percentagem com uma casa decimal (`0.8125` → `81,2%`);
- as outras métricas: três casas decimais (`0,612`);
- as contagens (`higher_is_better: null` e um valor inteiro): sem casas;
- `null`: sem valor, mostra-se `—`.

## O que se calcula com o que já vem

Algumas coisas que a interface simples lê de propriedades do Python não vêm no JSON, porque
se calculam com campos que vêm:

| Precisas de | Calcula-se com |
|---|---|
| A métrica principal de um run | `tasks[task_type].primary_metric` do vocabulário, procurada em `reports[main_partition].metrics` |
| O relatório que resume um run | `reports[main_partition]` |
| A seed de um run | `config.seed` |
| Os passos do treino | o último `history[].step` |
| Se há animação | `animation.frames.length > 1` |
| A parte de nulos de uma coluna | `n_missing / n_rows` |
| O valor principal de um run no histórico | `metrics[primary_metric]` |
| O vencedor dos agentes | `candidates[winner - 1]` |
| Se a procura usou uma amostra | `search_rows < train_rows` |
| Se um candidato falhou | `error !== null` |
| A linha e a coluna de uma célula da grelha | `Math.floor(estado / columns)` e `estado % columns` |
| A parte de células em falta de um dataset | `n_missing_cells / (n_rows × n_columns)` |
| Quantas colunas há de cada tipo | contar `columns[].kind` no perfil |
| Todos os avisos do perfil | `notices` e os de cada coluna (`columns[].notices`), pela ordem das colunas |
| O equilíbrio das classes | `classes.at(-1).count / classes[0].count` (vêm da mais para a menos frequente) |
| As colunas de uma matriz de confusão | `column_labels`; vazio quer dizer as mesmas classes de `labels` (e a diagonal são os acertos) |
| Os avisos do diagnóstico de um run | `notices` sem os que também estão em `training_notices` |
| A cor de cada classe | a posição em `class_labels`; no perfil, que não os traz, a mesma ordem do backend (numérica se os valores forem todos números, senão a do texto) |

As regras de domínio vêm sempre do backend, já decididas: se os dados podem seguir
(`validation.is_usable`), que divisões existem para uma tarefa
(`GET /datasets/task-options`), qual é a classe anómala, o que o modelo aprendeu
(`learned`), o vencedor dos agentes (`winner`, `ranking`).

## Tamanho das respostas

O resultado de um modelo iterativo (com a animação da fronteira e as camadas de uma rede)
ronda os 2 MB de JSON: pede-se uma vez e guarda-se. Durante o treino, cada `updates` traz só
a fotografia nova. A API corre no mesmo computador, por isso não comprime as respostas:
comprimir 2 MB custa mais tempo do que os poupa numa ligação local.

## Os ecrãs e as rotas

| Ecrã | Operações |
|---|---|
| 1 · Dados | `datasets_samples`, `datasets_generators`, `datasets_load_sample`, `datasets_generate`, `datasets_upload`, `datasets_load`, `datasets_loaded`, `datasets_summary`, `datasets_preview` |
| 2 · Agentes | `automl_options`, `automl_plan`, `automl_start`, `automl_search_plan` (o plano de uma procura, para desenhar os agentes enquanto ela decorre), `automl_updates` (e o WebSocket), `automl_pause`, `automl_resume`, `automl_cancel`, `automl_result`, `automl_train_candidate`; o vencedor é um run como os outros (`experiments_result`, ou `history_open` depois de a API recomeçar) |
| 3 · Explorar | `datasets_profile` |
| 4 · Preparar | `datasets_suggest_task`, `datasets_task_options`, `datasets_suggest_cleaning`, `datasets_cleaning_operations`, `datasets_prepare` |
| 5 · Modelo | `catalog_models`, `catalog_model`, `catalog_suggested_model`, `catalog_validate_hyperparameters`, `catalog_demo`, `experiments_start` |
| 6 · Treinar | `experiments_updates` (e o WebSocket), `experiments_pause`, `experiments_resume`, `experiments_cancel`, `experiments_job`, `experiments_preview`, `experiments_result` |
| 7 · Avaliar | `experiments_result`, `analysis_charts`, `analysis_cross_validation`, `analysis_learning_curve`, `catalog_metrics` |
| 8 · Explicar | `analysis_permutation_importance`, `analysis_partial_dependence`, `analysis_test_rows`, `analysis_explain_row` |
| 9 · Experiências | `history_runs`, `history_entry`, `history_open`, `history_open_reinforcement`, `history_save`, `history_rename`, `history_delete`, `history_compare`, `history_reproduce` |
| 10 · Laboratório | `laboratory_overfitting`, `laboratory_learning_rate`, `laboratory_knn`, `laboratory_regularization` |
| 11 · Reinforcement learning | `reinforcement_environments`, `reinforcement_agents`, `reinforcement_start`, `reinforcement_updates` (e o WebSocket), `reinforcement_pause`, `reinforcement_resume`, `reinforcement_cancel`, `reinforcement_job`, `reinforcement_result` |
| Todos | `explanations_vocabulary`, `explanations_topics`, `explanations_explain`, `meta_health` |

Na interface web, o Treinar (6) e o Avaliar (7) são um só ecrã, "Treinar e avaliar": o treino
ao vivo e, quando acaba, a avaliação, com uma gaveta por métrica (a fórmula de
`catalog_metrics` e os `terms` do run).

O WebSocket de uma procura manda uma mensagem também quando só muda o trabalho a decorrer (o
candidato que um agente mede, o fold em que vai, a fase): é com isso que o fluxograma mostra
quem trabalha agora, entre um candidato e o seguinte.

A página de cada modelo usa o cartão (`catalog_model`: a ficha já dividida em `sections`,
as demonstrações do Laboratório que se aplicam em `laboratory` e os conjuntos de exemplo em
`examples`) e a demonstração (`catalog_demo`). A demonstração devolve o mesmo que um run (o
campo em cada fotografia, a loss, o que aprendeu, os relatórios e os gráficos), mas não é um
run: não tem `run_id`, não vai para o histórico e não se abre nas rotas de `/experiments`.
Pedir outra vez os mesmos valores é imediato (as últimas ficam guardadas).

Os caminhos de cada operação estão no `openapi.json`. Os formulários (os hiperparâmetros de
um modelo, os parâmetros de um gerador ou de um ambiente, o tamanho da procura dos agentes)
geram-se do JSON Schema que vem nos cartões, com as extensões `x-scale` (`log`: um seletor
em escala logarítmica) e `x-labels` (o nome a mostrar de cada opção).
