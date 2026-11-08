# A interface do Orbyte (React + TypeScript)

Feita na fase 10, é a interface do Orbyte (antes ML Lab; o pacote Python continua `ml_lab`): a
interface simples, em Streamlit, saiu depois dela (ADR 0031). Tem design próprio, animações
suaves e o treino em tempo real. Fala com o backend só através da API (HTTP para os pedidos,
WebSocket para o treino) e nunca importa código Python. Tudo o que mostra está em inglês, venha
dela ou do backend (ADR 0043); os comentários e esta documentação ficam em português.

Tecnologias: React 19, TypeScript 6, Vite 8, Tailwind CSS 4, shadcn/ui (com o Radix) e
Motion; TanStack Router e TanStack Query; o cliente gerado pelo `@hey-api/openapi-ts`; D3 para
os gráficos e KaTeX para as fórmulas. As razões estão na ADR 0027; o sistema visual, na
ADR 0032 (com a letra dos títulos e os rótulos da ADR 0045); as duas interfaces (Manual e
Agentes) e a entrada, na ADR 0033.

Este ficheiro tem três partes: como correr a interface, **como mexer nela** (que ficheiro muda
o quê) e o **guia do contrato com o backend**, tudo o que o frontend precisa de saber para
falar com a API sem adivinhar (as razões estão nas ADRs 0019, 0022 e 0026).

## Como correr

Precisa do Node 24 (traz o npm). Na primeira vez, nesta pasta:

```bash
npm install
```

Depois, a API e a interface juntas, com um só comando na raiz do projeto (a interface abre em
<http://localhost:5173> e recarrega sozinha ao gravar; `Ctrl+C` pára as duas):

```bash
uv run python scripts/dev.py
```

Na primeira vez, e depois de um `git pull` que mude o `package-lock.json`, o script instala as
dependências com `npm ci` (as versões exatas do lockfile). Também se arrancam à parte, cada uma
no seu terminal: `uv run python -m ml_lab.api` na raiz e `npm run dev` aqui.

Os pedidos da interface vão para `/api`, e o Vite passa-os à API no endereço de
`ML_LAB_API_HOST` e `ML_LAB_API_PORT` (os mesmos do `.env` da raiz do projeto).

| Comando | O que faz |
|---|---|
| `npm run api` | Gera o cliente (`src/api/generated/`) a partir do `openapi.json`. Corre sozinho antes do `dev`, do `build` e do `check` |
| `npm run dev` | O servidor de desenvolvimento, que recarrega ao gravar |
| `npm run check` | A verificação toda: gera o cliente e corre o lint, os tipos e os testes |
| `npm run typecheck` | Os tipos (o "mypy" do frontend), com o cliente já gerado |
| `npm run lint` | O ESLint, sem avisos (o "ruff") |
| `npm run format` | O Prettier, que também ordena as classes do Tailwind; `npm run format:check` só confere. Aceita os fins de linha de cada ficheiro (`endOfLine: "auto"`): no Windows, o Git costuma dar CRLF, e o repositório guarda LF na mesma (ADR 0054) |
| `npm test` | Os testes do Vitest (o "pytest") |
| `npm run e2e` | Os testes de ponta a ponta, no Chromium, com a API numa pasta temporária |
| `npm run build` | O build final, em `dist/` |

O lint, os tipos e os testes não geram o cliente: usam o que lá está. Gerar enquanto o `dev`
corre reescrevia os ficheiros que ele vigia, e o Vite chegou a ir abaixo com isso (ADR 0046).
Por isso, depois do `npm install` de um clone novo, ou de mudar um serviço do backend
(`uv run python scripts/export_openapi.py` na raiz), corre `npm run api` aqui, ou o
`npm run check`. Os tipos que deixarem de bater certo aparecem no `npm run typecheck`.

## Organização

```
src/
├── api/         o cliente (gerado em generated/), os erros, a cache, o vocabulário, os temas,
│                a demonstração de um modelo e o acompanhamento pelo WebSocket (stream.ts)
├── components/  ui/ (os do shadcn, adaptados), shell/ (a moldura), explain/ (o "?"), charts/
│                (os gráficos, em D3 e SVG), forms/ (o formulário gerado de um JSON Schema),
│                markdown/ (o texto com fórmulas), data/ (a tabela de dados), notices/ (os
│                avisos)
├── features/    um ecrã por pasta: home, data, explore, prepare, models, training, explain,
│                save, agents, history, laboratory, metrics, reinforcement
├── hooks/       hooks partilhados (o tamanho de um elemento, um valor com atraso)
├── lib/         format.ts (os números, como o site os escreve), colors.ts e classes.ts (as cores dos
│                dados), seed.ts (a seed de um pedido), keys.ts (os atalhos), utils.ts
├── routes/      um ficheiro por rota, que só aponta para o ecrã (o TanStack Router gera
│                routeTree.gen.ts)
├── state/       o pouco estado do cliente (session.ts)
└── styles/      globals.css: os tokens do sistema visual
```

## Mexer na interface

Um mapa do que muda o quê, para pôr a interface como queres sem partir o contrato com o
backend.

| Para mudar | Onde |
|---|---|
| As cores da moldura (o fundo, os cartões, o texto, os contornos) | `src/styles/globals.css`, no `:root`: cada cor tem o nome da ADR 0032 (`--ground`, `--surface`, `--text-soft`, `--line`…), e os papéis que os componentes leem apontam para elas (`--background`, `--card`, `--primary`, `--ring`…) |
| Os tipos de letra e a escala | `src/main.tsx` (a Geist e a Geist Mono) e o `@theme` de `globals.css` (`--font-sans`, `--font-mono` e os tamanhos) |
| As cores dos dados (as classes, as séries, as escalas) | `src/lib/colors.ts` e as `--data-1` a `--data-8` de `globals.css`: a paleta categórica e as rampas. Uma paleta nova tem de se validar outra vez contra o cartão (`#191919`): o contraste, e a distinção para quem não vê bem as cores |
| A cor de cada família de modelos (a mesma nos Modelos, nos agentes e na entrada) | `src/lib/families.ts` (ADR 0037): uma cor fixa por família, das oito da paleta; uma família nova no backend precisa da sua aqui |
| O desenho animado de cada modelo, na lista dos Modelos | `src/features/models/glyphs.ts` (que desenho tem cada família e tarefa), `model-glyph.tsx` (os desenhos) e as classes `glyph-*` de `globals.css` (o movimento) |
| Os componentes de base (botões, campos, gavetas, separadores) | `src/components/ui/`: são os do shadcn, copiados para o projeto para se adaptarem. As variantes dos estados que eles usam (`data-open:`, `data-checked:`…) estão no `globals.css`; a ferramenta do shadcn não fica instalada, e corre-se quando é precisa: `npx shadcn@latest add <nome>` (ADR 0055) |
| A moldura (a barra lateral, a pesquisa, o cabeçalho das páginas) | `src/components/shell/`; o cartão da secção aberta que desliza e o ponto do trilho estão em `app-sidebar.tsx` (`SLIDE`) |
| Os blocos das páginas: o cartão (`Panel`), o número grande (`Stat`), a secção (`Section`) e os factos | `src/components/shell/blocks.tsx`; a classe `panel` e o rótulo `mono-label` estão em `globals.css` |
| O "?" das explicações: ao lado de um título (`Explain`) e no título da página (`PageExplain`) | `src/components/explain/explain.tsx` |
| A ordem da barra, os nomes e os ícones dos ecrãs | `src/components/shell/screens.ts`: `FIRST_STEPS` são os primeiros passos das duas interfaces (por cima da troca), `NAVIGATION` os grupos de cada uma (o primeiro continua os passos), e `ONLY_IN` os ecrãs que só existem numa |
| O botão de prosseguir e o fim de cada passo | `Proceed` e `NextStep` em `src/components/shell/blocks.tsx`; o aspeto é a variante e o tamanho `proceed` do `Button` |
| A página de entrada | `src/features/home/` (fica fora da moldura: não tem barra) |
| Um gráfico (todos partilham a moldura de `chart.tsx`: os eixos, a legenda, a dica e o "Como se lê") | `src/components/charts/` |
| Um ecrã (a disposição, as secções, o que aparece primeiro) | `src/features/<ecrã>/`; a rota, em `src/routes/`, só aponta para ele |
| Os textos didáticos (o "?" de cada coisa, o "Como se lê" dos gráficos, os avisos, as fichas) | No backend, não aqui (regra 5 do `CONTRIBUTING.md`): os temas em `core/*/explanations.py` e `core/evaluation/topics.py`, as fichas em `docs/algorithms/` |

O sistema visual tem algumas regras que dão coerência ao todo (ADR 0032):

- a moldura é preto, cinzentos escuros e branco; a cor é dos dados, e segue a entidade (a
  classe "1" tem a mesma cor em todos os gráficos);
- as páginas largas ocupam o ecrã, com 2 cm de margem de cada lado (ADR 0034);
- cada coisa vive numa secção aberta (`Panel`: o nome e o "?"), não numa caixa, e **o espaço
  separa, não as linhas** (ADR 0036): 12 a 16 px dentro de um grupo, 40 a 48 px entre blocos
  (`gap-x-12 gap-y-10`), 64 a 80 px entre secções, e mais espaço por cima de um título do que
  por baixo. Só há linhas onde elas dizem alguma coisa (as tabelas, um total, um eixo). Os
  números grandes ficam numa faixa (`StatStrip`). A caixa (`panel`) é só para o que se escolhe,
  os avisos de um passo em falta e os diálogos;
- prosseguir é sempre o mesmo botão (`Proceed`), no topo à direita e no fim da página
  (`NextStep`); os outros botões da página são secundários;
- as explicações não ficam à vista: cada título tem o seu "?", e os textos vêm do backend pelo
  nome (`topic="…"`). Também não há subtítulos nem números (01, 02…): o título diz o que é, o
  contexto de um run é um número da faixa, e o resto vai para o "?" (ADR 0038);
- a cor de uma coisa (a família de um agente, como acabou um episódio) é o contorno do
  retângulo dela: inteiro quando está escolhida ou ativa, fraco quando não está, e sempre com o
  nome escrito ao lado. Nada de bolas coloridas nem de brilhos à volta (ADR 0038);
- o texto é Geist; os números pequenos, os eixos e os rótulos são Geist Mono (`mono-label`);
  dentro dos gráficos, os nomes das categorias ficam em Geist, que cabe nas margens;
- um gráfico nunca tem dois eixos de y, e tem legenda a partir de duas séries;
- o texto pequeno nunca é mais escuro do que `--text-dim` (4,5:1 no cartão);
- as páginas reorganizam-se pela largura que têm (`@container/main`), não pela do ecrã.

Enquanto se mexe, o `uv run python scripts/dev.py` recarrega a página ao gravar. Antes de dar
uma mudança por pronta: `npm run check`, e `npm run e2e`. Os
testes de ponta a ponta procuram os elementos pelo papel e pelo nome visível
(`getByRole("button", { name: "Treinar o agente" })`): ao mudar o texto de um botão ou de um
título, muda-se também no teste que o usa (procura-se o texto em `e2e/`).

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
    loss (`StepRecord.metrics`: `accuracy`, `val_loss`, `oob_score`), e as de cada episódio de
    um agente de RL (`reward`, `length`, `success`, `epsilon`), com o nome a mostrar, se são
    percentagem e se são uma loss (a de validação desenha-se com a do treino).
- `GET /explanations` dá todos os temas (`{tema: {title, text}}`), e
  `GET /explanations/{tema}` um só. São os do "?" de cada página e de cada secção, e também o
  "como ler" de cada gráfico (`correlation_matrix`, `confusion_matrix`, `lineage`,
  `train_vs_test`…): a interface mostra-os no "?" ao lado do que explicam (`Explain`, ADR
  0032), e não à vista. Pedem-se ao arrancar, como o vocabulário, e a interface
  pede cada um pelo nome; `tests/contract/test_topics_in_use.py` procura esses nomes no código
  da interface e falha se algum não existir. Os temas que explicam a página de cada
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
| Pergunta ao analista | `POST /automl/{search_id}/questions` | `GET /automl/questions/{question_id}/updates` | `.../cancel` (sem pausa) | `GET /automl/questions/{question_id}/result` |

- A resposta de começar é um `JobState`: o `run_id` (na procura dos agentes, o id da procura,
  `search-...`), o estado e o progresso (`planned_steps` dá a barra de progresso).
- `updates` devolve só o que é novo, a partir de dois cursores que o frontend guarda:
  - modelos e agentes de RL: `after_step` (o último passo que já tem) e `after_snapshot` (o
    passo da última fotografia);
  - procura dos agentes: `after_step` (o último candidato) e `after_message` (a última
    mensagem da conversa);
  - pergunta ao analista: `after_step` (o último passo do analista).
- **WebSocket** (não aparece no OpenAPI): `/experiments/{run_id}/stream`,
  `/reinforcement/{run_id}/stream`, `/automl/{search_id}/stream` e
  `/automl/questions/{question_id}/stream` enviam o mesmo que `updates`, sozinhos, de 0,25 em
  0,25 s, em mensagens `{"kind": "update", "data": ...}` (ou `{"kind": "error", "data":
  <ApiError>}`). A ligação fecha quando o trabalho termina; o resultado pede-se então por HTTP.
- Na procura dos agentes, cada `updates` traz também o `working`: o candidato que um agente
  está a medir agora, o que muda nele e o fold em que vai (`fold` de `folds`). É `null` entre
  candidatos e fora das rondas. É o que acende o agente certo no fluxograma. O `thought` é o
  raciocínio do agente de IA até aqui; na pergunta ao analista, o `live` é o que ele está a
  pensar e a escrever. Os dois crescem sem passos novos, e cada bocado novo também vem numa
  mensagem do WebSocket.
- Pedir o resultado antes do fim responde 409 (`training_in_progress`).
- `POST /automl/{search_id}/candidates/{number}/train` treina um candidato à mão: devolve o
  `JobState` de um treino normal, que se acompanha nas rotas de `/experiments`.

## Erros

O corpo de um erro é sempre um `ApiError`: `{code, message, problems}`.

- `message` está em inglês, como o resto do site, e diz o que fazer: mostra-se como está.
- `code` é estável (o nome do erro, em `snake_case`) e serve para o frontend decidir.
- `problems` só vem num pedido inválido, com uma frase por problema.

| HTTP | Quando | Códigos |
|---|---|---|
| 404 | O identificador não existe | `run_not_found`, `dataset_not_found`, `job_not_found`, `search_not_found`, `candidate_not_found`, `run_not_in_memory`, `component_not_found`, `package_not_found` |
| 409 | O trabalho ainda decorre, ou falhou | `training_in_progress`, `training_failed` |
| 422 | Os valores do pedido não servem | `invalid_request`, `hyperparameter`, `invalid_parameter`, `column_not_found`, `data`, `data_source`, `data_validation`, `invalid_seed`, `package` |
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

Algumas coisas não vêm no JSON como campo próprio (as propriedades do Python não vão nele), porque
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
| O ecrã que abre um run do histórico | `tasks[task_type].learns_from_data` do vocabulário: falso é o controlo (um agente de RL, no ecrã do RL); verdadeiro, um modelo (Treinar e avaliar) |
| O valor da métrica principal de uma entrada do histórico | `metrics[primary_metric]`, escrito com `primary_as_percent` |

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
| Dados | `datasets_samples`, `datasets_generators`, `datasets_load_sample`, `datasets_generate`, `datasets_upload`, `datasets_load`, `datasets_loaded`, `datasets_summary`, `datasets_preview` |
| Explorar | `datasets_profile` |
| Preparar | `datasets_suggest_task`, `datasets_task_options`, `datasets_suggest_cleaning`, `datasets_cleaning_operations`, `datasets_prepare` |
| Modelos (a lista e a página de cada modelo) | `catalog_models`, `catalog_model`, `catalog_suggested_model`, `catalog_validate_hyperparameters`, `catalog_demo`, `experiments_start` |
| Treinar e avaliar | `experiments_updates` (e o WebSocket), `experiments_pause`, `experiments_resume`, `experiments_cancel`, `experiments_job`, `experiments_preview`, `experiments_result` (ou `history_open`), `analysis_charts`, `analysis_cross_validation`, `analysis_learning_curve`, `catalog_metrics` |
| Explicar | `history_story` (a história do treino, em capítulos com texto e visual, para qualquer run; ADR 0047), `experiments_result` (ou `history_open`), `analysis_permutation_importance`, `analysis_partial_dependence`, `analysis_test_rows`, `analysis_explain_row` |
| Guardar | `packages_candidates`, `packages_options`, `packages_preview`, `packages_save`, `packages_saved`, `packages_contents`, `packages_load`, `packages_trust`, `packages_delete`, e o ficheiro (`GET /packages/{package_id}/file`, uma ligação normal) |
| Agentes | `automl_options`, `automl_plan`, `automl_start`, `automl_search_plan` (o plano de uma procura, para desenhar os agentes enquanto ela decorre), `automl_updates` (e o WebSocket), `automl_pause`, `automl_resume`, `automl_cancel`, `automl_result`, `automl_train_candidate`; o vencedor é um run como os outros (`experiments_result`, ou `history_open` depois de a API recomeçar). O analista (ADR 0053): `automl_analyst_options` (as perguntas sugeridas e as ferramentas), `automl_questions` (as respondidas), `automl_ask`, `automl_question_updates` (e o WebSocket), `automl_question_job`, `automl_question_result`, `automl_cancel_question`, e `llm_status` para o painel dizer o que falta |
| Métricas | `catalog_metrics` |
| Laboratório | `laboratory_demos` (o que cada demonstração mostra, o seu guia, os valores do seu controlo e o de partida), `laboratory_overfitting`, `laboratory_learning_rate`, `laboratory_knn`, `laboratory_regularization`, `laboratory_threshold`, `laboratory_data_size` |
| Reinforcement learning | `reinforcement_environments`, `reinforcement_agents`, `reinforcement_start`, `reinforcement_updates` (e o WebSocket), `reinforcement_pause`, `reinforcement_resume`, `reinforcement_cancel`, `reinforcement_job`, `reinforcement_result` (ou `history_open_reinforcement`), `history_entry`, `history_save` |
| History | `history_runs`, `history_entry`, `history_open`, `history_open_reinforcement`, `history_save`, `history_rename`, `history_delete`, `history_compare`, `history_reproduce` |
| Todos | `explanations_vocabulary`, `explanations_topics`, `explanations_explain`, `meta_health` |

O Treinar e avaliar é o treino ao vivo e, quando acaba, a avaliação, com uma gaveta por
métrica (a fórmula de `catalog_metrics` e os `terms` do run).

O WebSocket de uma procura manda uma mensagem também quando só muda o trabalho a decorrer (o
candidato que um agente mede, o fold em que vai, a fase): é com isso que o fluxograma mostra
quem trabalha agora, entre um candidato e o seguinte.

**Um run que a API já não tem em memória** (só ficam os últimos, e nenhum depois de ela
recomeçar) responde `run_not_in_memory`. O ecrã Treinar e avaliar, o Explicar e o do RL pedem-no
então ao histórico (`history_open`, `history_open_reinforcement`): um run guardado volta com o
modelo (e com ele os gráficos e o Explicar); um só registado volta com tudo menos isso. Só um
run que também não está no histórico (`run_not_found`) se perdeu. O botão "Guardar experiência"
do RL lê a entrada do histórico (`history_entry`) e guarda com `history_save`; o "Guardar" do
Treinar e avaliar leva ao ecrã Guardar, onde gravar o pacote também guarda o run.

**Os pacotes (o ecrã Guardar, ADR 0039).** Um `.mllab` é um ZIP com a ficha, o manifesto, a
preparação, o run e, se se quiser, o modelo e algumas linhas de exemplo:

- `packages_options` diz o nome sugerido e as partes que se podem deixar de fora, em
  `choices`: cada uma com o título, o que leva, se vai por omissão e, quando o run não a tem, a
  razão. O `key` de cada uma é o campo do pedido que a liga (`include_model`,
  `include_examples`, `include_training`): o formulário desenha-se da lista, sem saber quais são;
- `packages_preview` devolve o ficheiro parte a parte antes de existir, com o mesmo código que o
  grava: o que se vê é o que se guarda. Pede-se outra vez depois de uma pausa curta, a cada
  opção que muda;
- `packages_save` grava na biblioteca, guarda o run no histórico e devolve o `package_id`; o
  ficheiro descarrega-se pela ligação `GET /packages/{package_id}/file` (a única rota que não
  devolve JSON);
- `packages_load` recebe o ficheiro (`multipart/form-data`). Feito neste Orbyte, o run entra
  logo no histórico (`trusted`); de outra instalação, `trusted` vem falso e o aviso (os
  `notices`) mostra-se antes de `packages_trust`, porque abrir o modelo corre o código de quem
  o fez. Um pacote estragado (`intact` falso) só se lê;
- um run ainda a treinar responde `training_in_progress`, e um só registado (sem o modelo)
  `run_not_in_memory`.

**Os controlos do Laboratório** desenham-se do `laboratory_demos`: os valores que cada um
oferece (de onde a taxa de aprendizagem se arrasta até onde explode, os alphas de cada modelo)
são uma escolha didática, e decidem-se no backend. Um valor que a demonstração não pode usar
(um grau fora dos que compara, um alpha que não é positivo, um limiar fora de 0 a 1) responde
422, com a razão. O guia de cada demonstração (`guide`: a ideia, os passos a experimentar e o
que fica) também vem daí, e a interface mostra-o à vista (ADR 0051).

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
