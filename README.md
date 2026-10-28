# ML Lab

Laboratório de machine learning para aprender. Carregas um dataset, escolhes um tipo de
aprendizagem e um algoritmo, e vês o treino passo a passo (snapshots), as métricas de
avaliação e o registo da experiência.

O plano completo está em [docs/architecture.md](docs/architecture.md); as decisões e as suas
razões estão em [docs/adr/](docs/adr/).

## Estado

| Fase | O que entrega | Estado |
|---|---|---|
| 0 — Fundações | Repositório, configuração, logs, registo de componentes, tipos, eventos, seeds, testes de arquitetura, esqueleto da interface | ✅ concluída |
| 1 — Camada de dados | As nove etapas da camada de dados; ecrãs Dados, Explorar e Preparar | ✅ concluída |
| 2 — Primeiro fluxo completo | Contrato de modelo, regressão logística, avaliação de classificação, serviço de experiências; ecrãs Modelo e Avaliar | ✅ concluída |
| 3 — Eventos e snapshots | Contexto de treino, snapshots com a fronteira de decisão, treino em segundo plano, regressões didáticas em NumPy, avaliação de regressão; ecrã Treinar | ✅ concluída |
| 4 — Catálogo clássico | 32 modelos supervisionados e não supervisionados, avaliadores de todas as tarefas, diagnósticos, validação cruzada, explicações das previsões; ecrãs Avaliar (completo), Explicar e Laboratório | ✅ concluída |
| 5 — Memória | Registo de experiências em SQLite, runs guardados com o modelo, comparação e reprodução de runs; ecrã Experiências | ✅ concluída |
| 6 — Deep learning | Adaptador PyTorch, rede neuronal (MLP) de classificação e de regressão, early stopping, gradientes e ativações de cada camada; o ecrã Treinar mostra as camadas da rede | ✅ concluída |
| 7 — API | Os serviços por HTTP com FastAPI (uma rota por método, documentação em `/docs`), treino em segundo plano acompanhado por HTTP ou WebSocket, erros com código e mensagem em português | ✅ concluída |
| 8 — Reinforcement learning | Contratos de ambiente e agente, uma grelha com três mapas, Q-learning (NumPy) e DQN (PyTorch), avaliação da política aprendida, runs no histórico; ecrã Reinforcement learning | ✅ concluída |
| 9 — Agentes (AutoML) | O modo automático: um agente de dados, um agente por família de modelos e um coordenador que procura o melhor modelo com validação cruzada, sem tocar no teste; o vencedor fica no histórico; ecrã Agentes | ✅ concluída |
| 10 — Interface aprimorada | Todos os ecrãs refeitos em React, por cima da API; a interface simples, em Streamlit, saiu depois dela (ADR 0031) | em curso |

## Requisitos

- [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 sozinho se for preciso.
- [Node.js](https://nodejs.org/) 24 ou mais recente (traz o npm), para a interface.
- Git.

## Instalação

```bash
uv sync
```

Este comando cria o ambiente virtual em `.venv/`, instala as versões exatas fixadas em
`uv.lock` (as bibliotecas do pacote e as ferramentas de desenvolvimento) e instala o pacote
`ml_lab` em modo editável: as alterações em `src/` ficam ativas sem reinstalar. A biblioteca
mais pesada é o PyTorch (as redes neuronais, desde a fase 6): no Windows vem a versão só para
CPU, que chega para as redes do laboratório.

Os datasets de exemplo (Iris, Wine, Titanic e California Housing) já vêm no repositório, em
`data/samples/`. Para os voltar a descarregar das origens, com verificação SHA-256:

```bash
uv run python scripts/download_samples.py --force
```

## Abrir a interface

A interface é uma aplicação web (React, em `ui/web`) que fala com o backend pela API. Um só
comando, a partir da raiz do projeto, arranca as duas:

```bash
uv run python scripts/dev.py
```

A interface abre em <http://localhost:5173> e, como a API, só este computador lhe tem acesso.
Recarrega sozinha quando gravas um ficheiro, e `Ctrl+C` pára as duas. Na primeira vez, o
comando instala as dependências da interface (`npm install`, em `ui/web`).

No **Início** há dois caminhos: **Começar à mão** (Explorar, Preparar, Modelo e "Treinar e
avaliar") e **Começar com os agentes** (o ecrã Agentes). Os dois começam nos Dados e acabam
nos mesmos ecrãs: o "Treinar e avaliar", o Explicar e as Experiências.

Para o modo automático: em **Dados**, carrega o Titanic e escolhe "Pedir aos agentes". No
ecrã **Agentes**, o alvo `Survived` já vem escolhido, e ao lado estão o plano do agente de
dados, com as razões de cada escolha, e os agentes que vão trabalhar. Carrega em "Pedir aos
agentes". Enquanto procuram, o fluxograma mostra o agente que mede cada candidato (e o fold em
que vai), e por baixo ficam o melhor até agora, os candidatos medidos e a conversa dos
agentes. No fim:

- o vencedor, a validação cruzada que o escolheu e o teste, usado uma só vez;
- o gráfico da procura (a triagem e a afinação, com o melhor até aí);
- o melhor de cada agente, comparado com a referência (a regressão logística);
- o quadro de todos os candidatos.

"Abrir o vencedor" leva-o para o "Treinar e avaliar", como um run feito à mão, e "Treinar no
modo manual" treina qualquer candidato do quadro, passo a passo.

Um percurso para o modo manual: em **Dados**, carrega o exemplo Titanic e vê o tipo de cada
coluna e porquê; em **Explorar**, vê o perfil das colunas e os avisos (77% de nulos em
`Cabin`, `PassengerId` é um identificador); em **Preparar**, o alvo `Survived` e as limpezas
sugeridas já vêm escolhidos, e ao lado vês a linhagem, a divisão e a projeção 2D; em
**Modelos**, abre a regressão logística, lê a ficha, vê-a aprender num exemplo enquanto
mudas os hiperparâmetros e carrega em "Treinar com estes valores". No **Treinar e avaliar**,
vês o treino ao vivo e, no fim, a fronteira de decisão no plano da projeção, as métricas no
treino e no teste (carrega no nome de uma para ver a fórmula e a conta deste run), a matriz
de confusão e os pesos que o modelo aprendeu. Experimenta baixar o `C` e treinar outra vez.

O treino usa sempre a preparação aberta: noutro dataset, passa pelo **Preparar** antes de
escolher o modelo (com as sugestões, basta abri-lo).

Para ver o treino passo a passo: em **Dados**, gera as "Luas" (separador Sintéticos) e
prepara-as; em **Modelos**, abre a "Regressão logística (gradiente descendente)" e carrega em
"Treinar com estes valores". No **Treinar e avaliar**, carrega no botão por baixo do campo (▶,
ou ↺ para ver outra vez) e vê a fronteira a mover-se época a época, com a loss a descer; o
controlo ao lado percorre as fotografias uma a uma. Experimenta taxas de aprendizagem
diferentes: com 0,05 o treino arrasta-se; com 50 a fronteira salta de um lado para o outro.
Com treinos longos (California Housing com a regressão linear e 20 000 épocas), o ecrã
mostra o progresso ao vivo, com Pausar, Retomar e Parar.

Para avaliar a fundo e explicar, treina um classificador. A avaliação tem a matriz de
confusão, as curvas ROC e de precisão-recall e os pontos onde o modelo erra, no plano. "Correr
a validação cruzada" treina o modelo 5 vezes, e "Desenhar a curva de aprendizagem" mostra
como ele muda com mais dados. "Explicar as previsões" abre o ecrã **Explicar**, com três
perguntas: o que pesa no conjunto (a importância por permutação), como a previsão muda com
uma feature (a dependência parcial) e porque saiu uma previsão concreta.

Para aprender sem alvo, carrega o Iris e, em **Preparar**, mantém o alvo `species` mas
escolhe a tarefa "Clustering". A coluna deixa de entrar no treino e fica como referência.
Treina a "Mistura de gaussianas" e vê as elipses a ajustarem-se, iteração a iteração; na
avaliação, "Grupos e referência" mostra se os grupos encontrados correspondem às espécies.
Para anomalias, gera o exemplo sintético "Anomalias", escolhe a tarefa "Deteção de
anomalias" e treina o Isolation Forest. As pontuações de anomalia mostram as anomalias
verdadeiras de um lado e do outro do limiar.

O ecrã **Laboratório** tem quatro demonstrações, cada uma com um só controlo:
- o grau do polinómio e o overfitting;
- a taxa de aprendizagem do gradiente descendente;
- o k do KNN;
- a força da regularização na Ridge e no Lasso.

O ecrã **Métricas** tem o glossário das 22, cada uma com a fórmula e o que mede.

Cada treino fica registado sozinho no ecrã **Experiências**. Para comparar dois runs:

1. treina o Iris duas vezes, com valores de `C` diferentes;
2. num deles, carrega em "Guardar experiência", no "Treinar e avaliar";
3. no ecrã Experiências, marca os dois runs. Vês as métricas lado a lado, com o melhor
   marcado, só as escolhas que mudaram e as curvas da loss.

Pára a API e arranca-a outra vez: os runs continuam lá, e o guardado volta a abrir com o
modelo (a avaliação e o Explicar funcionam). Na gaveta de cada run, "Reproduzir" treina outra
vez a mesma configuração, e a comparação confirma que as métricas são iguais.

Para ver uma rede neuronal por dentro:

1. em **Dados**, gera as "Luas" com ruído 0,1 e prepara-as;
2. em **Modelos**, abre a "Rede neuronal (MLP)" e treina.

No **Treinar e avaliar**, a fronteira curva-se à volta das luas, o que a regressão logística
não consegue. A secção "As camadas da rede" mostra, passo a passo, a ativação de cada
neurónio, o tamanho dos gradientes de cada camada e os neurónios parados. Duas experiências para fazer
aí:

- a sigmoide com 4 camadas: o gradiente encolhe de camada para camada, e na primeira é
  dezenas de vezes menor do que na saída (o gradiente que desaparece). Com a ReLU, fica
  parecido em todas as camadas;
- a ReLU com uma taxa de aprendizagem de 0,5: mais de metade dos neurónios acabam
  parados, e a exatidão cai.

Para ver um agente aprender por tentativa e erro, abre o ecrã **Reinforcement learning**:

1. escolhe a grelha com o mapa "Penhasco" e o agente "Q-learning";
2. carrega em "Treinar o agente".

A recompensa por episódio sobe à medida que o agente aprende. Na grelha, percorre os
episódios e vê o valor a espalhar-se a partir do objetivo, célula a célula, até à partida.
No fim, compara as métricas: com a política aprendida, chega sempre ao objetivo; a treinar,
caía nas armadilhas em cerca de 15% dos episódios, porque ainda explorava ao acaso. Depois
experimenta:

- a exploração no fim (ε) a 0: a treinar, também deixa de cair, mas o agente deixa de
  procurar caminhos melhores;
- o desconto (γ) a 0,5: o valor da partida fica negativo, porque o objetivo, a 13 passos,
  quase não conta (0,5¹³ ≈ 0,0001);
- o DQN: os mesmos valores Q, mas numa rede neuronal, e muito mais devagar.

## Usar a API

A API (fase 7) põe os serviços a falar HTTP, e é por ela que a interface chega ao backend.
Para a correr sozinha:

```bash
uv run python -m ml_lab.api
```

A documentação interativa fica em <http://127.0.0.1:8000/docs>. Lá dá para:

1. enviar um CSV (`POST /datasets/uploads`);
2. começar um treino (`POST /experiments`, que responde logo com o `run_id`);
3. acompanhar o treino (`GET /experiments/{run_id}/updates`, com os passos e a fronteira de
   cada snapshot);
4. pedir o resultado (`GET /experiments/{run_id}/result`).

O WebSocket `/experiments/{run_id}/stream` envia o mesmo que o passo 3, sozinho. Os agentes
de reinforcement learning têm as mesmas rotas, em `/reinforcement`, e o modo automático
também, em `/automl` (`POST /automl` com o dataset e o alvo; os `updates` trazem a conversa
dos agentes e os candidatos novos). Os pormenores estão em
[src/ml_lab/api/README.md](src/ml_lab/api/README.md).

Para a interface (fase 10), o contrato da API fica guardado em `ui/web/openapi.json`, de
onde se geram os tipos TypeScript, e o guia do contrato está em
[ui/web/README.md](ui/web/README.md): as rotas de cada ecrã, o vocabulário
(`GET /vocabulary`), o acompanhamento do treino e os códigos de erro. Depois de mudar um
serviço, uma rota ou um tipo devolvido, volta a gerar o contrato:

```bash
uv run python scripts/export_openapi.py
```

## Testes e verificações

Uma fase só termina quando estas quatro verificações passam (corre-as antes de cada push):

```bash
uv run ruff format .
uv run ruff check .
uv run mypy
uv run pytest
```

Para um ciclo rápido, sem os testes de integração e da API:

```bash
uv run pytest -m "not integration and not api"
```

A interface tem as suas, dentro de `ui/web`:

```bash
npm run lint
npm run typecheck
npm test
npm run e2e
```

| Pasta | O que verifica | Quando corre |
|---|---|---|
| `tests/unit/` | Uma função ou classe isolada; espelha `src/ml_lab/` | Sempre |
| `tests/contract/` | Todos os componentes registados cumprem o seu contrato: modelos, métricas, fontes de dados, limpezas, divisões, agentes e ambientes, e a fronteira dos serviços | Sempre |
| `tests/architecture/` | Regras de importação entre camadas | Sempre |
| `tests/integration/` | Fluxos de ponta a ponta: o critério de cada fase, nos serviços | Antes de cada push |
| `tests/api/` | Endpoints HTTP e WebSocket (fase 7) | Antes de cada push |
| `ui/web/src/` (`*.test.ts`, `*.test.tsx`) | As peças da interface: os números à portuguesa, as cores, as escolhas do Preparar, os erros da API e os avisos | Sempre (`npm test`) |
| `ui/web/e2e/` | A interface num browser, ecrã a ecrã (Playwright), com a API numa pasta temporária | Antes de cada push (`npm run e2e`) |

Os avisos (`warnings`) contam como erros nos testes: um aviso do pandas ou do scikit-learn
costuma anunciar um problema real, por isso é melhor resolvê-lo logo.

## Configuração

Os valores por omissão chegam para começar. Para mudar alguma coisa, copia `.env.example`
para `.env` e edita-o, ou define variáveis de ambiente (que têm prioridade sobre o `.env`).

| Variável | Para que serve | Omissão |
|---|---|---|
| `ML_LAB_UPLOADS_DIR` | Datasets carregados pelo utilizador | `data/uploads` |
| `ML_LAB_ARTIFACTS_DIR` | Modelos treinados e snapshots | `artifacts` |
| `ML_LAB_DATABASE_PATH` | Ficheiro SQLite do registo de experiências | `artifacts/ml_lab.db` |
| `ML_LAB_LOG_LEVEL` | Nível mínimo dos logs | `INFO` |
| `ML_LAB_LOG_FORMAT` | `text` (terminal) ou `json` (logs estruturados) | `text` |
| `ML_LAB_API_HOST` | Onde a API escuta (`127.0.0.1` é só este computador) | `127.0.0.1` |
| `ML_LAB_API_PORT` | A porta da API | `8000` |
| `ML_LAB_API_CORS_ORIGINS` | As páginas que podem chamar a API a partir do navegador | o servidor do Vite |

## Estrutura

```text
src/ml_lab/          pacote instalável
├── config.py        definições: caminhos, base de dados, logs, API
├── logging.py       logs estruturados
├── core/            núcleo: não importa api/ nem ui/
│   ├── foundation/  tipos, erros, registo de componentes, eventos, seeds
│   ├── data/        camada de dados: as nove etapas (o único sítio com pandas)
│   ├── models/      algoritmos: um ficheiro = um modelo
│   ├── rl/          reinforcement learning: ambientes e agentes
│   ├── automl/      os agentes do modo automático: dados, um por família, coordenador
│   ├── services/    a única porta de entrada do núcleo
│   └── ...          training, evaluation, explainability, tracking...
└── api/             FastAPI: uma rota por método dos serviços (fase 7)
ui/web/              a interface (React + TypeScript, fase 10)
tests/               unit, contract, integration, architecture, api
data/samples/        datasets de exemplo (versionados)
scripts/             utilitários: arrancar a API e a interface, descarregar os exemplos…
docs/                arquitetura, ADRs e fichas didáticas dos algoritmos
```

A árvore completa, com o papel de cada ficheiro, está em
[docs/architecture.md](docs/architecture.md#estrutura-de-pastas). Cada camada tem um
`README.md` curto com as suas regras.

## Convenções

- Nomes de código em inglês; documentação e explicações didáticas em português.
- Tipos em todas as funções públicas, verificados com `mypy --strict`.
- Docstrings no estilo Google, em português.
- Imports sempre absolutos (`from ml_lab.core.foundation.types import TaskType`).
- Ramo `main` sempre a funcionar; trabalho em ramos `feat/...` e `fix/...`.
- Mensagens de commit no formato Conventional Commits (`feat: adiciona k-means didático`).
- Cada decisão que não seja óbvia fica registada numa ADR em `docs/adr/`.
- O backend escreve-se pronto para o frontend: os serviços devolvem dados simples que
  passam a JSON, e as regras de domínio ficam no backend. As regras completas estão em
  [CONTRIBUTING.md](CONTRIBUTING.md).

Para acrescentar um algoritmo, uma métrica, uma fonte de dados ou um tipo de tarefa, segue
[Manutenção: como adicionar componentes](docs/architecture.md#manutenção-como-adicionar-componentes).
