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
| 5 — Memória | Base de dados, registo de experiências, artefactos, comparação e reprodução de runs; ecrã Experiências | próxima |
| 6 a 9 | Ver o [roadmap](docs/architecture.md#roadmap-por-fases) | por fazer |

## Requisitos

- [uv](https://docs.astral.sh/uv/), que instala o Python 3.12 sozinho se for preciso.
- Git.

## Instalação

```bash
uv sync
```

Este comando cria o ambiente virtual em `.venv/`, instala as versões exatas fixadas em
`uv.lock` (as bibliotecas do pacote, as ferramentas de desenvolvimento, o Streamlit e o
Plotly) e instala o pacote `ml_lab` em modo editável: as alterações em `src/` ficam ativas sem
reinstalar.

Os datasets de exemplo (Iris, Wine, Titanic e California Housing) já vêm no repositório, em
`data/samples/`. Para os voltar a descarregar das origens, com verificação SHA-256:

```bash
uv run python scripts/download_samples.py --force
```

## Abrir a interface simples

```bash
uv run streamlit run ui/simple/app.py
```

Corre-o a partir da raiz do projeto. A app abre em <http://localhost:8501> e só este
computador lhe tem acesso (ver `.streamlit/config.toml`). Na primeira vez, o Streamlit pode
pedir um email no terminal: basta carregar em Enter.

Um percurso para começar: em **Dados**, carrega o exemplo Titanic; em **Explorar**, vê o
perfil das colunas e os avisos (77% de nulos em `Cabin`, `PassengerId` é um identificador);
em **Preparar**, escolhe o alvo `Survived`, aceita as limpezas sugeridas e vê a linhagem, a
divisão e a projeção 2D; em **Modelo**, lê a ficha da regressão logística, ajusta os
hiperparâmetros e carrega em "Treinar"; em **Treinar**, vês a fronteira de decisão no plano
da projeção; em **Avaliar**, vês as métricas (cada uma com a sua explicação), a matriz de
confusão, o treino contra o teste e os pesos que o modelo aprendeu. Experimenta baixar o `C`
e treinar outra vez.

Para ver o treino passo a passo: em **Dados**, gera as "Luas" (separador Sintéticos); em
**Modelo**, escolhe a "Regressão logística (gradiente descendente)" e carrega em "Treinar".
No ecrã **Treinar**, carrega em "▶ Reproduzir" e vê a fronteira a mover-se época a época, com
a loss a descer. Experimenta taxas de aprendizagem diferentes: com 0,05 o treino arrasta-se;
com 50 a fronteira salta de um lado para o outro. Com treinos longos (California Housing com
a regressão linear e 20 000 épocas), o ecrã mostra o progresso ao vivo, com Pausar e Parar.

Para avaliar a fundo e explicar, treina um classificador e abre o ecrã **Avaliar**. Tens a
matriz de confusão, as curvas ROC e de precisão-recall e os pontos onde o modelo erra, no
plano. No separador "Validação cruzada", o botão "Calcular" treina o modelo 5 vezes e
desenha a curva de aprendizagem. O ecrã **Explicar** mostra três coisas: que features o
modelo usa (importância por permutação), como a previsão muda com cada uma (dependência
parcial) e porque saiu uma previsão concreta.

Para aprender sem alvo, carrega o Iris e, em **Preparar**, mantém o alvo `species` mas
escolhe a tarefa "Clustering". A coluna deixa de entrar no treino e fica como referência.
Treina a "Mistura de gaussianas" e vê as elipses a ajustarem-se, iteração a iteração; no
**Avaliar**, o separador "Referência" mostra se os grupos encontrados correspondem às
espécies. Para anomalias, gera o exemplo sintético "Anomalias", escolhe a tarefa "Deteção
de anomalias" e treina o Isolation Forest. O separador "Pontuações" mostra as anomalias
verdadeiras de um lado e do outro do limiar.

O ecrã **Laboratório** tem quatro demonstrações, cada uma com um só controlo:
- o grau do polinómio e o overfitting;
- a taxa de aprendizagem do gradiente descendente;
- o k do KNN;
- a força da regularização na Ridge e no Lasso.

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

| Pasta | O que verifica | Quando corre |
|---|---|---|
| `tests/unit/` | Uma função ou classe isolada; espelha `src/ml_lab/` | Sempre |
| `tests/contract/` | Todos os componentes registados cumprem o seu contrato: modelos, métricas, fontes de dados, limpezas, divisões | Sempre |
| `tests/architecture/` | Regras de importação entre camadas | Sempre |
| `tests/integration/` | Fluxos de ponta a ponta: o critério de cada fase, nos serviços e na interface | Antes de cada push |
| `tests/api/` | Endpoints HTTP e WebSocket (fase 7) | Antes de cada push |

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

## Estrutura

```text
src/ml_lab/          pacote instalável
├── config.py        definições: caminhos, base de dados, logs
├── logging.py       logs estruturados
├── core/            núcleo: não importa api/ nem ui/
│   ├── foundation/  tipos, erros, registo de componentes, eventos, seeds
│   ├── data/        camada de dados: as nove etapas (o único sítio com pandas)
│   ├── models/      algoritmos: um ficheiro = um modelo
│   ├── services/    a única porta de entrada do núcleo
│   └── ...          training, evaluation, explainability, tracking...
└── api/             FastAPI (fase 7)
ui/simple/           interface simples (Streamlit)
ui/web/              interface aprimorada (React, fase 9)
tests/               unit, contract, integration, architecture, api
data/samples/        datasets de exemplo (versionados)
scripts/             utilitários, como descarregar os exemplos
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

Para acrescentar um algoritmo, uma métrica, uma fonte de dados ou um tipo de tarefa, segue
[Manutenção: como adicionar componentes](docs/architecture.md#manutenção-como-adicionar-componentes).
