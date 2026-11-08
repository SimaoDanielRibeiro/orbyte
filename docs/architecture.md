# ML Lab — Plano de arquitetura do backend

22 de outubro de 2026 · Simão

> Versão em Markdown do plano original (`ML Lab - Plano de arquitetura do backend.pdf`, na raiz
> do repositório). Este ficheiro é o que se mantém atualizado: quando a arquitetura mudar,
> muda aqui e regista a decisão numa ADR em [adr/](adr/). As notas da implementação estão no
> fim, em [Notas da implementação](#notas-da-implementação).

## Visão geral

O ML Lab é um backend em Python onde se carrega um dataset, se escolhe um tipo de
aprendizagem e um algoritmo, e se obtém o treino passo a passo (snapshots), as métricas de
avaliação e o registo da experiência. A interface visual fica para a última fase; até lá,
tudo é usado por código, testes e uma API.

### Objetivos

1. **Aprender ML:** cada algoritmo expõe o que acontece durante o treino, não só o resultado
   final.
2. **Aprender as tecnologias:** cada biblioteca tem um papel único e claro (secção
   [Tecnologias](#tecnologias)).
3. **Manutenção fácil:** adicionar um algoritmo, uma métrica ou um tipo de tarefa = criar um
   ficheiro e registá-lo, sem alterar código existente.

### Princípios de arquitetura

- **Núcleo independente da interface.** O pacote `core` não importa FastAPI, nem bibliotecas
  de gráficos, nem nada de UI. A API consome o núcleo, e a interface consome a API; nunca o
  contrário.
- **Contratos antes de implementações.** Modelos, avaliadores, transformações e fontes de
  dados obedecem a interfaces abstratas. O resto do sistema só conhece a interface.
- **Registo (plugins).** Cada componente regista-se sozinho num registo central com um
  decorador. Nada de `if algoritmo == "knn"` espalhados pelo código.
- **Configuração declarativa e validada.** Hiperparâmetros e experiências são descritos por
  esquemas Pydantic: validados à entrada, serializáveis para JSON e reutilizáveis pela UI
  para gerar formulários.
- **Observabilidade por eventos.** O treino emite eventos (início, passo, snapshot, fim).
  Quem precisa deles subscreve-os: registo, API em tempo real, testes.
- **Reprodutibilidade.** Toda a aleatoriedade passa por uma seed explícita; cada experiência
  guarda dataset, versão do código, configuração e seed.
- **Dependências apontam para dentro.** Camadas externas dependem das internas, nunca o
  inverso (ver [Arquitetura em camadas](#arquitetura-em-camadas)).

### Decisões assumidas (a confirmar)

- Stack enxuta: 13 tecnologias no backend e na interface simples; FastAPI e React só nas
  fases 7 e 10 (ver [Tecnologias](#tecnologias)). Depois da fase 10, a interface simples saiu
  e levou o Streamlit e o Plotly (ADR 0031).
- Python 3.12+, scikit-learn para todo o ML clássico, PyTorch para deep learning.
- pandas apenas na camada de dados (`core/data/`); o resto do sistema trabalha com NumPy.
- Implementações didáticas em NumPy puro para os algoritmos simples, ao lado das versões do
  scikit-learn.
- Registo de experiências em SQLite com o módulo `sqlite3` da biblioteca padrão, atrás de um
  contrato trocável.
- Interface em duas versões com as mesmas funcionalidades: simples (Streamlit) a par de cada
  fase; aprimorada e personalizada (React) na fase 10. Quando a web ficou com tudo, a simples
  saiu (ADR 0031).
- Reinforcement learning na fase 8, com ambiente próprio em grelha, sem bibliotecas de RL.
- Agentes de AutoML na fase 9: um modo automático ao lado do manual, com agentes
  deterministas por cima dos serviços, sem modelos de linguagem (ADR 0025).
- Execução local, num só utilizador; sem autenticação nem filas distribuídas.

## Mapa do projeto

Cada experiência atravessa sete regiões, da entrada dos dados até à interface. Como num
cérebro, há dois circuitos de retorno: a avaliação devolve o que aprendeu à preparação, e a
interface lança novas experiências.

Sete regiões, 28 etapas, 2 ciclos:

| Região | Etapas (por ordem) | Onde vive |
|---|---|---|
| 1 · Entrada de dados | **Carregar** (CSV, exemplos, sintéticos) → **Validar** (ficheiro usável? colunas, tamanho) → **Perfilar** (tipos, nulos, distribuições) → **Limpar** (duplicados, tipos, nulos, outliers) | `core/data` · pandas |
| 2 · Preparação | **Definir tarefa** (escolher o alvo, sugerir `TaskType`) → **Dividir** (treino, validação e teste, com seed) → **Transformar** (imputar, codificar, escalar: só treino) → **Converter** (DataFrame → NumPy, fim do pandas) | `core/data` · pandas → NumPy |
| 3 · Modelação | **Catálogo** (modelos filtrados pelo `TaskType`) → **Configurar** (hiperparâmetros validados) → **Treinar** (o Trainer chama fit passo a passo) → **Prever** (previsões em treino e teste) | `core/models` · `core/training` |
| 4 · Observação do treino | **Eventos** (o modelo anuncia cada passo) → **Snapshots** (estado do modelo a cada N passos) → **Progresso** (loss por época, logs estruturados) → **Controlo** (parar, pausar, early stopping) | EventBus · `core/training` |
| 5 · Avaliação | **Métricas** (por tipo de tarefa) → **Diagnósticos** (overfitting, desequilíbrio) → **Robustez** (validação cruzada, learning curves) → **Explicar** (importância de features) | `core/evaluation` · explainability |
| 6 · Memória | **Registar** (run, métricas, configuração) → **Guardar** (modelo treinado e snapshots, a pedido) → **Comparar** (2 a 5 runs lado a lado) → **Reproduzir** (mesma configuração e mesma seed) | `core/tracking` · SQLite |
| 7 · Apresentação | **Serviços** (única porta do núcleo) → **Gráficos** (fronteiras, curvas, projeções 2D) → **API** (HTTP e WebSocket, uma rota por método) → **Interface** (React, todos os ecrãs) | services · visualization_data · api · ui/web |

- **Ciclo de aprendizagem:** a avaliação volta à preparação para ajustar dados ou
  hiperparâmetros.
- **Nova experiência:** a partir da interface, volta-se a carregar ou escolher dados.

O que cada região faz, em resumo:

| Região | Pergunta a que responde | Entrega à região seguinte |
|---|---|---|
| 1 · Entrada de dados | Que dados tenho e em que estado estão? | Tabela validada, perfilada e limpa |
| 2 · Preparação | O que quero prever e como separo os dados? | `Dataset` em NumPy, dividido e transformado |
| 3 · Modelação | Que algoritmo uso e com que hiperparâmetros? | Modelo treinado e previsões |
| 4 · Observação do treino | O que aconteceu durante o treino? | Sequência de eventos e snapshots |
| 5 · Avaliação | O modelo é bom? Porquê? Onde falha? | Relatório com métricas, diagnósticos e explicações |
| 6 · Memória | Como guardo, comparo e repito? | Run guardado e comparável |
| 7 · Apresentação | Como mostro tudo isto ao utilizador? | Ecrãs da interface |

As regiões 3 e 4 acontecem ao mesmo tempo: enquanto o modelo treina, a observação recebe
cada passo. A região 4 alimenta também a 6 (os snapshots ficam guardados) e a 7 (o treino
aparece em tempo real).

## Tecnologias

A stack final tem **11 tecnologias** para o backend e mais **10** que só entram nas fases 7
e 10 (a API e a interface em React). Todos os tipos de aprendizagem do plano ficam cobertos.
Cada tecnologia tem um único papel; a coluna "Fase" diz quando a começas a usar. Até à fase
10 havia mais duas, as da interface simples (o Streamlit e o Plotly), que saíram com ela
(ADR 0031).

### Ferramentas de desenvolvimento

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| Python 3.12+ | Linguagem principal do ecossistema de ML | Todo o backend | 0 |
| uv | Gestor de pacotes e ambientes virtuais, muito rápido | Instalar dependências, criar o ambiente, fixar versões (`uv.lock`), correr comandos | 0 |
| ruff | Linter e formatador numa só ferramenta | Estilo uniforme e deteção de erros comuns antes de correr o código | 0 |
| mypy | Verificador estático de tipos | Garantir que cada implementação respeita os contratos | 0 |
| pytest | Framework de testes | Testes unitários, de contrato, de integração e de arquitetura | 0 |

### Dados e machine learning

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| NumPy | Arrays e álgebra linear rápida | Formato interno dos dados depois da camada de dados; implementações didáticas | 1 |
| pandas | Tabelas (DataFrames) com colunas de tipos diferentes | Toda a camada de dados (ver [Processamento de dados com pandas](#processamento-de-dados-com-pandas)) | 1 |
| scikit-learn | A biblioteca padrão de ML clássico | Todos os algoritmos clássicos (incluindo gradient boosting), pipelines, divisão, validação cruzada, métricas, importância por permutação e dependência parcial. Traz o joblib, usado para guardar modelos | 2 |
| PyTorch | Framework de deep learning com loop de treino explícito | Redes neuronais (MLP, CNN) e o DQN do reinforcement learning | 6 |

### Backend

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| Pydantic v2 | Validação de dados com tipos Python | Hiperparâmetros, configurações de experiência, fichas técnicas; o JSON Schema de onde a interface gera os formulários | 2 |
| SQLite (módulo `sqlite3`) | Base de dados num único ficheiro, incluída no Python | Registo de experiências, runs e métricas | 5 |

Configuração e logs usam só a biblioteca padrão do Python (um módulo de configuração e o
módulo `logging`), sem dependências extra.

### Interface simples (saiu depois da fase 10)

Serviu da fase 0 à 10 e saiu quando a interface web ficou com tudo (ADR 0031). Fica aqui o que
usava, para a história; a última versão com ela está no commit `dcdcc5e`.

| Tecnologia | O que é | Para que serviu no projeto | Fase |
|---|---|---|---|
| Streamlit | Transforma scripts Python em aplicações web, sem HTML nem JavaScript | Todos os ecrãs da interface simples, construídos a par do backend | 0 |
| Plotly | Gráficos interativos (zoom, hover, animação) | Gráficos da interface simples (a aprimorada desenha os seus com o D3) | 1 |

### API e interface aprimorada (fases 7 e 10)

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| FastAPI | Framework web assíncrona baseada em Pydantic, com WebSockets incluídos | Expor os serviços por HTTP e transmitir o treino em tempo real; documentação automática em `/docs` | 7 |
| Uvicorn | Servidor que corre a aplicação FastAPI | Correr a API localmente | 7 |
| React | Biblioteca para interfaces feitas de componentes | Interface aprimorada e personalizada | 10 |
| TypeScript | JavaScript com tipos | Evitar erros; tipos gerados a partir do contrato da API (`openapi.json`) | 10 |
| Vite | Ferramenta de desenvolvimento e build para front-end | Servidor local rápido, o proxy para a API e o build final | 10 |
| Tailwind CSS | CSS por classes utilitárias | Sistema de design próprio: os tokens de cor, tipo e raio (ADR 0032) | 10 |
| shadcn/ui (com o Radix) | Componentes acessíveis cujo código fica no projeto | A barra lateral, as gavetas, as dicas, os botões | 10 |
| Motion | Animações para React | Transições que respondem a uma ação e o fluxograma dos agentes | 10 |
| TanStack Router e Query | Rotas tipadas e cache dos pedidos | Um ficheiro por ecrã; os dados da API pedidos uma vez e guardados | 10 |
| D3.js | Visualização de baixo nível | Gráficos feitos à medida: fronteiras a mover-se, redes neuronais, centróides | 10 |

As peças à volta (o gerador do cliente, o Zustand, o KaTeX, o ESLint, o Vitest e o
Playwright) e as razões de cada escolha estão na ADR 0027.

A fase 9 (os agentes) não traz tecnologias novas: usa o scikit-learn, o PyTorch e a validação
cruzada que já existem.

### Extensões futuras (fora da stack inicial)

Ficaram de fora para manter a stack pequena. A arquitetura já tem o sítio de cada uma, por
isso podem entrar mais tarde sem refazer nada.

| Tecnologia | O que acrescentaria | Onde entraria |
|---|---|---|
| XGBoost / LightGBM | Boosting usado na indústria, para comparar com o do scikit-learn | Novos modelos em `models/ensemble/` |
| SHAP | Explicação de previsões individuais | `explainability/` |
| SQLAlchemy + Alembic | ORM e migrações de base de dados | Nova implementação do registo de experiências |
| MLflow | Registo de experiências pronto, com interface própria | Nova implementação do registo de experiências |
| pandera | Validação de DataFrames por esquema | `data/validation/` |
| UMAP | Redução de dimensionalidade não linear | `models/unsupervised/dimensionality/` |
| Gymnasium / Stable-Baselines3 | Ambientes padrão de RL e algoritmos prontos (PPO) | `rl/` |
| pre-commit / GitHub Actions | Verificações automáticas antes de cada commit e a cada push | Raiz do repositório |
| Polars | Alternativa mais rápida ao pandas | Só `core/data/` |

O TensorFlow/Keras não entra: o PyTorch cobre o deep learning e manter dois frameworks não
acrescenta aprendizagem.

## Tipos de aprendizagem e algoritmos

O sistema organiza-se por **tipo de tarefa** (`TaskType`). O tipo de tarefa decide que
algoritmos aparecem, que métricas se calculam e que snapshots fazem sentido. "Didático" =
implementação própria em NumPy/PyTorch, feita para aprender; "sklearn" = envolve a
implementação oficial.

### Catálogo inicial

| Paradigma | Tipo de tarefa | Algoritmos | Implementação | Snapshots durante o treino |
|---|---|---|---|---|
| Supervisionada | Classificação | Regressão logística | Didático + sklearn | Pesos e fronteira por época, loss |
| Supervisionada | Classificação | Perceptron | Didático | Fronteira após cada correção |
| Supervisionada | Classificação | KNN | Didático + sklearn | Vizinhos de cada ponto (sem treino iterativo) |
| Supervisionada | Classificação | Naive Bayes (Gaussiano) | sklearn | Médias e variâncias por classe |
| Supervisionada | Classificação | Árvore de decisão | sklearn | Árvore crescendo por profundidade |
| Supervisionada | Classificação | Random Forest | sklearn | Erro OOB à medida que se juntam árvores |
| Supervisionada | Classificação | SVM (linear e RBF) | sklearn | Vetores de suporte, margem |
| Supervisionada | Classificação / Regressão | Gradient Boosting e AdaBoost | sklearn | Previsão e loss árvore a árvore |
| Supervisionada | Regressão | Regressão linear (gradiente descendente) | Didático | Reta/plano e loss por época, trajetória na superfície de erro |
| Supervisionada | Regressão | Regressão linear (mínimos quadrados), Ridge, Lasso | sklearn | Coeficientes vs. força da regularização |
| Supervisionada | Regressão | Regressão polinomial | sklearn (pipeline) | Curva por grau, demonstração de overfitting |
| Supervisionada | Regressão | Árvore / Random Forest de regressão, SVR | sklearn | Função em degraus a refinar-se |
| Não supervisionada | Clustering | K-means | Didático + sklearn | Centróides e atribuições por iteração |
| Não supervisionada | Clustering | DBSCAN | sklearn | Pontos núcleo, fronteira e ruído |
| Não supervisionada | Clustering | Clustering hierárquico (aglomerativo) | sklearn | Dendrograma, fusões passo a passo |
| Não supervisionada | Clustering | Gaussian Mixture | sklearn | Elipses por iteração do EM |
| Não supervisionada | Redução de dimensionalidade | PCA | Didático + sklearn | Componentes, variância explicada |
| Não supervisionada | Redução de dimensionalidade | t-SNE | sklearn | Embedding ao longo das iterações |
| Não supervisionada | Deteção de anomalias | Isolation Forest, One-Class SVM, LOF | sklearn | Pontuação de anomalia por ponto |
| Deep learning | Classificação / Regressão | MLP (rede densa) | PyTorch | Loss no treino e na validação, fronteira, gradientes, ativações e neurónios parados por época |
| Deep learning (extensão) | Classificação de imagem | CNN pequena | PyTorch | Filtros e mapas de ativação |
| Reinforcement learning (fase 8) | Controlo | Q-learning tabular, DQN | Didático (NumPy) / PyTorch | Recompensa por episódio, valor e política de cada célula, tabela Q, caminho aprendido |

### Pontos de extensão previstos

A arquitetura não precisa de mudanças para receber, mais tarde:

- **Semi-supervisionada** (Label Spreading, Self-Training): novo `TaskType` + avaliador.
- **Séries temporais** (ARIMA, previsão com janelas): novo `TaskType`, divisor de dados
  temporal e métricas próprias.
- **Sistemas de recomendação** (fatorização de matrizes): novo `TaskType` + avaliador de
  ranking.
- **NLP básico** (TF-IDF + classificador): nova transformação em `preprocessing`,
  reaproveita os classificadores existentes.

### Regras de deteção automática do tipo de tarefa

1. Sem coluna-alvo → só aparecem clustering, redução de dimensionalidade e deteção de
   anomalias.
2. Alvo categórico (texto, booleano, ou inteiro com poucos valores distintos) →
   classificação (binária ou multiclasse).
3. Alvo numérico contínuo → regressão.
4. O utilizador pode sempre forçar o tipo; a deteção é só uma sugestão. Com um alvo
   categórico, pode escolher também uma tarefa sem alvo: a coluna fica como referência,
   só para comparar no fim (ADR 0015).

## Processamento de dados com pandas

Toda a manipulação de dados vive numa única camada isolada, `core/data/`, e é o único sítio
do projeto onde o pandas é usado. Os dados entram como ficheiro, passam por nove etapas e
saem como arrays NumPy prontos para os modelos.

### O que é o pandas

O pandas é a biblioteca de Python para trabalhar com **dados em tabela**, como uma folha de
Excel controlada por código. Está construído por cima do NumPy: cada coluna é, por dentro,
um array NumPy, o que o torna rápido. É a ferramenta padrão para ler, explorar e limpar
dados antes do machine learning.

Os três objetos que vais usar sempre:

| Objeto | O que é | Analogia |
|---|---|---|
| DataFrame | A tabela completa: linhas e colunas com nome | Uma folha de cálculo |
| Series | Uma única coluna, com um tipo de dados | Uma coluna da folha |
| Index | Os rótulos das linhas (por omissão 0, 1, 2…) | Os números das linhas no Excel |

Cada coluna tem um **tipo de dados (dtype)**: inteiro, decimal, texto, categoria, booleano
ou data. Grande parte do trabalho de preparação é garantir que cada coluna tem o tipo certo;
por exemplo, um preço lido como texto não pode ser usado num modelo.

### pandas vs. NumPy: porque usamos os dois

| | pandas | NumPy |
|---|---|---|
| Forma dos dados | Tabela com colunas de tipos diferentes (texto, números, datas) | Matriz com um só tipo numérico |
| Nomes | Colunas e linhas com nome | Apenas posições (linha 3, coluna 5) |
| Valores em falta | Suportados e fáceis de detetar | Exigem cuidado manual |
| Ponto forte | Ler, explorar, limpar, transformar | Cálculo numérico rápido |
| Onde entra no projeto | Etapas 1 a 8 da camada de dados | Modelos, treino, avaliação |

Os modelos de ML precisam de matrizes numéricas; os dados reais chegam como tabelas mistas.
O pandas faz a ponte, e a última etapa da camada converte tudo para NumPy.

### Porque isolar o pandas numa só camada

- **Clareza:** sabes sempre onde procurar qualquer operação sobre dados.
- **Fronteira simples:** o resto do sistema recebe sempre o mesmo objeto (`Dataset` com
  arrays NumPy e metadados), venha o dado de onde vier.
- **Substituição:** trocar o pandas pelo Polars (mais rápido, mais recente) afeta só esta
  pasta.
- **Testes:** cada etapa testa-se sozinha com tabelas pequenas feitas à mão.
- **Verificação automática:** um teste de arquitetura falha se algum ficheiro fora de
  `core/data/` importar o pandas.

### As nove etapas da camada de dados

Cada etapa tem uma pasta própria (ver [Estrutura de pastas](#estrutura-de-pastas)) e uma só
responsabilidade.

| # | Etapa | Pasta | O que faz | Altera os dados? |
|---|---|---|---|---|
| 1 | Entrada | `io/` | Lê CSV/Excel, datasets de exemplo ou gera dados sintéticos para um DataFrame | Não |
| 2 | Validação | `validation/` | Verifica se o ficheiro é utilizável: vazio, colunas repetidas, tamanho, codificação | Não |
| 3 | Perfil | `profiling/` | Descreve cada coluna: tipo inferido, nulos, valores únicos, estatísticas, distribuição, correlações, desequilíbrio de classes | Não |
| 4 | Limpeza | `cleaning/` | Remove duplicados, corrige tipos (texto → número/data), remove colunas inúteis (IDs), trata outliers se pedido | Sim, com registo |
| 5 | Tarefa | `task/` | O utilizador escolhe a coluna-alvo; o sistema sugere o tipo de tarefa | Não |
| 6 | Divisão | `splitting/` | Separa treino, validação e teste (aleatória, estratificada ou temporal), com seed | Separa |
| 7 | Transformação | `transforms/` | Imputação de nulos, codificação de categorias, escala, engenharia de features; aprende os parâmetros só no treino | Sim |
| 8 | Projeção | `projection/` | Prepara uma versão 2D (duas colunas ou PCA) para os gráficos | Cópia à parte |
| 9 | Saída | `boundary.py` | Converte para arrays NumPy, guarda os nomes das features e entrega o `Dataset` | Converte |

### Regras da camada de dados

1. **Limpeza antes da divisão, transformação depois.** A limpeza não aprende nada com os
   dados (remover duplicados é igual em qualquer amostra). A transformação aprende
   parâmetros (a média para imputar, o desvio para escalar) e por isso só pode ver o treino.
   Se aprendesse com o teste, o modelo estaria a "espreitar" as respostas: é o *data
   leakage*, e o projeto vai mostrá-lo como exemplo didático.
2. **Nunca alterar o original.** Cada etapa devolve um DataFrame novo; o ficheiro carregado
   fica intacto.
3. **Tudo fica registado (linhagem).** Cada operação aplicada ("removidas 12 linhas
   duplicadas", "coluna idade convertida para número") é guardada em `lineage.py`. A
   interface mostra este histórico com o antes/depois, e a experiência pode ser reproduzida
   exatamente.
4. **Explicar em vez de esconder.** Cada etapa produz avisos didáticos: "30% de nulos na
   coluna cabine", "classe positiva só tem 5% das linhas".
5. **Datasets grandes:** o perfil e os gráficos usam uma amostra; o treino usa tudo.
6. **Nomes de colunas preservados até ao fim:** o scikit-learn pode devolver DataFrames nas
   transformações, por isso os nomes das features chegam à explicabilidade e à interface.

### Conceitos de pandas que vais aprender neste projeto

| Conceito | Para que serve | Etapa |
|---|---|---|
| Leitura de ficheiros (`read_csv`, `read_excel`) | Carregar dados para DataFrame, com separador e codificação certos | 1 |
| Tipos (`dtypes`, `astype`, `to_numeric`, `to_datetime`) | Ver e corrigir o tipo de cada coluna | 3, 4 |
| Resumos (`describe`, `value_counts`, `nunique`) | Estatísticas e contagens por coluna | 3 |
| Valores em falta (`isna`, `fillna`, `dropna`) | Detetar e tratar nulos | 3, 4, 7 |
| Correlação (`corr`) | Medir relações entre colunas numéricas | 3 |
| Duplicados (`duplicated`, `drop_duplicates`) | Encontrar e remover linhas repetidas | 4 |
| Seleção (`loc`, `iloc`, `select_dtypes`) | Escolher linhas e colunas por nome, posição ou tipo | Todas |
| Agrupamento (`groupby`) | Estatísticas por classe (ex.: média de idade por sobrevivente) | 3 |
| Categorias (`category`) | Poupar memória e marcar colunas categóricas | 4 |
| Copy-on-Write | Perceber quando uma operação cria uma cópia ou altera o original | Todas |
| Conversão (`to_numpy`) | Passar da tabela para a matriz numérica | 9 |

## Arquitetura em camadas

O sistema tem cinco camadas e as dependências só apontam para baixo: a interface usa os
serviços (diretamente na versão simples, através da API na aprimorada), os serviços usam o
núcleo, e tudo assenta na fundação. O núcleo não depende de nada exterior: a API e a
interface dependem dele.

```text
┌──────────────────────────────────────────────────────────────────────┐
│ Interface (simples, depois aprimorada)                               │
│ A simples chama os serviços diretamente; a aprimorada usa a API      │
└──────────────────────────────────┬───────────────────────────────────┘
┌──────────────────────────────────▼───────────────────────────────────┐
│ API (FastAPI)                                                        │
│ Rotas HTTP + WebSocket que transmite os eventos de treino            │
└──────────────────────────────────┬───────────────────────────────────┘
┌──────────────────────────────────▼───────────────────────────────────┐
│ Serviços (casos de uso)                                              │
│ datasets · catálogo · experiências: única porta de entrada do núcleo │
└──────────────────────────────────┬───────────────────────────────────┘
┌──────────────────────────────────▼───────────────────────────────────┐
│ Núcleo de domínio (core)                    não importa api nem ui   │
│  Dados → Modelos → Treino → Avaliação                                │
│  Dados para gráficos · Explicabilidade · Registo de experiências     │
└──────────────────────────────────┬───────────────────────────────────┘
┌──────────────────────────────────▼───────────────────────────────────┐
│ Fundação                                                             │
│ TaskType · registo genérico · eventos · esquemas Pydantic · config · │
│ seeds                                                                │
└──────────────────────────────────────────────────────────────────────┘
```

Dentro do núcleo, as setas mostram o caminho de uma execução: dados → modelo → treino →
avaliação. As três peças de baixo (dados para gráficos, explicabilidade, registo de
experiências) são usadas a pedido.

### Fluxo de uma experiência

1. A interface (ou um teste, ou um notebook) pede ao serviço de experiências para correr
   uma experiência, com uma configuração validada: dataset, preparação, modelo,
   hiperparâmetros e seed.
2. A camada de dados entrega um `Dataset` já limpo, dividido e transformado
   (transformações aprendidas só no treino), em NumPy.
3. O serviço vai buscar o modelo ao registo e cria-o com os hiperparâmetros validados.
4. O Trainer abre um run no registo de experiências, anuncia o início e manda o modelo
   treinar.
5. Durante o treino, o modelo anuncia cada passo. O EventBus entrega cada evento a quem
   subscreveu: gravador de snapshots, registo de experiências e interface.
6. No fim, o avaliador do tipo de tarefa gera um relatório para treino e para teste; a
   diferença entre os dois é o alerta de overfitting.
7. Relatório e modelo ficam guardados; o serviço devolve o resultado.
8. A interface pede os dados dos gráficos (fronteira de decisão, curvas, projeção 2D) e
   desenha-os.

Na fase 2, o `ExperimentService` fazia os passos 1 a 7 sem o Trainer nem o registo de
experiências: anunciava o run no `EventBus`, treinava, avaliava e devolvia o resultado.
Desde a fase 3, os passos 4 e 5 são do Trainer e do gravador de snapshots, e o passo 8 usa
a animação montada a partir dos snapshots. Desde a fase 5, o registo de experiências
subscreve o fim de cada run (`RunFinished`) e regista-o sozinho. No passo 7, o modelo e
os snapshots só se guardam quando se pede (ADR 0020).

### Regra de dependências

| Camada | Pode importar | Nunca importa |
|---|---|---|
| Interface (web) | API (por HTTP e WebSocket) | Nada do Python diretamente |
| API | `services`, `schemas` | Modelos, dados, treino ou base de dados diretamente |
| Serviços | Todo o `core` | `api`, `ui` |
| Núcleo | Fundação, bibliotecas de ML; pandas só em `core/data/`; SQLite e joblib só em `core/tracking/`; PyTorch só nas redes (`models/neural/`, o seu adaptador e o DQN em `rl/agents/dqn.py`) | `api`, `ui`, `services` |
| Fundação | Biblioteca padrão, Pydantic | Qualquer outra camada |

Estas regras são verificadas em `tests/architecture/test_dependency_rules.py`.

## Estrutura de pastas

O projeto usa o layout `src/` (o código instalável fica em `src/ml_lab/`), que obriga os
testes a importar o pacote tal como um utilizador o faria e evita importações acidentais.
Cada pasta do núcleo corresponde a uma responsabilidade; cada ficheiro de algoritmo é
independente.

```text
ml-lab/
├── pyproject.toml            # dependências, grupos opcionais e config de ruff/mypy/pytest
├── uv.lock                   # versões exatas fixadas (gerado pelo uv)
├── README.md                 # como instalar, correr e contribuir
├── .env.example              # variáveis de ambiente de exemplo
├── .pre-commit-config.yaml   # verificações antes de cada commit (extensão futura)
├── .github/workflows/ci.yml  # testes e lint a cada push (extensão futura)
│
├── docs/
│   ├── architecture.md       # este plano, mantido atualizado
│   ├── adr/                  # decisões de arquitetura (uma por ficheiro)
│   └── algorithms/           # uma ficha didática por algoritmo
├── scripts/                  # utilitários: recriar a base de dados, descarregar exemplos
├── data/
│   ├── samples/              # datasets de exemplo pequenos (versionados)
│   └── uploads/              # datasets carregados (ignorado pelo Git)
├── artifacts/                # modelos treinados e snapshots (ignorado pelo Git)
├── notebooks/                # explorações; nunca importados pelo código
│
├── src/ml_lab/
│   ├── config.py             # definições: caminhos, base de dados, logs, API
│   ├── logging.py            # configuração de logs estruturados
│   ├── formatting.py         # números à portuguesa (19,9%, 20 640)
│   ├── process.py            # pede ao Windows para não travar o processo (ADR 0023)
│   │
│   ├── core/                 # NÚCLEO: não importa api/ nem ui/
│   │   ├── foundation/       # FUNDAÇÃO: peças usadas por todas as camadas
│   │   │   ├── types.py      # TaskType, Paradigm, tipos partilhados
│   │   │   ├── errors.py     # exceções do domínio
│   │   │   ├── registry.py   # registo genérico de componentes (plugins)
│   │   │   ├── events.py     # eventos de treino e EventBus
│   │   │   ├── notices.py    # avisos didáticos (info, aviso, erro)
│   │   │   └── random.py     # gestão central de seeds
│   │   │
│   │   ├── data/             # CAMADA DE DADOS: o único sítio onde o pandas é usado
│   │   │   ├── README.md     # regras desta camada
│   │   │   ├── dataset.py    # Dataset: o objeto que sai da camada (NumPy + metadados)
│   │   │   ├── schema.py     # descrição de colunas: tipo, papel (feature/alvo/ignorar)
│   │   │   ├── lineage.py    # histórico de todas as operações aplicadas aos dados
│   │   │   ├── preparation.py  # etapas 4 a 9 pela ordem certa: configuração → Dataset
│   │   │   ├── explanations.py # textos do painel "Porquê?" de cada etapa
│   │   │   ├── io/           # 1. ENTRADA
│   │   │   │   ├── base.py               # contrato DataSource
│   │   │   │   ├── file_source.py        # CSV e Excel carregados pelo utilizador
│   │   │   │   ├── builtin_source.py     # Iris, Wine, Titanic, California Housing
│   │   │   │   └── synthetic_source.py   # moons, blobs, circles, regressão, anomalias
│   │   │   ├── validation/   # 2. VALIDAÇÃO
│   │   │   │   ├── checks.py             # vazio, colunas repetidas, tamanho, codificação
│   │   │   │   └── schemas.py            # regras esperadas por coluna
│   │   │   ├── profiling/    # 3. PERFIL (só descreve, nunca altera)
│   │   │   │   ├── type_inference.py     # numérica, categórica, booleana, data, texto, ID
│   │   │   │   ├── column_profile.py     # nulos, únicos, estatísticas por coluna
│   │   │   │   └── dataset_profile.py    # resumo global, correlações, desequilíbrio
│   │   │   ├── cleaning/     # 4. LIMPEZA (cada operação fica registada)
│   │   │   │   ├── base.py               # registo CLEANING, receita e aplicação
│   │   │   │   ├── duplicates.py
│   │   │   │   ├── type_coercion.py      # texto → número, data, booleano
│   │   │   │   ├── columns.py            # remover IDs e colunas inúteis
│   │   │   │   ├── missing.py            # remover linhas com nulos (sobretudo no alvo)
│   │   │   │   ├── outliers.py           # detetar e, se pedido, tratar extremos
│   │   │   │   └── suggestions.py        # receita sugerida a partir do perfil
│   │   │   ├── task/         # 5. TAREFA
│   │   │   │   └── task_detection.py     # sugere o TaskType a partir do alvo
│   │   │   ├── splitting/    # 6. DIVISÃO
│   │   │   │   ├── base.py               # SplitConfig, registo SPLITTERS
│   │   │   │   ├── random_split.py
│   │   │   │   ├── stratified_split.py
│   │   │   │   └── time_split.py
│   │   │   ├── transforms/   # 7. TRANSFORMAÇÃO (aprende só com o treino)
│   │   │   │   ├── base.py               # contrato Transform
│   │   │   │   ├── imputation.py
│   │   │   │   ├── encoding.py
│   │   │   │   ├── scaling.py
│   │   │   │   ├── feature_engineering.py  # polinomiais, datas, intervalos
│   │   │   │   └── pipeline_builder.py   # monta o pipeline a partir da configuração
│   │   │   ├── projection/   # 8. PROJEÇÃO 2D para gráficos
│   │   │   │   ├── pca.py                # PCA didática em NumPy
│   │   │   │   └── projection_2d.py      # duas colunas ou PCA
│   │   │   └── boundary.py   # 9. SAÍDA: DataFrame → NumPy (fim do pandas)
│   │   │
│   │   ├── models/           # algoritmos (um ficheiro = um modelo)
│   │   │   ├── base.py       # contrato LabModel + ficha técnica do modelo + registo MODELS
│   │   │   ├── distances.py  # distâncias aos blocos (KNN, DBSCAN, hierárquico)
│   │   │   ├── adapters/     # ligação ao scikit-learn e ao PyTorch
│   │   │   │   ├── sklearn_adapter.py    # SklearnModel: o contrato, uma vez, para o scikit-learn
│   │   │   │   ├── growth.py             # modelos do scikit-learn que crescem passo a passo
│   │   │   │   └── torch_adapter.py      # TorchModel: o ciclo de treino das redes, early
│   │   │   │                             # stopping e a fotografia das camadas
│   │   │   ├── supervised/
│   │   │   │   ├── gradient_descent.py  # o que os modelos por gradiente descendente partilham
│   │   │   │   ├── trees.py      # o que as árvores de classificação e de regressão partilham
│   │   │   │   ├── forests.py    # o mesmo, para as Random Forests
│   │   │   │   ├── kernels.py    # o mesmo, para o SVM e o SVR (o kernel)
│   │   │   │   ├── classification/  # logística (sklearn e gradiente), perceptron, KNN (sklearn
│   │   │   │   │                    # e NumPy), Naive Bayes, árvore, Random Forest, SVM
│   │   │   │   └── regression/      # linear (OLS e gradiente), Ridge, Lasso, polinomial,
│   │   │   │                        # árvore, Random Forest, SVR
│   │   │   ├── ensemble/     # gradient boosting e AdaBoost, de classificação e de regressão
│   │   │   ├── unsupervised/
│   │   │   │   ├── clustering/      # k-means (NumPy e sklearn), DBSCAN, hierárquico, GMM
│   │   │   │   ├── dimensionality/  # PCA (NumPy e sklearn), t-SNE
│   │   │   │   └── anomaly/         # detectors.py: Isolation Forest, One-Class SVM, LOF
│   │   │   └── neural/       # redes PyTorch: mlp.py, a rede densa (classificação e regressão)
│   │   │
│   │   ├── training/         # Trainer, snapshots, treino em segundo plano
│   │   │   ├── context.py    # TrainingContext (com a sonda) e a amostragem dos snapshots
│   │   │   ├── control.py    # pedidos de pausa e de paragem, entre fios de execução
│   │   │   ├── trainer.py    # Trainer: liga o modelo ao EventBus e fotografa o fim
│   │   │   ├── recorder.py   # gravador de snapshots (subscritor do EventBus)
│   │   │   ├── jobs.py       # TrainingJob: o treino em segundo plano
│   │   │   └── explanations.py  # textos do painel "Porquê?" do ecrã Treinar
│   │   ├── evaluation/       # métricas, avaliadores por tarefa, diagnósticos, validação
│   │   │   ├── metrics/      # base.py (registo METRICS) e um ficheiro por tipo de tarefa
│   │   │   ├── base.py       # contrato Evaluator, registo EVALUATORS, métrica principal
│   │   │   ├── classification.py  # avaliador de classificação
│   │   │   ├── regression.py      # avaliador de regressão (comparado com prever a média)
│   │   │   ├── clustering.py      # avaliador de clustering (e a comparação com a referência)
│   │   │   ├── dimensionality.py  # avaliador de redução de dimensionalidade
│   │   │   ├── anomaly.py         # avaliador de deteção de anomalias
│   │   │   ├── report.py     # EvaluationReport, MetricResult, ConfusionMatrix
│   │   │   ├── explanations.py    # o que mede cada métrica e como se lê
│   │   │   ├── topics.py          # temas do painel "Porquê?" de Avaliar, Explicar e Laboratório
│   │   │   ├── diagnostics.py     # comparação treino vs. teste (overfitting)
│   │   │   └── validation.py      # validação cruzada e curva de aprendizagem
│   │   ├── explainability/   # explicações que servem a qualquer modelo
│   │   │   ├── permutation.py          # importância por permutação
│   │   │   └── partial_dependence.py   # dependência parcial e curvas ICE
│   │   ├── visualization_data/  # prepara DADOS para gráficos (sem desenhar)
│   │   │   ├── grid.py       # a grelha da sonda, no plano da projeção 2D
│   │   │   ├── animation.py  # TrainingAnimation: o campo do modelo em cada snapshot
│   │   │   ├── embedding.py  # o mapa da redução de dimensionalidade
│   │   │   ├── evaluation.py # os gráficos do ecrã Avaliar (ROC, resíduos, silhueta…)
│   │   │   ├── learned.py    # o que o modelo aprendeu: pesos, regras, centros, com nomes
│   │   │   ├── network.py    # NetworkTrace: as camadas de uma rede em cada snapshot
│   │   │   └── policy.py     # PolicyTrace: o valor e a política de um agente, na grelha
│   │   ├── tracking/         # MEMÓRIA: o registo de experiências e os ficheiros dos runs
│   │   │   ├── store.py      # contrato ExperimentStore, RunEntry e o registo em memória
│   │   │   ├── sqlite_store.py  # o registo em SQLite, com migrações (user_version)
│   │   │   ├── artifacts.py  # o modelo (joblib) e o treino passo a passo (JSON) guardados
│   │   │   └── explanations.py  # textos do painel "Porquê?" do ecrã Experiências
│   │   ├── schemas/          # esquemas Pydantic partilhados
│   │   │   ├── hyperparameters.py  # base dos hiperparâmetros (formulários, validação, x-search)
│   │   │   ├── experiment.py # ExperimentConfig: dados, preparação, modelo e seed
│   │   │   ├── reinforcement.py  # ReinforcementConfig: ambiente, agente e seed
│   │   │   ├── automl.py     # AutoMLRequest: o pedido aos agentes e o tamanho da procura
│   │   │   ├── serialization.py  # a fronteira em JSON: to_json, json_schema, @internal
│   │   │   └── validation.py # erros do Pydantic explicados em português
│   │   ├── services/         # casos de uso: a única porta de entrada do núcleo
│   │   │   ├── datasets.py   # carregar, perfilar e preparar dados (ecrãs 1, 3 e 4)
│   │   │   ├── automl.py     # os agentes: o plano, a procura e o vencedor (ecrã 2)
│   │   │   ├── catalog.py    # catálogo de modelos (ecrã 5)
│   │   │   ├── experiments.py  # treinar (também em segundo plano), avaliar, anunciar (5 a 7)
│   │   │   ├── analysis.py   # depois do treino: gráficos, validação cruzada, explicações (7 e 8)
│   │   │   ├── history.py    # o histórico: registar, guardar, abrir, comparar, reproduzir (9)
│   │   │   ├── comparison.py # a comparação de runs lado a lado
│   │   │   ├── laboratory.py # as demonstrações do ecrã Laboratório (10)
│   │   │   ├── reinforcement.py  # treinar agentes num ambiente e avaliá-los (11)
│   │   │   ├── explanations.py # o painel "Porquê?": os temas de todas as camadas
│   │   │   ├── vocabulary.py # o nome a mostrar de cada valor e o que muda com cada tarefa
│   │   │   └── lab.py        # todos os serviços de uma sessão, ligados entre si
│   │   ├── rl/               # REINFORCEMENT LEARNING: aprender num ambiente, sem dataset
│   │   │   ├── environments/ # contrato Environment, registo ENVIRONMENTS e a grelha
│   │   │   ├── agents/       # contrato Agent, o ciclo de episódios (ε-greedy), Q-learning
│   │   │   │                 # (NumPy) e DQN (PyTorch), registo AGENTS
│   │   │   ├── evaluation.py # as métricas: o treino (a explorar) e a política aprendida
│   │   │   └── explanations.py  # textos do painel "Porquê?" do ecrã Reinforcement learning
│   │   ├── automl/           # AGENTES DO MODO AUTOMÁTICO: o melhor modelo para um pedido
│   │   │   ├── data_agent.py # o agente de dados: do perfil à preparação, com as razões
│   │   │   ├── model_agent.py  # um agente por família de modelos, saído do registo MODELS
│   │   │   ├── search_space.py # o espaço de procura de cada modelo, lido do x-search
│   │   │   ├── objective.py  # a métrica a otimizar e o seu sentido
│   │   │   ├── coordinator.py  # triagem e afinação, com validação cruzada só no treino
│   │   │   ├── search_log.py # a conversa dos agentes e os candidatos, entre fios
│   │   │   └── explanations.py  # textos do painel "Porquê?" do ecrã Agentes
│   │   └── llm/              # MODELOS DE LINGUAGEM dos agentes de IA (fase 13, ADR 0042)
│   │       ├── base.py       # a conversa (mensagens, pedaços, ferramentas) e LanguageModel
│   │       ├── ollama.py     # o Ollama por HTTP: o único pedido à rede do núcleo
│   │       └── scripted.py   # respostas escritas à mão, para os testes sem o Ollama
│   │
│   └── api/                  # API: os serviços por HTTP e WebSocket (fase 7, ADR 0022)
│       ├── app.py            # create_app: um Lab por processo, CORS, erros
│       ├── routing.py        # ServiceRouter: respostas pelo to_json, ids serviço_método
│       ├── errors.py         # o código HTTP de cada categoria de erro (ApiError)
│       ├── stream.py         # o WebSocket: o mesmo updates dos serviços, de 0,25 em 0,25 s
│       ├── openapi.py        # o OpenAPI com as respostas completas (ADR 0026)
│       ├── requests.py       # corpos dos pedidos que juntam vários valores
│       └── routes/           # uma rota por método público de cada serviço
│
├── ui/
│   └── web/                  # A INTERFACE (React + TypeScript), fase 10
│       ├── README.md         # como correr, o guia do contrato e como mexer na interface
│       ├── openapi.json      # o contrato, gerado pelo Python (o cliente gera-se daqui)
│       ├── e2e/              # testes de ponta a ponta (Playwright), com a API numa pasta temporária
│       └── src/
│           ├── api/          # o cliente gerado, os erros, a cache e o vocabulário
│           ├── components/   # ui/ (shadcn), shell/ (a moldura), charts/ (D3), forms/, notices/…
│           ├── features/     # um ecrã por pasta: dados, treino, agentes, experiências…
│           ├── lib/          # format.ts (os números à portuguesa, como no Python), as cores
│           ├── routes/       # um ficheiro por ecrã (TanStack Router)
│           ├── state/        # o pouco estado do cliente (Zustand)
│           └── styles/       # globals.css: os tokens do sistema visual (ADR 0032)
│
└── tests/
    ├── unit/                 # espelha src/ml_lab/
    ├── contract/             # aplicados a TODOS os componentes registados
    ├── integration/          # fluxo completo dataset → métricas → base de dados
    ├── architecture/         # regras de importação entre camadas
    └── api/                  # testes dos endpoints
```

### Regras de organização

- `core/data/` é o único sítio onde o pandas é importado; tudo o que sai desta pasta é
  NumPy dentro de um `Dataset`. Um teste de arquitetura verifica isto.
- As pastas da camada de dados estão numeradas pela ordem das etapas (1 a 9), para leres o
  processamento de cima para baixo.
- `core/` nunca importa `api/` nem `ui/`.
- `foundation/` não importa nenhuma outra pasta do projeto.
- `visualization_data/` só calcula números (grelhas, curvas, coordenadas). Desenhar é
  trabalho da interface, que escolhe a forma e as cores; assim, os números servem qualquer
  interface (serviram duas até à fase 10).
- `services/` é a única porta de entrada do núcleo: a API chama-a, e a interface fala com a
  API.
- Os métodos públicos dos serviços devolvem só dados simples, que passam a JSON, e as
  regras de domínio ficam no backend: a interface só desenha (ADR 0019, `CONTRIBUTING.md`). O
  teste de contrato da fronteira (`tests/contract/test_service_boundary.py`) verifica-o.
- Um algoritmo novo = um ficheiro novo na pasta do seu paradigma. Nenhum ficheiro existente
  muda.

## Contratos centrais

Um contrato é uma promessa: "qualquer componente deste tipo recebe X e devolve Y". O resto
do sistema só conhece o contrato, nunca a implementação concreta. É isto que permite
adicionar algoritmos sem mexer no código existente.

### Visão geral dos contratos

| Contrato | Responsabilidade | Recebe | Devolve | Pasta |
|---|---|---|---|---|
| TaskType | Lista fechada dos tipos de tarefa (classificação binária, multiclasse, regressão, clustering, redução de dimensionalidade, anomalias) | — | — | `foundation/` |
| Registry | Guardar e encontrar componentes pelo nome; filtrar por tipo de tarefa | Componentes que se registam no arranque | O componente pedido, ou um erro claro | `foundation/` |
| DataSource | Obter dados de uma origem | Referência (ficheiro, nome de exemplo, parâmetros do gerador) | DataFrame + metadados | `data/io/` |
| Transform | Uma transformação de dados que aprende com o treino | Dados de treino (para aprender), depois quaisquer dados | Dados transformados | `data/transforms/` |
| Dataset | O objeto que sai da camada de dados | — | Arrays NumPy, nomes das features, esquema, linhagem | `data/` |
| LabModel | Um algoritmo de ML | Hiperparâmetros validados, seed, dados, contexto de treino | Previsões (classe, valor, grupo ou anomalia); conforme a tarefa, probabilidades, coordenadas novas (`transform`) ou pontuações de anomalia; estado interno | `models/` |
| TrainingContext | Canal entre o modelo e o exterior durante o treino | Pedidos do modelo para emitir um passo | Eventos publicados; ordem de parar | `training/` |
| EventBus | Distribuir eventos de treino a quem os subscreve | Eventos | Entrega a cada subscritor | `foundation/` |
| Evaluator | Avaliar um modelo para um tipo de tarefa | Modelo, dados, partição (treino/teste) | Relatório com métricas, artefactos e avisos | `evaluation/` |
| Metric | Uma métrica isolada | O que declara precisar (`needs`): valores reais e previstos, probabilidades, features, coordenadas ou pontuações | Um número + se "maior é melhor" (nenhum sentido, nas que só descrevem) | `evaluation/metrics/` |
| ExperimentStore | Guardar e consultar experiências | Configuração, eventos, relatórios, modelos | Runs guardados, comparações | `tracking/` |
| Environment | Um ambiente de reinforcement learning, pequeno e discreto | Um estado e uma ação (e o acaso) | O estado seguinte, a recompensa, se o episódio acabou; como se desenha | `rl/environments/` |
| Agent | Um agente que aprende a agir num ambiente | Hiperparâmetros validados, seed, o ambiente, contexto de treino | A política (a ação de maior Q), a tabela Q, fotografias por episódio | `rl/agents/` |

### Ficha técnica de cada modelo

Cada modelo descreve-se a si próprio. É esta ficha que permite à interface montar o
catálogo, filtrar por tarefa e gerar formulários sem código específico por algoritmo.

| Campo | Exemplo (KNN) | Para que serve |
|---|---|---|
| Identificador | `knn` | Nome único no registo |
| Nome a mostrar | K-Nearest Neighbors | Título na interface |
| Tipos de tarefa | Classificação binária e multiclasse | Filtrar o catálogo |
| Família | Baseado em instâncias | Agrupar no catálogo |
| Implementação | Didática ou scikit-learn | Mostrar a origem; comparar as duas |
| Iterativo | Não | Saber se emite snapshots passo a passo |
| Probabilidades | Sim | Permite curva ROC |
| Hiperparâmetros | número de vizinhos (1 a 100, omissão 5); ponderação (uniforme ou por distância) | Validar e gerar sliders e menus |
| Ficha didática | `docs/algorithms/knn.md` | Texto explicativo na interface |
| Limite de passos | — (não treina por passos) | Nos iterativos, o hiperparâmetro que dá a barra de progresso (`max_epochs`) |
| Prevê linhas novas | Sim | O t-SNE não: só coloca no mapa os pontos do treino |

### Regras de comportamento

1. **O modelo nunca sabe quem o está a observar.** Durante o treino, só comunica através do
   contexto de treino: anuncia cada passo (época, loss, estado) e pergunta se deve parar.
2. **Capacidades opcionais são declaradas, não adivinhadas.** Se um modelo não dá
   probabilidades, a ficha diz isso e a interface esconde a curva ROC.
3. **Adaptadores evitam repetição.** Um adaptador para o scikit-learn e outro para o
   PyTorch implementam o contrato uma vez; cada modelo concreto só indica o estimador, a
   ficha e os hiperparâmetros.
4. **Métricas são funções puras.** Não guardam estado; o mesmo input dá sempre o mesmo
   resultado, o que as torna triviais de testar.
5. **Um avaliador por tipo de tarefa.** Recolhe automaticamente todas as métricas
   registadas para esse tipo; uma métrica nova aparece sozinha nos relatórios.
6. **O registo de experiências é trocável.** Uma versão em memória para testes, uma com
   SQLite para uso real, e no futuro uma com MLflow, sem mudar quem as usa.
7. **Os snapshots são amostrados.** Guardar o estado a cada passo pode gerar gigabytes; o
   registo guarda a cada N passos, configurável.

### O que vai dentro de cada evento de treino

| Evento | Classe (`kind`) | Quando acontece | Conteúdo |
|---|---|---|---|
| Início do run | `RunStarted` (`run_started`) | Antes do primeiro passo | Configuração, tamanho dos dados |
| Passo | `StepCompleted` (`step`) | Cada época ou iteração | Número do passo, loss, métricas parciais |
| Snapshot | `SnapshotTaken` (`snapshot`) | A cada N passos | Estado do modelo: pesos, centróides, árvore, fronteira |
| Avaliação | `EvaluationCompleted` (`evaluation`) | Depois do treino | Relatório de treino e de teste |
| Fim do run | `RunFinished` (`run_finished`) | No final | Duração, resultado, caminho do modelo guardado |
| Falha | `RunFailed` (`run_failed`) | Se ocorrer um erro | Mensagem explicativa |

## Manutenção: como adicionar componentes

Adicionar qualquer componente segue sempre o mesmo padrão: criar um ficheiro, implementar a
interface, registar com o decorador, escrever a ficha didática. Os testes de contrato
verificam o novo componente automaticamente.

### Adicionar um algoritmo

1. Criar o ficheiro na pasta do paradigma, ex.:
   `models/supervised/classification/extra_trees.py`.
2. Definir a classe de hiperparâmetros (Pydantic) e a classe do modelo com `info`
   preenchido. Nos hiperparâmetros que vale a pena afinar, dizer aos agentes o que
   experimentar, com `"x-search"` (ver `schemas/hyperparameters.py`): sem isso, os agentes da
   fase 9 experimentam o modelo, mas só com os valores por omissão.
3. Decorar com `@MODELS.register("extra_trees")`.
4. Escrever `docs/algorithms/extra_trees.md` (intuição, hiperparâmetros, quando usar).
5. Correr `uv run pytest tests/contract` — o modelo é testado sem escrever testes novos,
   incluindo o espaço de procura. O modelo entra sozinho no agente da sua família.

### Adicionar uma métrica

1. Escrever uma função pura em `evaluation/metrics/<tipo>.py`.
2. Decorar com `@metric("balanced_accuracy", task_types={...}, higher_is_better=True)`.
3. Juntar a explicação em `evaluation/explanations.py`: o que mede, como se lê, a fórmula
   em LaTeX e o significado de cada símbolo.
4. Se a métrica for uma conta com quantidades que se contam (os verdadeiros positivos, a
   soma dos erros), dar também os termos da fórmula: `@metric(..., breakdown=...)`, com os
   símbolos escritos como na fórmula (ADR 0028).

A métrica aparece em todos os relatórios dos tipos de tarefa indicados, e no catálogo das
métricas (`CatalogService.metrics`).

### Adicionar uma fonte de dados

Implementar `DataSource.load() -> LoadedTable` (o DataFrame e os seus metadados) e
`DataSource.spec()` num novo ficheiro em `data/io/`, e registar com
`@DATA_SOURCES.register("nome")`. Ex.: uma fonte que lê de uma URL, ou de uma base de dados.
O `Dataset` com arrays NumPy só existe no fim da camada de dados (ADR 0009).

### Adicionar um agente ou um ambiente de reinforcement learning

- **Um agente** (por exemplo, o SARSA): um ficheiro em `rl/agents/` com uma subclasse de
  `EpisodicAgent` que diz como aprende com um passo (`_learn`) e onde guarda os Q (`_q`,
  `_q_table`), decorada com `@AGENTS.register("sarsa")`, e a ficha em `docs/algorithms/`.
- **Um ambiente**: um ficheiro em `rl/environments/` com uma subclasse de `Environment` (o
  `step`, o `reset` e o `layout` para se desenhar), decorada com
  `@ENVIRONMENTS.register("nome")`.

Em ambos os casos, `uv run pytest tests/contract/test_rl.py` testa-o sem escrever testes
novos: aprende a grelha simples, repete com a mesma seed, pára a pedido, o objetivo
alcança-se (ADR 0024).

### Adicionar um tipo de tarefa (o caso maior)

1. Novo valor em `TaskType`.
2. Regra em `task_detection.py` (se puder ser detetado).
3. Estratégia de divisão em `splitting.py`, se for diferente (séries temporais não podem ser
   baralhadas).
4. Um `Evaluator` novo + as suas métricas.
5. Funções em `visualization_data/` para os gráficos específicos.
6. Modelos que declaram o novo tipo em `task_types`.

Nenhum código existente muda, exceto o enum e, se necessário, o detetor.

### Testes de contrato (a rede de segurança)

Em `tests/contract/`, testes parametrizados percorrem o registo e aplicam a TODOS os
componentes:

- Cada modelo treina num dataset mínimo de cada `TaskType` que declara, sem erros.
- `predict` devolve a forma certa; `predict_proba` soma 1 quando existe.
- Mesma seed → mesmo resultado (reprodutibilidade).
- Modelos com `iterative=True` emitem pelo menos um evento `step`.
- Hiperparâmetros por omissão são válidos e o JSON Schema é gerado.
- Cada modelo tem ficha didática no caminho declarado.
- Cada métrica devolve um valor finito em dados de exemplo.

### Decisões de arquitetura (ADR)

Cada decisão importante fica num ficheiro curto em `docs/adr/` (ex.:
`0001-pytorch-em-vez-de-tensorflow.md`), com contexto, decisão e consequências. Daqui a
meses vais saber porque é que algo foi feito assim.

## Qualidade e convenções

O objetivo é que qualquer alteração seja verificada automaticamente antes de entrar no
repositório: formatação, tipos, testes e regras de arquitetura.

### Testes

| Nível | Pasta | O que verifica | Quando corre |
|---|---|---|---|
| Unitários | `tests/unit/` | Uma função ou classe isolada (ex.: perfil de dados, uma métrica com valores calculados à mão) | Sempre |
| Contrato | `tests/contract/` | Todos os componentes registados cumprem o seu contrato | Sempre |
| Arquitetura | `tests/architecture/` | `core` não importa `api` nem `ui`; pandas só em `core/data/`; `foundation` não importa nada do projeto | Sempre |
| Integração | `tests/integration/` | Fluxo completo dataset → preparação → treino → avaliação → base de dados | Antes de cada push |
| API | `tests/api/` | Endpoints HTTP e WebSocket (a partir da fase 7) | Antes de cada push |

As implementações didáticas têm um teste extra: o resultado tem de ficar próximo do da
versão oficial do scikit-learn no mesmo dataset (ex.: coeficientes da regressão linear
iguais até 1e-4). É assim que sabes que a tua implementação está certa.

### Convenções de código

- Tipos em todas as funções públicas; `mypy --strict` no pacote `core`.
- Docstrings no estilo Google, em português, nas classes e funções públicas.
- Nomes de código em inglês (padrão do ecossistema); documentação e explicações didáticas
  em português.
- Sem estado global mutável, exceto os registos (preenchidos só no arranque).
- Erros do domínio com exceções próprias em `core/foundation/errors.py`, com mensagens que
  explicam o que fazer.
- Arrays internos sempre NumPy `float64`; conversão para tensores PyTorch só dentro de
  `TorchAdapter`.

### Git

- Ramo `main` sempre a funcionar; trabalho em ramos `feat/...`, `fix/...`.
- Mensagens de commit no formato Conventional Commits (`feat: adiciona k-means didático`).
- Antes de cada push: `ruff check`, `ruff format`, `mypy` e `pytest`. Automatizar isto
  (pre-commit, GitHub Actions) fica como extensão.

### Documentação

- `README.md`: instalar, correr testes, correr a API.
- `docs/architecture.md`: este plano, atualizado quando a arquitetura mudar.
- `docs/algorithms/<algoritmo>.md`: intuição, matemática essencial, hiperparâmetros,
  quando usar, armadilhas.
- `docs/adr/`: decisões e as suas razões.
- A API gera documentação interativa automaticamente em `/docs` (Swagger), a partir dos
  esquemas Pydantic.

## Interface

A interface foi feita em duas versões com **as mesmas funcionalidades**: primeiro uma simples
(Streamlit), que cresceu a par do backend em cada fase; no fim, uma aprimorada e bem
desenhada (React), que mudou a apresentação mas não o que se pode fazer. Assim validou-se
tudo cedo, e o redesenho final não obrigou a repensar o produto.

Quando a web ficou com tudo, a simples saiu (ADR 0031). A interface do ML Lab é a web, em
`ui/web`, e fala com o backend só pela API.

### Ecrãs

Os ecrãs seguem a ordem natural de uma experiência, da esquerda para a direita numa barra de
navegação. Depois de carregar os dados há dois caminhos (fase 9):

- **o automático**: no ecrã Agentes, pede-se o alvo e a métrica, e os agentes preparam os
  dados, experimentam os modelos e entregam o melhor, com as razões de cada escolha;
- **o manual**: Explorar, Preparar, Modelo e Treinar, decidindo tudo à mão.

Os dois acabam nos mesmos ecrãs (Avaliar, Explicar, Experiências): o vencedor dos agentes é
um run como os outros, e qualquer candidato que eles experimentaram se pode treinar à mão.

| # | Ecrã | O que o utilizador faz | O que vê | Fase do backend |
|---|---|---|---|---|
| 1 | Dados | Carrega um CSV/Excel, escolhe um dataset de exemplo ou gera um sintético (com controlos de ruído e número de pontos); escolhe o modo automático ou o manual | Lista de datasets, pré-visualização das primeiras linhas, erros de validação | 1 |
| 2 | Agentes | Escolhe o alvo, a métrica a otimizar, os agentes e o orçamento, e lança a procura | O que o agente de dados decidiu e porquê, a conversa dos agentes, o quadro dos candidatos com a validação cruzada, o vencedor e o teste dele | 9 |
| 3 | Explorar | Navega pelas colunas | Perfil de cada coluna, histogramas, mapa de correlações, mapa de nulos, desequilíbrio de classes, avisos | 1 |
| 4 | Preparar | Escolhe a coluna-alvo, confirma o tipo de tarefa, aplica limpezas e transformações, define a divisão | Antes/depois de cada operação, histórico de passos (linhagem), tamanho de treino/teste | 1, 2 |
| 5 | Modelo | Escolhe um algoritmo do catálogo filtrado pela tarefa e ajusta hiperparâmetros | Cartões por família, ficha didática, formulário gerado da ficha técnica | 2, 4 |
| 6 | Treinar | Inicia, pausa, cancela; percorre o treino passo a passo | Barra de progresso, loss em tempo real, animação dos snapshots com controlos de reprodução, fronteira de decisão / centróides / árvore | 3 |
| 7 | Avaliar | Lê os resultados | Cartões de métricas com explicação, matriz de confusão, ROC e PR, resíduos, silhouette, comparação treino vs. teste, diagnósticos | 2, 4 |
| 8 | Explicar | Escolhe uma previsão ou o modelo inteiro | Importância de features, gráficos de dependência parcial | 4 |
| 9 | Experiências | Filtra o histórico, escolhe de 2 a 5 runs, abre, guarda ou reproduz um run | Tabela de runs, comparação lado a lado de métricas, escolhas diferentes e curvas da loss | 5 |
| 10 | Laboratório | Mexe num controlo e vê o efeito imediato | Demos de conceitos: overfitting (complexidade do modelo), taxa de aprendizagem, k do KNN, regularização | 4 |
| 11 | Reinforcement learning | Escolhe ambiente e agente, inicia o treino | Recompensa por episódio, tabela Q, política aprendida | 8 |
| 12 | Guardar | Escolhe o run, o nome e as partes opcionais do ficheiro; guarda-o e descarrega-o; carrega um modelo já treinado | A pré-visualização do ficheiro (cada parte, o tamanho e o conteúdo), os avisos sobre o que vai dentro, a biblioteca dos modelos guardados e carregados | 11 |
| 13 | Prever | Escolhe um pacote e preenche uma linha (ou carrega um CSV) | A previsão, as probabilidades, o contributo de cada feature, a curva do "e se…" | 12 |

Na interface web, o Treinar e o Avaliar são um só ecrã, "Treinar e avaliar" (o treino ao vivo
e, quando acaba, a avaliação), e há mais dois: o Início, com os dois caminhos, e as Métricas,
o glossário das 22. A barra lateral mostra os passos do caminho escolhido (ver as notas da
[fase 10](#fase-10)).

### Elementos comuns a todos os ecrãs

- Barra lateral com o estado da experiência atual: dataset, tarefa, modelo, último
  resultado.
- Painel "Porquê?" com a explicação didática do que está no ecrã.
- Avisos destacados quando algo merece atenção (fuga de dados, desequilíbrio, overfitting).
- Botão para guardar a experiência em qualquer momento.

### Simples vs. aprimorada

As duas, como foram planeadas (a simples saiu depois da fase 10, ADR 0031):

| Aspeto | Interface simples | Interface aprimorada |
|---|---|---|
| Tecnologia | Streamlit + Plotly (só Python) | React + TypeScript + Vite + Tailwind CSS + shadcn/ui + Motion + D3 |
| Ligação ao backend | Chama a camada de serviços diretamente, no mesmo processo Python | Através da API (HTTP para pedidos, WebSocket para o treino) |
| Tempo real | Atualização periódica do ecrã de treino | Snapshots chegam em fluxo contínuo |
| Animações | Slider para percorrer snapshots | Animações suaves, transições entre estados |
| Design | Tema padrão do Streamlit com pequenos ajustes | Sistema de design próprio: cores, tipografia, componentes, modo escuro |
| Objetivo | Validar funcionalidades e aprender depressa | Experiência polida para usar e mostrar |
| Quando | Cresce a cada fase, a partir da fase 1 | Fase 10, com a API estável |

### Regras da interface

Nasceram para as duas versões partilharem tudo, e continuam com a web sozinha (as regras
completas do backend estão no `CONTRIBUTING.md`):

1. Nenhuma lógica de ML ou de dados vive na interface; a interface só pede e mostra.
2. Os gráficos são desenhados a partir dos dados preparados por `visualization_data/`: a
   interface recebe os números prontos e só escolhe como os desenhar.
3. Os formulários de hiperparâmetros são gerados a partir das fichas técnicas dos modelos
   (o JSON Schema de cada uma).
4. Os textos didáticos vêm do backend (fichas, explicações de métricas, temas do "Porquê?"),
   por isso só se escrevem uma vez.

Até à fase 10 havia uma quinta: cada ecrã da versão simples era o rascunho funcional do mesmo
ecrã da aprimorada. Cumpriu-se, e a simples saiu.

## Roadmap por fases

Catorze fases (0 a 13): as onze primeiras constroem o laboratório; a 11 e a 12 levam o que
ele aprende para fora dele, primeiro para um ficheiro e depois para previsões novas; a 13 junta
agentes de IA (um modelo de linguagem no próprio computador) aos agentes de regras da 9, e
faz-se antes da 12, a pedido. Cada uma
termina com algo que funciona, está testado e já se vê na interface (na simples até à fase 9,
na web desde a 10). Cada fase introduz poucas tecnologias novas, para as aprenderes uma de
cada vez.

| Fase | Backend | Interface simples (ecrãs que ganha) | Tecnologias novas | Pronto quando |
|---|---|---|---|---|
| 0 — Fundações ✅ | Repositório, configuração, logs, registo genérico, tipos, eventos, seeds; testes de arquitetura | Esqueleto com navegação entre ecrãs vazios | Python, uv, ruff, mypy, pytest, Streamlit | `pytest` e `mypy` passam; a app abre |
| 1 — Camada de dados ✅ | As nove etapas: entrada, validação, perfil, limpeza, tarefa, divisão, transformação, projeção, saída; linhagem | Dados, Explorar, Preparar | pandas, NumPy, Plotly | Carregar o Titanic mostra perfil correto, avisos e a sugestão "classificação binária"; o teste de arquitetura confirma o pandas isolado |
| 2 — Primeiro fluxo completo ✅ | Contrato de modelo, adaptador scikit-learn, regressão logística, avaliador de classificação, serviço de experiências | Modelo, Avaliar (básico) | scikit-learn, Pydantic | No Iris: escolher modelo, treinar e ver métricas na interface; testes de contrato a passar |
| 3 — Eventos e snapshots ✅ | Contexto de treino, gravador de snapshots, regressões didáticas em NumPy, dados da fronteira de decisão | Treinar (com reprodução passo a passo) | — (gradiente descendente, padrão observer) | Ver a fronteira da regressão logística didática a mover-se; resultado final igual ao do scikit-learn |
| 4 — Catálogo clássico ✅ | Todos os modelos supervisionados e não supervisionados; avaliadores de regressão, clustering e anomalias; diagnósticos; validação cruzada; importância por permutação | Avaliar (completo), Explicar, Laboratório | — (cada algoritmo e as suas métricas) | Todos os modelos passam os testes de contrato e têm ficha didática |
| 5 — Memória ✅ | Base de dados, registo de experiências, artefactos, comparação e reprodução de runs | Experiências | SQLite (`sqlite3`), joblib | Fechar a app, reabrir e comparar dois runs |
| 6 — Deep learning ✅ | Adaptador PyTorch, MLP, captura de ativações e gradientes, early stopping | Treinar mostra as camadas da rede | PyTorch | A MLP resolve "moons" e os snapshots mostram a fronteira a curvar |
| 7 — API ✅ | Rotas de dados, catálogo e experiências; treino em background; WebSocket | — (a simples continua a usar os serviços) | FastAPI, Uvicorn | Pela documentação em `/docs`: carregar CSV, lançar treino e receber snapshots |
| 8 — Reinforcement learning ✅ | Contratos de ambiente, agente e episódio; ambiente em grelha próprio; Q-learning didático; DQN em PyTorch | Reinforcement learning | — | Um agente aprende a atravessar a grelha e o run fica registado |
| 9 — Agentes (AutoML) ✅ | Agente de dados (limpeza, tarefa, divisão e transformações, cada decisão com a razão); um agente por família de modelos; coordenador com triagem e procura aleatória, escolhidas por validação cruzada só com o treino; orçamento, pausa e paragem; o vencedor treina como um run normal | Agentes (o modo automático, ao lado do manual) | — (procura de hiperparâmetros, validação cruzada para escolher) | No Titanic, os agentes preparam os dados, experimentam uma família de cada vez e afinam os finalistas sem tocar no teste; o vencedor usa o teste uma só vez, fica no histórico e abre-se no Avaliar |
| 10 — Interface aprimorada | Ajustes à API que a interface pedir: o agente que trabalha, as demonstrações dos modelos, as fórmulas das métricas | Todos os ecrãs refeitos em React com design próprio, barra lateral recolhível e um painel inicial com os dois caminhos | React, TypeScript, Vite, Tailwind CSS, shadcn/ui, Motion, TanStack, D3.js | Tudo o que a versão simples faz, com animações suaves e treino em tempo real; cada modelo e cada métrica explicados com exemplos visuais |
| 11 — Guardar e carregar o modelo ✅ | A preparação aprendida fica com o dataset e passa a JSON; o pacote `.mllab` (um ZIP com a ficha, o manifesto assinado, a entrada, a preparação, o run, o modelo e o ONNX); a biblioteca dos pacotes; carregar um pacote, com um aviso se vier de fora; o `PackageService` (ADR 0039) | Guardar: o ficheiro pré-visualizado parte a parte antes de existir, a biblioteca e o carregar | ONNX (`onnx`, `skl2onnx`, `onnxruntime`) | No Titanic, depois de treinar, o ecrã Guardar mostra o ficheiro antes de o gravar; guardá-lo descarrega um `.mllab` que abre em qualquer programa de ZIP; carregado noutra instalação, pede confirmação e o run abre no Treinar e avaliar; e o modelo de dentro dá as mesmas previsões do run original, no laboratório e no ONNX Runtime |
| 12 — Prever | Abrir um pacote, transformar linhas novas com a preparação guardada, prever com a explicação de cada previsão e avisos de extrapolação; um CSV inteiro de uma vez | Prever: o formulário gerado da entrada do pacote, a previsão ao vivo, o "e se…" e o ficheiro com a coluna da previsão | — | Escolher o pacote do Titanic, preencher um passageiro e ver a previsão mudar ao mexer na idade; um CSV inteiro sai com a previsão de cada linha |
| 13 — Agentes de IA ✅ | A interface `LanguageModel` e o cliente do Ollama, com o raciocínio à parte, saída estruturada e ferramentas; o estado do modelo como serviço; um agente de IA que propõe hiperparâmetros a pensar (o coordenador continua a medir); um analista que responde sobre a procura com ferramentas; as respostas guardadas para repetir a procura (ADR 0042) | Agentes: o agente de IA no fluxograma e o pensamento ao vivo na conversa; a comparação com a procura aleatória; o analista | Ollama (`qwen3:8b`) | Com o Ollama aberto, no Titanic, o agente de IA pensa à vista antes de cada proposta, as propostas medem-se como as outras sem tocar no teste, e o analista explica o vencedor usando as ferramentas; sem o Ollama, o ecrã diz como o ligar e tudo o resto funciona |

### Ordem de leitura recomendada para aprender

1. Ler a ficha didática do algoritmo antes de o implementar.
2. Implementar a versão didática (quando existir) e comparar com a do scikit-learn.
3. Só depois ligar à interface e observar os snapshots.
4. Registar uma decisão de arquitetura (ADR) sempre que algo não for óbvio.

## Começar a implementar

A implementação começa pela fase 0. No fim dela, o projeto instala-se com um comando, os
testes correm, as regras de arquitetura são verificadas e a interface simples abre com a
navegação entre ecrãs vazios.

### Fase 0, passo a passo

- [x] Instalar o uv e criar o projeto com o layout `src/ml_lab/`
- [x] Configurar o `pyproject.toml`: dependências, grupo de desenvolvimento (ruff, mypy,
  pytest) e as regras de cada ferramenta
- [x] Criar a árvore de pastas da secção Estrutura de pastas, com ficheiros vazios e um
  README curto em cada camada
- [x] `config.py` e `logging.py` com a biblioteca padrão
- [x] Fundação: tipos de tarefa, exceções do domínio, gestão de seeds
- [x] Registo genérico de componentes, com testes (registar, obter, filtrar, erro claro em
  nome desconhecido)
- [x] EventBus com testes (subscrever, publicar, cancelar subscrição)
- [x] Testes de arquitetura: `core` não importa `api`/`ui`; pandas só em `core/data/`;
  `foundation` independente
- [x] App Streamlit com a navegação entre os ecrãs do plano, todos vazios
- [x] README com instalação, como correr os testes e como abrir a app
- [x] Primeiras decisões de arquitetura registadas em `docs/adr/`

### Primeiras decisões a registar (ADR)

| N.º | Decisão | Porquê |
|---|---|---|
| 0001 | PyTorch em vez de TensorFlow | Loop de treino explícito, ideal para observar o treino passo a passo |
| 0002 | pandas isolado em `core/data/` | Fronteira clara entre dados tabulares e cálculo numérico; substituível |
| 0003 | Stack enxuta, extensões atrás de contratos | Menos tecnologias para aprender de cada vez, sem fechar portas |
| 0004 | SQLite com `sqlite3` | Sem dependências extra; o contrato do registo permite trocar mais tarde |
| 0005 | Interface simples (Streamlit) primeiro, React no fim | Validar todas as funcionalidades cedo; o redesenho final só muda a apresentação (substituída pela 0031: depois da fase 10, a simples saiu) |

### Regra para cada fase

Uma fase só termina quando o critério "Pronto quando" do roadmap é cumprido, `ruff`, `mypy`
e `pytest` passam, e as fichas didáticas dos algoritmos novos estão escritas.

## Notas da implementação

### Fase 0

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Grupos de dependências:** `dev` (ruff, mypy, pytest) e `ui` (Streamlit). A interface
  simples não faz parte do pacote `ml_lab`, por isso o Streamlit é um grupo e não uma
  dependência do pacote. `uv sync` instala os dois (`default-groups`).
- **`Paradigm`** tem só `SUPERVISED` e `UNSUPERVISED`. Deep learning é uma família de
  modelos, não um paradigma (uma MLP que classifica é supervisionada); o reinforcement
  learning entra na fase 8, com o seu `TaskType`.
- **Seeds derivadas por nome** a partir da seed raiz da experiência (ADR 0006).
- **Descoberta automática dos componentes** registados, para que um ficheiro novo não
  obrigue a editar uma lista de imports (ADR 0007).
- **EventBus:** os subscritores são chamados por ordem de subscrição, no fio de quem
  publica; se um falhar, a exceção chega a quem publicou (não fica escondida).
- **Regras de arquitetura extra**, tiradas da tabela de dependências: o núcleo (fora de
  `services/`) não importa `services/`; a interface e a API só chegam ao núcleo por
  `services/` e `schemas/`. A configuração (`ml_lab.config`) e os logs (`ml_lab.logging`)
  ficam fora do núcleo e podem ser usados por todas as camadas.
- **Streamlit** só escuta em `localhost` e não envia estatísticas de utilização
  (`.streamlit/config.toml`).
- **`.pre-commit-config.yaml` e `.github/workflows/ci.yml`** não foram criados: o próprio
  plano trata-os como extensão futura.

Pequenas inconsistências encontradas no plano, por resolver nas fases em que importam:

- As exceções do domínio aparecem em `core/errors.py` nas convenções e em
  `core/foundation/errors.py` na árvore. Ficaram na fundação, porque a fundação não pode
  importar nada de fora dela (e o registo precisa das exceções).
- "Adicionar uma fonte de dados" falava em `data/sources/` e em `DataSource.load() -> Dataset`;
  a árvore usa `data/io/` e a tabela de contratos diz que a fonte devolve "DataFrame +
  metadados". Resolvido na fase 1 a favor da árvore e da tabela de contratos (ADR 0009).
- O catálogo inicial dizia "Reinforcement learning (fase 7)"; no roadmap o RL é a fase 8
  (a 7 é a API). Corrigido acima para fase 8.
- "Adicionar um tipo de tarefa" refere `splitting.py`; na árvore, a divisão é a pasta
  `splitting/` com três estratégias. Na fase 1, cada estratégia regista-se em `SPLITTERS`
  (`splitting/base.py`): uma estratégia nova é um ficheiro novo nessa pasta.

### Fase 1

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Dependências:** `numpy`, `pandas` e `openpyxl` no pacote (o `openpyxl` é o motor que o
  pandas usa para ler `.xlsx`, não uma tecnologia nova); `plotly` no grupo `ui`, porque só a
  interface desenha; `pandas-stubs` no grupo `dev`, para o mypy conhecer os tipos do pandas. O
  Plotly não traz tipos, por isso o mypy trata-o como `Any` (`[[tool.mypy.overrides]]`).
- **Sem scikit-learn na camada de dados** (ADR 0008): transformações, divisão, geradores
  sintéticos e PCA em pandas/NumPy. Na fase 2 são comparados com o scikit-learn.
- **A fonte de dados devolve uma tabela** (`LoadedTable`) e o `Dataset` só existe à saída
  (ADR 0009). **Datasets de exemplo versionados** e confirmados com SHA-256 (ADR 0010).
- **`preparation.py`** (fora da árvore original) junta as etapas 4 a 9 pela ordem certa. Uma
  `PreparationConfig` (alvo, tarefa, receita de limpeza, divisão, transformações, projeção) e
  a seed da experiência reproduzem sempre o mesmo `Dataset`; a divisão usa
  `seeds.seed_for("split")` (ADR 0006).
- **Mais registos com o mesmo padrão plugin:** `DATA_SOURCES`, `SAMPLES`, `GENERATORS`,
  `CLEANING` e `SPLITTERS`. Uma fonte, um exemplo, um gerador, uma operação de limpeza ou uma
  estratégia de divisão nova é um ficheiro (ou uma entrada) novo, e os testes de contrato em
  `tests/contract/test_data_components.py` testam-na sozinhos.
- **Tipo da coluna ≠ tipo do pandas:** `ColumnKind` diz o que a coluna significa (um `int64`
  com 0 e 1 é booleano; inteiros seguidos e todos diferentes são um identificador; números
  escritos com vírgula decimal são numéricos guardados como texto). Texto livre e
  identificadores ficam com o papel `IGNORED`; colunas por converter ficam de fora do modelo
  com um aviso.
- **Avisos didáticos** (`Notice`, em `foundation/notices.py`): nível (info, aviso, erro),
  código estável e mensagem em português. Todas as etapas os produzem; a interface agrupa-os.
- **Números à portuguesa** (`ml_lab/formatting.py`: `19,9%`, `20 640`), usáveis por todas as
  camadas exceto a fundação, como `config` e `logging`.
- **Limpeza:** os duplicados procuram-se antes de remover os identificadores (sem o
  `PassengerId`, passageiros diferentes com os mesmos dados pareceriam repetidos). Os limites
  do IQR dos valores extremos são calculados antes da divisão; a descrição da operação avisa.
- **Tarefa:** as linhas sem alvo são removidas pela preparação, com um passo na linhagem.
- **Divisão:** o teste e a validação arredondam para cima, como no scikit-learn (891 linhas
  dão 712 de treino e 179 de teste); a estratificada reparte as classes pelo método do maior
  resto.
- **pandas 3:** o Copy-on-Write está sempre ligado (nenhuma etapa altera a tabela recebida) e
  o texto é lido com o tipo `str`. `ExcelFile.parse` e `Series.dt.weekday` estão
  descontinuados: como os avisos contam como erros nos testes, foram apanhados logo.
- **Serviço de datasets:** guarda até 20 datasets em memória, identificados por
  `SourceSpec.key`; os ficheiros enviados ficam em `data/uploads/` com o início do SHA-256 no
  nome. Devolve só estruturas simples, e `core/services/__init__.py` reexporta os tipos de que
  a interface precisa (a interface só pode importar `services` e `schemas`).
- **Interface:** a barra lateral é desenhada depois do ecrã, para mostrar o que o ecrã acabou
  de mudar; os controlos usam `persist_state="session"` para manter as escolhas ao mudar de
  ecrã; o ecrã Preparar volta a preparar a cada mudança. Os textos do painel "Porquê?" vêm do
  backend (`core/data/explanations.py`).
- **Testes da interface:** o `AppTest` do Streamlit perde a barra lateral quando uma app com
  várias páginas volta a correr; o teste da barra lateral monta o estado da sessão antes da
  primeira corrida.

### Fase 2

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Dependências:** `scikit-learn` e `pydantic` no pacote. O scikit-learn não traz tipos (o
  mypy trata-o como `Any`, como o Plotly); o Pydantic tem um plugin para o mypy
  (`plugins = ["pydantic.mypy"]`).
- **Configurações** (ADR 0011): as dataclasses da preparação ficam como estão e o
  `ExperimentConfig` (Pydantic) compõe-nas; os hiperparâmetros de cada modelo são classes
  Pydantic, e os erros de validação chegam em português.
- **Contrato `LabModel`** (ADR 0012): genérico nos hiperparâmetros, com o contexto de treino
  já na assinatura do `fit` (nulo até à fase 3), `classes()` para alinhar as probabilidades e
  os avisos das bibliotecas transformados em avisos didáticos. O `ModelFamily` só tem, por
  agora, os modelos lineares: cresce com o catálogo da fase 4.
- **Regressão logística:** envolve o `LogisticRegression` do scikit-learn 1.9, com
  regularização L2 e o otimizador L-BFGS (o parâmetro `penalty` está descontinuado e não se
  passa). Hiperparâmetros: `C` (em escala logarítmica no formulário), `max_iter`,
  `class_weight` e `fit_intercept`. A ficha didática está em
  `docs/algorithms/logistic_regression.md`.
- **Avaliação:** sete métricas de classificação (exatidão, exatidão equilibrada, precisão,
  recall, F1, ROC AUC e log-loss), cada uma com explicação. A exatidão equilibrada está
  escrita à mão, com a definição do scikit-learn, porque a dele avisa quando o modelo prevê
  uma classe ausente das linhas avaliadas. Uma métrica que não está definida (a ROC AUC sem
  todas as classes no teste) aparece como "—", com uma nota a explicar.
- **Comparações didáticas:** cada relatório compara o modelo com "prever sempre a classe mais
  frequente"; o diagnóstico compara treino e teste na métrica principal (a exatidão) e avisa a
  partir de 10 pontos de diferença (overfitting).
- **Serviço de experiências:** a seed da divisão e a do modelo derivam da seed raiz
  (`"split"` e `"model"`); sem seed, gera-se uma e fica registada na configuração do
  resultado, que repete o run tal e qual. Os problemas encontrados antes do treino (modelo
  desconhecido, hiperparâmetros inválidos, tarefa que o modelo não resolve) não publicam
  eventos; uma falha durante o treino publica `RunFailed`.
- **Catálogo:** cada cartão traz o JSON Schema dos hiperparâmetros e o texto da ficha
  didática. A interface desenha o formulário com o mesmo `components/forms.py` dos geradores
  sintéticos, que entende `"x-scale": "log"` e `"x-labels"`.
- **Implementações didáticas vs. scikit-learn:**
  `tests/unit/core/data/test_matches_scikit_learn.py` confirma que a escala, a imputação, o
  one-hot, a PCA (incluindo o sinal das componentes), os tamanhos da divisão e os dados
  sintéticos da fase 1 dão o mesmo que o scikit-learn.
- **Interface:** o ecrã Modelo mostra os cartões por família, a ficha didática e o
  formulário, e "Treinar e avaliar" leva ao ecrã Avaliar: cartões de métricas com explicação,
  matriz de confusão, treino vs. teste, pesos aprendidos e a configuração do run. A barra
  lateral mostra o modelo e o último resultado.
- **Testes:** os testes de contrato percorrem `MODELS` e `METRICS` (treinar em cada tarefa
  declarada, probabilidades, seed, ficha didática com todos os hiperparâmetros, métricas
  finitas, puras e explicadas); o teste de integração da fase 2 percorre o Iris pela
  interface, de Dados a Avaliar.

### Fase 3

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Sonda e snapshots amostrados** (ADR 0013): cada snapshot leva as previsões do modelo
  numa grelha do plano da projeção 2D, e por isso qualquer modelo mostra a sua fronteira. A
  amostragem guarda no máximo 60 fotografias, espalhadas pelo treino, mais a do fim.
- **Treino em segundo plano** (ADR 0014): `TrainingJob` (um fio de execução) e
  `TrainingControl` (pausa e paragem no fim de cada passo). Parar não é um erro: o modelo é
  avaliado com o que aprendeu, e o resultado leva um aviso.
- **Modelos didáticos:** `logistic_regression_gd` e `linear_regression_gd`, em NumPy, por
  gradiente descendente em lote (todos os exemplos em cada época). A logística minimiza o
  mesmo custo que o scikit-learn: a penalização `w / (C·n)` é a do scikit-learn dividida por
  `n`, o que não muda o mínimo. Com tolerâncias apertadas, os pesos coincidem até à quinta
  casa decimal. O treino pára quando o gradiente fica abaixo de `tol` (na linear, medido na
  escala do alvo), quando se pede, ou quando a loss explode: deixa de ser um número, ou fica
  um milhão de vezes pior do que no início (`supervised/gradient_descent.py`). Sem isso, os
  números transbordavam e o NumPy rebentava com avisos.
- **`step_limit` na ficha técnica:** o hiperparâmetro que limita os passos (`max_epochs`). É
  o que dá a barra de progresso; um teste de contrato confirma que o limite é respeitado.
- **Avaliador de regressão antecipado** da fase 4, porque a regressão linear didática
  precisava dele: R², RMSE, MAE e MSE (as siglas cabem nos cartões; as explicações dizem os
  nomes por extenso), com a comparação com "prever sempre a média" (R² = 0). O diagnóstico
  treino vs. teste deixou de usar artigo ("Exatidão: 96,7% no treino…"), para servir a
  qualquer métrica principal.
- **Geometria da projeção:** a `Projection` guarda o centro (a média do treino) e as
  posições das colunas; `project` leva pontos das features para o plano e `to_features` faz
  o caminho inverso.
- **Serviço de experiências:** primeiro o plano (seed, modelo, hiperparâmetros, dados,
  tarefa), que falha logo; depois o treino. `run` espera pelo resultado; `start` devolve um
  `TrainingJob`; `preview(run_id)` dá a fronteira do último snapshot de um treino a decorrer.
  O resultado ganhou `history`, `stopped_early`, `animation` e `training_notices`. Os
  últimos cinco runs ficam em memória, com o modelo e os dados (`artifacts(run_id)`), para
  as análises da fase 4.
- **Painel "Porquê?":** cada camada escreve os seus temas (`core/data/explanations.py`,
  `core/training/explanations.py`) e `services/explanations.py` junta-os.
- **Números:** `format_scientific` (`1 × 10⁻⁶`, nos seletores logarítmicos) e
  `format_significant` (quatro algarismos significativos, para uma loss que tanto pode ser
  0,3209 como 4,662 × 10⁹).
- **Interface:** no ecrã Modelo, "Treinar" lança o treino em segundo plano e abre o ecrã
  Treinar. Enquanto o treino decorre, um `st.fragment` relê o progresso a cada meio
  segundo: barra de progresso, Pausar/Retomar e Parar, a fronteira do último snapshot e a
  loss. No fim, a animação corre no navegador (frames do Plotly, slider e botão de
  reprodução), com a curva da loss e a exatidão no treino. O Avaliar passou a mostrar
  regressões (sem matriz de confusão, com um só vetor de pesos).
- **Testes:** unitários do controlo, do contexto, da amostragem, do gravador, do Trainer e
  dos jobs. Os testes com fios de execução sincronizam-se com `threading.Event`, sem esperas
  à sorte. Os dois modelos didáticos são comparados com o scikit-learn (pesos a menos de
  10⁻⁴ na logística e de 10⁻⁶ na linear). O contrato ganhou o limite de passos, os snapshots
  com a sonda e a paragem a pedido. O teste de integração da fase 3 treina nas luas: a
  fronteira desloca-se (7% da grelha muda de classe entre a primeira e a última fotografia)
  e o resultado final dá as mesmas previsões que o scikit-learn. No `AppTest`, um
  `st.switch_page` feito pela própria app não muda a página que o teste volta a correr: o
  teste muda-a à mão.

### Fase 4

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Catálogo:** 32 modelos em 10 famílias (`ModelFamily`, pela ordem do catálogo) e 21
  fichas didáticas. A versão didática e a do scikit-learn de um algoritmo partilham a ficha,
  tal como a classificação e a regressão do mesmo algoritmo. Seis modelos são didáticos,
  em NumPy: os dois por gradiente descendente da fase 3, o perceptron, o KNN, o k-means e a
  PCA. Os quatro últimos dão exatamente o mesmo que o scikit-learn, e os testes
  comparam-nos. Cada tarefa tem um modelo sugerido, que o ecrã Modelo escolhe primeiro
  (`SUGGESTED_MODELS`).
- **Tarefas sem alvo** (ADR 0015): o mesmo contrato, com `transform` (redução de
  dimensionalidade) e `anomaly_scores`, e `supports_new_data` para o t-SNE. A coluna de
  referência não entra no treino, só na avaliação, e pode estratificar a divisão: no ecrã
  Preparar, a estratificada vem escolhida sempre que há classes.
- **Modelos do scikit-learn que crescem** (ADR 0017): as árvores crescem por
  profundidade, as florestas por árvores (com o erro OOB), o boosting por etapas e a
  mistura de gaussianas por iteração do EM. Sem ninguém a ver, treinam de uma vez, com o
  mesmo resultado (um teste de contrato confirma-o).
- **Validação cruzada** (ADR 0016): `prepare_folds` repete as transformações em cada fold,
  só com o treino desse fold, e o teste fica de fora. A curva de aprendizagem usa os
  mesmos folds.
- **Explicações** (ADR 0018): importância por permutação, dependência parcial com curvas
  ICE e a explicação de uma previsão, em NumPy, para qualquer modelo.
- **Métricas:** 22, todas com explicação. Cada uma declara o que precisa (`needs`) e fica
  de fora do relatório quando falta alguma coisa: um clustering sem referência não tem
  Rand ajustado.
  - Clustering: silhueta (sem os pontos de ruído, numa amostra de até 2000),
    Davies-Bouldin, Calinski-Harabasz, número de grupos, proporção de ruído e Rand
    ajustado.
  - Redução de dimensionalidade: a confiança dos vizinhos (5 vizinhos, amostra de 1000).
  - Anomalias: proporção marcada, precisão, recall e ROC AUC da pontuação.

  As métricas principais são a exatidão, o R², a silhueta, a confiança dos vizinhos e a
  proporção de anomalias marcadas. As métricas que só descrevem têm
  `higher_is_better = None`: o diagnóstico ignora-as, e as contagens aparecem sem casas
  decimais ("3", não "3,000").
- **scikit-learn 1.9:** o `SVC(probability=True)` está descontinuado. O SVM dá
  probabilidades com `CalibratedClassifierCV`, com o método sigmoide e
  `ensemble=False`. Há um só SVM, treinado com todos os dados, e a calibração aprende-se
  por validação cruzada: até 5 folds, sem passar o número de linhas da classe mais rara. O
  `criterion` do gradient boosting também está descontinuado e não se passa.
- **Previsões que o scikit-learn não dá:**
  - DBSCAN: um ponto novo fica no grupo do núcleo mais próximo, se estiver a menos de
    `eps`; senão, é ruído.
  - Clustering hierárquico: um ponto novo fica no grupo do ponto de treino mais próximo.
    Só aceita até 5000 linhas de treino, porque guarda as distâncias entre todos os
    pares.
  - LOF: usa `novelty=True`, e as pontuações do treino vêm do `negative_outlier_factor_`.

  A pontuação de anomalia é `-score_samples` (maior = mais anómalo). As distâncias
  calculam-se aos blocos (`models/distances.py`), para a memória ficar controlada.
- **Diferenças para o catálogo inicial:**
  - O t-SNE não mostra o mapa ao longo das iterações, porque o scikit-learn não dá as
    posições intermédias. Mostra o mapa final, e só do treino.
  - O dendrograma do clustering hierárquico aparece no ecrã Avaliar, e não junção a
    junção no Treinar.
  - O perceptron mostra a fronteira época a época, e não a cada correção.
- **Gerador sintético "anomalias":** duas nuvens de pontos normais e alguns pontos
  espalhados ao acaso à volta delas. A coluna `target` (normal/anomalia) serve de
  referência.
- **Gráficos:**
  - A animação do Treinar serve a todas as tarefas. Mostra a probabilidade, a classe, o
    valor ou a pontuação de anomalia em cada ponto do plano, os centros dos grupos e as
    suas elipses (a dois desvios), e os pontos com o símbolo da classe da referência.
  - A redução de dimensionalidade tem o seu mapa (`visualization_data/embedding.py`).
  - O ecrã Avaliar tem os gráficos de cada tarefa (`visualization_data/evaluation.py`):
    curvas ROC e PR, resíduos, perfil de silhueta, dendrograma, variância explicada e o
    histograma das pontuações com o limiar. Com referência, o histograma é pintado com
    as anomalias verdadeiras: vê-se o que escapa ao limiar e os falsos alarmes.
- **Serviços:**
  - `AnalysisService` (ecrãs Avaliar e Explicar) usa o modelo e os dados dos últimos runs
    e guarda os cálculos pesados de cada run.
  - `LaboratoryService` faz as quatro demonstrações do Laboratório (overfitting, taxa de
    aprendizagem, o k do KNN, regularização), em dados pequenos feitos para mostrar cada
    ideia.
- **Interface:**
  - O ecrã Avaliar tem separadores conforme a tarefa: métricas, matriz de confusão,
    curvas, onde erra, grupos, silhueta, referência, mapa, pontuações, treino vs. teste,
    validação cruzada e o que o modelo aprendeu (a árvore aparece escrita como regras).
  - O Explicar calcula a importância a pedido, a dependência parcial de uma feature e a
    explicação de uma linha do teste.
  - O Laboratório tem quatro separadores, cada um com um controlo e o seu efeito
    imediato.
- **Testes:** os testes de contrato percorrem os 32 modelos e as 22 métricas. Nos modelos,
  verificam cada tarefa declarada, a seed, as probabilidades, o treino com e sem
  observador, o limite de passos, a paragem a pedido e a ficha. O teste de integração da
  fase 4:
  - treina todos os modelos de ponta a ponta em dados reais;
  - confirma que a referência nunca chega ao modelo;
  - percorre na interface o clustering com referência, a validação cruzada, o Explicar e
    o Laboratório.

### Revisão do fim da fase 4 e fronteira para o frontend

Uma revisão de todo o código (fases 0 a 4) corrigiu dois erros na avaliação de anomalias:

- a precisão e o recall davam 0% quando não havia nada para contar (sem anomalias
  verdadeiras, ou sem pontos marcados): agora ficam indefinidas, como a ROC AUC;
- a classe anómala era escolhida em cada parte dos dados. Agora é a preparação que dá o
  número 1 à classe mais rara da referência, contada na tabela toda.

A revisão também juntou código repetido:

- `class_probabilities` em `models/base.py`, em vez de quatro cópias;
- `undefined_notices` em `evaluation/base.py`, em vez de cinco;
- `group_name` para os nomes dos grupos.

Depois, a fronteira dos serviços ficou pronta para a API (ADR 0019):

- a serialização única, com a marca `@internal` e o teste de contrato da fronteira;
- o retrato do treino em segundo plano (`JobState`);
- a vista do que o modelo aprendeu (`ExperimentResult.learned`), que a explicação de uma
  previsão também usa;
- as opções de cada tarefa no ecrã Preparar (`task_options`), que estavam na interface.

As regras para o código novo estão em `CONTRIBUTING.md`.

Ficou por decidir uma coisa: nas tarefas sem alvo, as linhas sem valor na coluna de
referência saem da preparação, embora o modelo nunca use essa coluna. Mantê-las obrigaria
a uma classe "sem referência" nos avaliadores.

### Fase 5

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Registar sempre, guardar a pedido** (ADR 0020):
  - cada run que termina entra sozinho no registo, porque o `Lab` liga
    `HistoryService.record` ao evento `RunFinished`. Grava-se o resultado em JSON, sem o
    treino passo a passo;
  - "Guardar experiência" junta o modelo (`model.joblib`) e o treino passo a passo
    (`animation.json` e, numa rede, `network.json`) em `artifacts/runs/<run_id>/`.
- **O registo é um contrato** (`ExperimentStore`), com duas implementações que passam os
  mesmos testes: em memória, para os testes, e em SQLite (`SqliteStore`), para a app.
  - Duas tabelas: `runs`, com o documento de cada run, e `metrics`, com uma linha por
    métrica.
  - As migrações contam-se com `PRAGMA user_version`. Uma base de dados de uma versão mais
    nova do ML Lab não abre, e a mensagem diz para atualizar.
  - Cada operação abre a sua ligação, porque o registo é chamado no fio do treino.
  - Os erros do `sqlite3` passam a `TrackingError`. Um run que não existe dá
    `RunNotFoundError`, que também é um `LookupError` (a API da fase 7 vai responder 404).
- **Reabrir um run guardado devolve tudo.** O modelo vem do ficheiro e os dados preparam-se
  outra vez, com a mesma configuração e a mesma seed. O run volta à memória do serviço de
  experiências, por isso o Avaliar e o Explicar funcionam depois de fechar a app. Se o
  modelo não abrir, o resultado abre na mesma, com o aviso `not_restored`.
- **Comparar** (`services/comparison.py`) aceita de 2 a 5 runs e mostra:
  - as métricas lado a lado, com a melhor marcada (quando não há empate);
  - as escolhas que mudam: o modelo, a seed, cada hiperparâmetro (com o título do
    formulário) e cada passo da preparação;
  - as curvas da loss, com até 300 pontos cada;
  - uma leitura: reproduziu-se, só mudou a seed (é o acaso), mudaram várias coisas ao
    mesmo tempo, ou as tarefas são diferentes e não se comparam.
- **Reproduzir** volta a treinar a configuração registada, em segundo plano. O run novo
  entra no histórico, e a comparação com o original confirma que as métricas são iguais.
- **A partição principal** (`ExperimentResult.main_partition`) é o teste, ou o treino
  quando o modelo não tem métricas no teste. A barra lateral e o histórico dizem de que
  partição é a métrica ("no teste").
- **Os testes nunca tocam na base de dados verdadeira.** O `tests/conftest.py` aponta os
  uploads, os artefactos e a base de dados para uma pasta temporária, durante toda a
  sessão de testes.
- **Interface:**
  - o botão "Guardar experiência" fica na barra lateral;
  - o ecrã Experiências filtra o histórico por tarefa, dataset e modelo;
  - compara runs com uma tabela de métricas, um gráfico de barras, as escolhas diferentes
    e as curvas da loss;
  - abre, reproduz, guarda, muda o nome ou apaga um run.

  Um run aberto do histórico sem ter sido guardado mostra no Treinar porque não tem o
  treino passo a passo.
- **Testes:**
  - os dois registos passam os mesmos testes de contrato. O SQLite tem ainda testes do
    esquema, das migrações, de uma base de dados mais nova e de registos vindos de outro
    fio;
  - o teste de integração da fase 5 treina dois runs num `Lab` e guarda um. Depois abre
    outro `Lab` na mesma base de dados, como quem reabre a app: os dois runs estão no
    histórico e comparam-se, e o guardado reabre com o modelo (as curvas ROC voltam). Na
    interface, guarda pela barra lateral e compara no ecrã Experiências;
  - no browser: dois runs do Iris (C = 1, guardado, e C = 10⁻⁴), comparados e reabertos
    depois de reiniciar a app.

### Fase 6

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **O PyTorch fica atrás de um adaptador** (ADR 0021). `TorchModel` escreve uma vez o
  ciclo de treino e fotografa as camadas:
  - mini-lotes, Adam ou SGD com momento, regularização L2;
  - early stopping que fica com os pesos da melhor época;
  - a seed passa por um `torch.Generator` próprio.

  Duas bases dizem a loss e como se lê a saída:
  - `TorchClassifier`: softmax, cross-entropy e a exatidão em cada época;
  - `TorchRegressor`: alvo padronizado e erro quadrático médio.
- **MLP** (`models/neural/mlp.py`): `mlp` e `mlp_regression`, na família "Redes
  neuronais". Tem onze hiperparâmetros:
  - oito do treino: otimizador, taxa de aprendizagem, tamanho do lote, máximo de épocas,
    regularização L2, early stopping, paciência e fração para validação;
  - três da forma da rede: camadas escondidas (1 a 4), neurónios por camada (2 a 128) e a
    ativação (ReLU, tanh ou sigmoide).

  Por omissão, são duas camadas de 16 neurónios com ReLU, treinadas com Adam a 0,01. O
  catálogo tem agora 34 modelos em 11 famílias, e 22 fichas.
- **O que cada snapshot mostra da rede.** Em todas as camadas, o tamanho do gradiente
  (média na época). Nas camadas escondidas, medidas numa amostra de até 256 linhas de
  treino:
  - a ativação média de cada neurónio;
  - os neurónios parados;
  - um histograma das ativações.

  `visualization_data/network.py` junta os snapshots num `NetworkTrace`
  (`ExperimentResult.network`), com a proporção de neurónios parados já contada. A loss
  na validação entra nas métricas de cada época (`val_loss`), e o gráfico da loss
  desenha-a a tracejado.
- **O rasto da rede guarda-se como a animação** (ADRs 0020 e 0021). Numa rede grande pesa
  mais de 800 KB, por isso fica fora do registo automático e vai para `network.json` quando
  se guarda o run.
- **Detalhes que os testes apanharam:**
  - o softmax em float32 dava probabilidades que só somavam 1 a menos de 10⁻⁷, e o
    scikit-learn avisava na log-loss. Agora calcula-se em float64;
  - as matrizes do `Dataset` são só de leitura, e um tensor criado a partir delas sem
    cópia fazia o PyTorch avisar. As features copiam-se para float32;
  - com as luas por omissão (ruído 0,2), nenhum modelo passa muito dos 95% num teste de
    40 pontos: a rede fica nos 94,6% em média e a regressão logística nos 86%. Por isso, o
    teste da fase 6 usa ruído 0,1.
- **Interface:** no Treinar, depois da animação e da loss, vem a secção "As camadas da
  rede". Escolhe-se um passo do treino num slider, e vê-se:
  - o diagrama da rede: cada neurónio pintado com a sua ativação média, os parados
    esbatidos, até 24 neurónios por camada;
  - os gradientes por camada, em escala logarítmica;
  - a percentagem de neurónios parados ao longo do treino;
  - os histogramas das ativações.

  O painel "Porquê?" ganhou a rede neuronal, a retropropagação, o early stopping e a
  leitura de cada gráfico.
- **Testes:**
  - os testes de contrato percorrem as duas redes: a seed, o treino observado ou não, a
    paragem a pedido e o estado em JSON. Desta fase em diante, verificam também que todos
    os modelos se gravam com o joblib e reabrem a responder o mesmo;
  - os unitários da MLP confirmam que:
    - a rede resolve as luas;
    - a regressão aprende uma curva na escala original;
    - o early stopping fica com a melhor época;
    - uma ReLU que nunca liga conta como parada;
  - o teste de integração da fase 6 treina nas luas com ruído 0,1:
    - a rede passa os 97% de exatidão no teste, mais de 5 pontos acima da regressão
      logística;
    - mais de 10% da grelha muda de classe entre a primeira e a última fotografia;
    - uma reta não reproduz a fronteira final (acerta em menos de 90% da grelha);
  - no browser, com as luas de ruído 0,1, a rede chegou aos 100% no teste. O early
    stopping parou na época 121 com os pesos da 101. A fronteira ainda mal curvava no passo
    24 e contornava as luas no fim.

### Fase 7

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **A API é uma camada fina por cima dos serviços** (ADR 0022):
  - há uma rota por método público de cada serviço, com o nome do método. O id de cada
    operação no OpenAPI é `<serviço>_<método>`, e é o nome que o cliente gerado para o React
    vai ter;
  - `tests/api/test_contract.py` falha se algum método público ficar sem rota. As exceções
    estão numa lista, com a razão: `experiments.run` e `reinforcement.run`, porque a API
    treina sempre em segundo plano;
  - o `ServiceRouter` passa todas as respostas pelo `to_json` e tira o resumo de cada rota da
    primeira linha da docstring, em português.
- **Os treinos em segundo plano ficam nos serviços.** O `JobRegistry` guarda-os pelo id do
  run, e os serviços ganham `job`, `pause`, `resume`, `cancel`, `updates` e `result`. A API
  não guarda estado, e a interface simples passou a acompanhar o treino com o mesmo
  `updates`.
- **Acompanhar o treino é sempre o mesmo pedido:** `updates(run_id, after_step,
  after_snapshot)` devolve o estado, os passos novos e a fronteira do snapshot novo.
  - por HTTP, pergunta-se de vez em quando;
  - o WebSocket pergunta ao serviço de 0,25 em 0,25 s e envia só o que mudou, em mensagens
    `{"kind": "update" | "error", "data": ...}`.

  A fronteira só se calcula quando há um snapshot novo, porque desenha a grelha inteira.
- **Os erros mapeiam-se pela categoria:**
  - `TrainingStateError` → 409;
  - `DataError` e `ValueError` → 422;
  - `LookupError` → 404;
  - `TrackingError` → 500;
  - outro `MLLabError` → 400.

  O corpo é sempre `{code, message, problems}`. Entraram erros novos para as categorias
  ficarem certas:
  - `DatasetNotFoundError` (antes um `InvalidParameterError`, que daria 422);
  - `RunNotInMemoryError`;
  - `JobNotFoundError`;
  - `TrainingInProgressError` e `TrainingFailedError`.

  Os pedidos com a forma errada respondem `invalid_request`, com as frases em português do
  `describe_problems`.
- **Definições novas:** `ML_LAB_API_HOST` (por omissão 127.0.0.1, só este computador),
  `ML_LAB_API_PORT` (8000) e `ML_LAB_API_CORS_ORIGINS` (por omissão, o servidor do Vite).
  A API abre-se com `uv run python -m ml_lab.api`.
- **Dependências:**
  - o FastAPI (0.162, com o Starlette 1.9);
  - o Uvicorn, com o `websockets`, sem o qual não aceita WebSockets;
  - o `python-multipart`, para os uploads;
  - nos testes, o `httpx2`, porque o `TestClient` do Starlette 1.9 avisa com o `httpx`, e os
    avisos contam como erros.
- **O processo pede ao Windows para não ser travado** (ADR 0023). Ao preparar o teste da
  fase 7, uma MLP de 150 épocas demorava 9 s. A investigação passou pelo scikit-learn, pelo
  OpenMP, pelo garbage collector, pelo `torch._dynamo` e pelos números subnormais, e chegou
  ao EcoQoS do Windows 11: um processo sem janela em primeiro plano passa, ao fim de um
  segundo, para os núcleos lentos de um processador híbrido. `run_at_full_speed()` pede para
  não ser travado, no arranque da app, da API e dos testes. A MLP passou de 19,5 para 3,5 ms
  por época.
- **Testes:**
  - `tests/api/` cobre o contrato com os serviços, os erros, o treino (com pausa, paragem e
    WebSocket), os dados, o histórico e o RL;
  - o teste de integração da fase 7 envia um CSV das luas, treina uma MLP e recebe os
    snapshots enquanto treina, por HTTP e pelo WebSocket;
  - no browser, a documentação em `/docs` abre e lista as rotas.

### Fase 8

Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Um paradigma novo com a maquinaria de sempre** (ADR 0024):
  - o controlo é uma tarefa (`TaskType.CONTROL`, paradigma `REINFORCEMENT`), sem dataset;
  - o agente fala com o mesmo `TrainingContext` dos modelos, e cada episódio é um passo;
  - vêm de graça a pausa e a paragem, os snapshots, o `JobRegistry`, a API e o WebSocket;
  - `TaskType.learns_from_data` separa o controlo das tarefas com dataset, na deteção de
    tarefas e no catálogo.
- **O ambiente não guarda o estado do episódio:** `step(estado, ação, rng)` devolve o
  estado seguinte, a recompensa e se acabou. Com `rng=None` não há acaso, e é assim que se
  desenha o caminho que o agente quer seguir. Esgotar os passos não é um estado final, por
  isso o alvo continua a contar com o futuro.
- **A grelha** (`gridworld`) tem três mapas e um parâmetro para escorregar:
  - o simples (5×5, com paredes), o penhasco (4×12, com uma fila de armadilhas) e o
    labirinto (7×7, com becos e armadilhas);
  - as recompensas são +1 no objetivo, −1 nas armadilhas e −0,01 por passo;
  - escorregar leva o agente para um dos lados, nunca para trás.
- **Os agentes partilham o ciclo de episódios** (`EpisodicAgent`): ε-greedy com o ε a descer
  em linha reta. Num empate, a ação escolhe-se ao acaso; escolher sempre a primeira faria o
  agente subir sempre no início.
  - **Q-learning** (didático, NumPy): aprende os três mapas em décimas de segundo, com os
    caminhos mais curtos (8, 13 e 16 passos).
  - **DQN** (PyTorch): uma rede com o estado em one-hot, uma memória de repetição, uma
    rede-alvo e a loss de Huber. Aprende os três mapas, mas demora 4 a 11 s, porque treina a
    rede em cada passo.
- **A avaliação** tem o treino (os últimos 100 episódios, a explorar) e o teste (20
  episódios com a política aprendida, sem explorar). A métrica principal é a taxa de sucesso
  no teste. Há dois avisos:
  - `exploration_cost`: no penhasco, o agente chega 100% das vezes no teste e só 88% a
    treinar, porque cai a explorar;
  - `not_solved`: a política aprendida não chega sempre lá, num ambiente sem acaso.
- **O histórico serve os agentes.** O `ReinforcementResult` usa os nomes que o histórico e a
  comparação usam (o ambiente faz de dataset, e o agente de modelo):
  - os runs registam-se, guardam-se, comparam-se e reproduzem-se como os outros;
  - abrem-se com `open_reinforcement`;
  - o que o agente sabia em cada snapshot (`trace`, perto de 230 KB num run de 500
    episódios) só se guarda a pedido, em `trace.json`.
- **Interface:** o ecrã Reinforcement learning tem:
  - a escolha do ambiente e do agente, com os formulários gerados dos esquemas e a ficha
    didática;
  - a vista ao vivo: a recompensa e a política no último snapshot, com Pausar e Parar;
  - no fim, as métricas do treino e do teste, a recompensa por episódio (com a média móvel e
    a taxa de sucesso), a grelha com o valor e as setas da política (com um seletor de
    episódio), a tabela Q por ação e o botão para guardar.

  O ecrã Experiências abre e reproduz os runs de agentes neste ecrã.
- **Testes:**
  - os contratos de `tests/contract/test_rl.py` percorrem todos os agentes e ambientes:
    - os agentes aprendem a grelha simples e repetem com a mesma seed;
    - um episódio é um passo, e param a pedido;
    - os ambientes ficam dentro da grelha e o objetivo alcança-se;
  - os unitários cobrem a regra do Q-learning, o fim e o esgotar de um episódio, o
    escorregar e a memória de repetição;
  - o teste de integração da fase 8:
    - o Q-learning aprende o labirinto (16 passos, 100% no teste);
    - o run fica no SQLite e outro `Lab` reabre-o;
    - o DQN aprende a grelha simples;
    - o ecrã mostra o que o agente aprendeu, e o ecrã Experiências abre lá um run;
  - no browser, o Q-learning no penhasco (seed 42) chegou ao objetivo em 100% dos episódios
    de teste e em 86% a treinar, com um caminho de 15 passos. No episódio 112, a grelha ainda
    estava a branco, mas a ação "baixo" por cima do penhasco já era vermelha: o agente aprende
    a não cair antes de saber onde fica o objetivo.

### Fase 9

A fase dos agentes entrou no roadmap antes da interface aprimorada, que passou para a fase
10. Decisões tomadas ao implementar (as não óbvias têm ADR própria):

- **Dois modos, os mesmos ecrãs no fim.** Depois do ecrã Dados, o modo automático (o ecrã
  Agentes, novo ecrã 2) e o manual (Explorar, Preparar, Modelo, Treinar). O vencedor dos
  agentes é um run normal, que se abre no Avaliar e no Explicar, e qualquer candidato do
  quadro se treina à mão (`train_candidate`). O ecrã Dados oferece os dois caminhos.
- **Agentes deterministas, sem modelos de linguagem** (ADR 0025), em `core/automl/`:
  - o **agente de dados** decide a partir do perfil (nunca das linhas, por isso não usa o
    pandas): a limpeza sugerida, a tarefa, a divisão estratificada com 20% para o teste, a
    imputação, o one-hot e a padronização, cada decisão com a razão. Adota uma preparação
    feita à mão, e com classes desequilibradas sugere a exatidão equilibrada;
  - um **agente por família**, gerado do registo `MODELS`. As versões didáticas ficam de fora
    quando a família tem uma das bibliotecas (o mesmo modelo, mais devagar), e os modelos sem
    probabilidades quando o objetivo as mede;
  - o **coordenador**: a triagem (todos os modelos com os valores por omissão, uma família
    de cada vez, a referência à frente) e a afinação dos finalistas, à vez, por procura
    aleatória, sem repetir combinações. O orçamento conta candidatos.
- **A escolha nunca vê o teste.** O coordenador só recebe os folds do treino
  (`prepare_folds`); o teste usa-se uma vez, quando o vencedor treina como um run do
  `ExperimentService` (com a mesma seed dos folds). O resultado compara a validação cruzada
  com o teste e explica a diferença (o vencedor é sempre um pouco otimista).
- **O espaço de procura vive no modelo**: a extensão `"x-search"` do JSON Schema
  (`{"values": [...]}` ou `{"range": [mín, máx]}`, na escala do `x-scale`) anota os
  hiperparâmetros de todos os modelos supervisionados. Alguns valores ficaram de fora de
  propósito: o kernel polinomial do SVM (lento com C grande), o grau polinomial acima de 3
  (multiplica as colunas), a ativação sigmoide, o erro absoluto das árvores de regressão
  (muito mais lento) e o epsilon do SVR (depende da escala do alvo).
- **A validação cruzada mede qualquer métrica**: `cross_validate(..., metric=...)`, com as
  probabilidades quando a métrica as pede. Os objetivos possíveis são uma lista escolhida:
  a precisão e o recall, sozinhos, ganham-se com um modelo inútil, e o MSE ordena como a
  RMSE.
- **Datasets grandes procuram-se numa amostra**: `prepare_folds(..., max_rows=...)` tira uma
  amostra do treino (estratificada, pelo método de Hamilton) antes dos folds, e a linhagem
  di-lo. Por omissão, 2 000 linhas; o vencedor treina com o treino todo.
- **A procura é um treino em segundo plano** com o contrato de sempre (`job`, `pause`,
  `resume`, `cancel`, `updates`, `result`): cada candidato é um passo, e pausar ou parar
  acontece no fim do candidato atual. A conversa e os candidatos ficam no `SearchLog`,
  numerados; `updates` tem dois cursores (`after_step` para os candidatos e
  `after_message` para as mensagens), e o WebSocket usa o segundo como o do snapshot.
- **Erros novos:** `SearchNotFoundError` e `CandidateNotFoundError` (404).
- **Interface:** o ecrã Agentes tem o pedido (o alvo, a tarefa, a métrica sugerida pelo
  agente de dados, os agentes chamados e o formulário do tamanho da procura, gerado do
  esquema), a pré-visualização do que o agente de dados vai fazer, a vista ao vivo (o
  progresso, o melhor até agora, o gráfico da procura e as últimas mensagens), e no fim o
  vencedor, a leitura, os gráficos, o quadro e a conversa toda.
  - Os gráficos seguem as regras de visualização de dados: a paleta do projeto (Tableau 10)
    falhava a verificação para daltonismo com sete famílias num gráfico de pontos, por isso
    a cor diz só a ronda (as duas primeiras cores da paleta validada) e a família vai no
    texto e na posição (o gráfico "o melhor de cada agente" tem uma linha por família).
- **Testes:**
  - os contratos de `tests/contract/test_search_spaces.py` percorrem todos os modelos: cada
    valor que os agentes podem sortear é válido, e cada modelo supervisionado diz o que
    procurar (ou porque não precisa);
  - os unitários de `tests/unit/core/automl/` cobrem o agente de dados no Titanic, os agentes
    de modelo, o espaço de procura (a escala logarítmica, os inteiros), o objetivo e o
    coordenador (o orçamento, os finalistas, o sentido da métrica, as seeds, um candidato que
    falha, a pausa e a paragem, e as linhas que ele vê, todas do treino);
  - o teste de integração da fase 9: no Titanic, com 15 candidatos e 3 folds, o agente de
    dados tira o identificador e o texto livre, a triagem passa pelos 9 modelos das 7
    famílias, só os 3 finalistas são afinados, o vencedor bate a regressão logística na
    validação cruzada, passa dos 75% no teste e é o único run no registo, que outro `Lab`
    reabre; na interface, o ecrã Agentes faz um pedido e abre o vencedor no Avaliar, e um
    candidato treina-se à mão no ecrã Treinar;
  - no browser, o Titanic com os valores por omissão (24 candidatos, 5 folds, seed 42):
    - a procura demorou 17,5 s;
    - passaram à afinação o Gradient boosting (81,6%), o SVM e a Random Forest (80,2%);
    - ganhou um Gradient boosting afinado (`n_estimators` 200, `learning_rate` 0,0338,
      `max_depth` 4), com 82,9% ± 2,9% na validação cruzada e 85,5% no teste, 4,5 pontos acima
      da regressão logística;
    - o segundo ficou a 0,3 pontos, menos do que o desvio entre folds, e a leitura do
      resultado di-lo (um empate).

    A verificação apanhou quatro coisas, já corrigidas:
    - a razão da métrica dizia "Foi a métrica pedida" quando o formulário só confirmava a
      sugestão do agente de dados;
    - o espaço de procura mostrava as opções pelos códigos, e passou a usar os nomes do
      `x-labels`;
    - as linhas e os textos dos gráficos perdiam-se no tema escuro;
    - os cartões do vencedor ficavam cortados.

### Preparação da fase 10

Antes de começar o frontend, uma revisão do código e do contrato pela perspetiva de quem o vai
consumir, com os tipos TypeScript gerados do OpenAPI (ADR 0026). O que mudou:

- **Um esquema por tipo.** As dataclasses dos pedidos (`SourceSpec`, `PreparationConfig`,
  `CleaningStep`, `SplitConfig`, `TransformConfig`, `ProjectionConfig`) têm agora a sua
  própria configuração do Pydantic, dada na fronteira. Antes, herdavam a do modelo onde
  apareciam, e o OpenAPI tinha `ExperimentConfig-Input` e `ExperimentConfig-Output`.
- **As respostas prometem todos os campos.** `api/openapi.py` marca como obrigatórios todos
  os campos dos tipos que só aparecem em respostas (o `to_json` manda-os sempre); os tipos
  que também se enviam ficam com os seus opcionais.
- **O vocabulário** (`GET /vocabulary`): o nome a mostrar de cada valor de cada enum
  devolvido, e o que muda com cada tarefa. `Paradigm`, `NoticeLevel`, `CellKind` e
  `AgentRole` ganharam nomes. `GET /explanations` dá todos os temas de uma vez.
- **As regras de domínio viajam**: `ValidationReport.is_usable` passou de propriedade a campo;
  a métrica principal de cada tarefa está no vocabulário.
- **O contrato guardado**: `ui/web/openapi.json` (`scripts/export_openapi.py`), com um teste
  que falha quando fica desatualizado, e o guia do contrato em `ui/web/README.md` (as rotas
  de cada ecrã, os cursores do acompanhamento, os WebSockets, os códigos de erro, a
  formatação dos números e o que se calcula com o que já vem).
- **Erros 500 descritos**: as respostas de erro de todas as rotas, incluindo a 500, são um
  `ApiError` no OpenAPI.
- **Um erro corrigido nos agentes**: o serviço guardava o plano das últimas 10 procuras e
  esquecia o resto, mesmo o de uma procura ainda a correr (quem a acompanhava perdia-a, com
  "procura não encontrada"). Agora só esquece os planos de procuras que já acabaram.
- **Verificado sem mudar nada**: o registo em SQLite abre uma ligação por operação (aguenta
  pedidos em paralelo), a cache da análise só pode calcular duas vezes a mesma coisa, e 31
  resultados reais, de todas as tarefas, passam a JSON e voltam a validar contra o próprio
  tipo (nenhum `null` onde o esquema promete um número).

### Fase 10

A interface aprimorada, em `ui/web`. O pedido: tons escuros, uma barra lateral que se
recolhe em vez da barra no topo, um painel inicial para escolher entre testar um modelo à
mão e deixar os agentes procurar, os agentes a trabalhar à vista, uma página por modelo que
explique como ele funciona (com exemplos visuais e os hiperparâmetros), e o ecrã Treinar com
as avaliações e uma explicação visual de cada métrica.

**O desenho** (ADR 0029): "ardósia e giz". A interface é o quadro onde o laboratório
desenha: fundo de ardósia, texto a giz, a cor só para os dados e um giz amarelo que marca o
agora (o passo atual, o agente que trabalha, o foco do teclado). O que se lê é serifado
(Literata); o que se usa é sem serifa (Atkinson Hyperlegible Next). O plano passou pelo
processo do skill `frontend-design`, com uma revisão contra o que qualquer interface escura
de ML teria. Na 10.9, a ardósia e giz deu lugar ao sistema visual de uma imagem de referência
(ADR 0032), e a interface separou-se em duas, o Manual e os Agentes (ADR 0033).

**As etapas**, cada uma com o seu "Pronto quando":

| Etapa | O quê | Pronto quando | Estado |
|---|---|---|---|
| 10.1 | A base: o projeto, os tokens, as fontes, o cliente gerado, a verificação e os testes | Lint, tipos, testes e build passam, e uma página mostra o `/health` e o vocabulário vindos da API | ✅ |
| 10.2 | A moldura: a barra lateral, as rotas, o painel inicial, o telemóvel e a acessibilidade | Navega-se por tudo só com o teclado e, com a barra aberta ou fechada, nada fica cortado a 1440, 1024 e 390 px | ✅ |
| 10.3 | O backend que a interface pede: o agente que trabalha, as demonstrações, as fórmulas | `ruff`, `mypy` e `pytest` passam, o `openapi.json` está regenerado e as ADRs estão escritas | ✅ |
| 10.4 | Início (com o campo ao vivo), Dados, Explorar e Preparar | Um teste vai do início ao Titanic, explora, prepara e chega a Modelos, filtrado pela tarefa | ✅ |
| 10.5 | Modelos: a lista, a transição do nome e a página de cada modelo | Os 34 modelos abrem com exemplo e ficha (um teste percorre-os todos) e mudar um hiperparâmetro muda o exemplo | ✅ |
| 10.6 | Treinar e avaliar: o treino ao vivo e a gaveta de cada métrica | Um teste treina um modelo de ponta a ponta e abre todas as métricas, e tudo funciona com o movimento reduzido | ✅ |
| 10.7 | Agentes: o fluxograma ao vivo, a conversa e a classificação | Uma procura pequena passa por todas as fases no fluxograma, e pausar, retomar e parar funcionam | ✅ |
| 10.8 | Explicar, Experiências, Laboratório, RL, o glossário das métricas e o acabamento | Os 11 ecrãs da interface simples têm equivalente e passam todas as verificações; aí decide-se o que fazer à interface Streamlit | ✅ |
| 10.9 | O redesenho: a entrada, as duas interfaces, o sistema visual da referência e as explicações atrás do "?" | As duas interfaces abrem da entrada, cada uma com a sua barra; os temas, o "como se lê" dos gráficos e as ajudas dos campos estão no "?"; a paleta passa no validador (menos a banda de luminosidade, de propósito); os testes de ponta a ponta e o axe passam, e nenhum ecrã desliza para o lado a 390 px | ✅ |

O ecrã Avaliar deixa de existir à parte: entra em "Treinar e avaliar". A interface Streamlit
continuou a funcionar até à 10.8 e saiu depois dela (ADR 0031; ver a última nota desta fase).

**10.1, a base** (ADR 0027):

- React 19, TypeScript 6.0, Vite 8, Tailwind CSS 4, shadcn/ui (com o Radix) e Motion;
  TanStack Router e TanStack Query; Zustand para o pouco estado do cliente.
- **TypeScript 6 e não o 7**: o 7.0 é o mais recente, mas o `typescript-eslint` só aceita até
  ao 6.0.
- **O cliente gera-se do contrato** com o `@hey-api/openapi-ts`: os tipos, uma função por
  operação e as opções do TanStack Query, a partir do `openapi.json`. Não vai para o Git e
  volta a gerar-se antes de cada comando.
- **A mesma origem**: os pedidos vão para `/api`, e o Vite passa-os à API, com os WebSockets,
  no endereço de `ML_LAB_API_HOST` e `ML_LAB_API_PORT` (os mesmos do backend).
- **Os números como no Python**: `lib/format.ts` segue `ml_lab/formatting.py` com o
  arredondamento metade para o par. Sem isso, `0,8125` dava `81,3%` em vez de `81,2%`; os
  testes usam os mesmos exemplos dos de Python.
- **Os erros** chegam sempre como `ApiRequestError`, com o `code` da API. Um pedido sem
  resposta tem o código `api_unreachable`, e a página diz como arrancar a API.
- As fontes servem-se do próprio projeto, e as duas têm algarismos tabulares (confirmado no
  browser), por isso as tabelas alinham os números sem uma monoespaçada.
- Ficou no `launch.json` das pré-visualizações como `ml-lab-web`.

**10.2, a moldura**:

- A barra lateral tem 256 px aberta e 48 px recolhida. Abre e fecha com `Ctrl+B`, lembra-se
  do estado e, no telemóvel, passa a gaveta (que se fecha ao escolher um ecrã).
- Mostra os passos do caminho escolhido, numerados porque são uma sequência:
  - à mão: Dados, Explorar, Preparar, Modelo, Treinar e avaliar, Explicar;
  - com os agentes: Dados, Agentes, Treinar e avaliar, Explicar.
- Mostra também "Aprender", o RL, as Experiências e o estado da API. Um ecrã aparece uma só
  vez: no caminho à mão, os Modelos são o passo 4.
- As páginas reorganizam-se pela largura que têm (*container queries*). Com a barra
  recolhida, o início passa de uma para duas colunas a 1024 px.
- Os ecrãs que ainda não existem dizem em que etapa chegam.
- **Os testes de ponta a ponta** (Playwright) correm sobre o build final, com a API na porta
  8001 e os dados numa pasta temporária. Testam:
  - o teclado e o `Ctrl+B` a 1440 e 1024 px, com o conteúdo a ganhar o espaço sem cortes;
  - a gaveta a 390 px e a troca de caminho;
  - a acessibilidade com o axe, em três páginas.
- Correr sobre o build evita as recargas do servidor de desenvolvimento, que fazia falhar o
  primeiro teste ao otimizar as dependências. O axe apanhou o cabeçalho e o rodapé da barra
  fora do `nav`, e a barra passou a ser toda a navegação.

**10.3, o backend que a interface pede**:

- **O agente que trabalha** (`AutoMLUpdate.working`): o candidato que um agente está a
  medir, o que muda nele e o fold em que vai. A `cross_validate` avisa a cada fold, e o
  coordenador anuncia o candidato antes de o medir. Antes disto, um candidato só aparecia
  depois de medido, e o fluxograma não sabia quem estava a trabalhar.
- **As demonstrações dos modelos** (`CatalogService.demo`, ADR 0028):
  - cada modelo treina num conjunto de exemplo pequeno da sua tarefa (luas, círculos, nuvens,
    uma onda…), num serviço de experiências só seu, sem ninguém a registar: nada vai para o
    histórico;
  - as 34 demonstrações de omissão demoram 4,6 s ao todo, e as últimas ficam guardadas;
  - a regressão ganhou um gerador, a **onda** (um seno com ruído);
  - o cartão do modelo ganhou a ficha por secções, as demonstrações do Laboratório que se
    aplicam e os conjuntos de exemplo;
  - o `LearnedView` ganhou a árvore como estrutura, para a desenhar.
- **As fórmulas das métricas** (ADR 0028):
  - cada uma das 22 tem a fórmula em LaTeX e o significado de cada símbolo
    (`CatalogService.metrics`);
  - 14 dão os termos da fórmula com os números do run (`MetricResult.terms`), por classe na
    multiclasse.
- O contrato tem agora 72 operações. Os testes de contrato confirmam que:
  - cada modelo tem demonstração e a ficha com as seis secções do modelo;
  - cada métrica tem fórmula, e cada símbolo explicado está nela;
  - cada termo é um símbolo da fórmula, e os termos dão o valor da métrica.

**As fundações dos ecrãs** (para a 10.4 em diante):

- **Os gráficos** são componentes próprios, em D3 e SVG (`components/charts/`), com a
  paleta do skill de visualização de dados validada sobre a ardósia: linhas, barras,
  histogramas, barras de 100%, mapas de calor, dispersões, o campo de decisão (em canvas, com
  as fronteiras em SVG), a árvore, a rede, a silhueta, o dendrograma e o mapa dos nulos. Todos
  têm a dica ao passar o rato (o valor primeiro), a legenda com duas séries ou mais e a
  legenda de leitura por baixo.
- **A cor segue a classe**: a classe "1" tem a mesma cor no equilíbrio das classes, na
  projeção e no campo, pela ordem do `class_labels` do backend. Com mais de três classes numa
  dispersão, a forma também muda.
- **Os formulários geram-se do JSON Schema** (`SchemaForm`): os hiperparâmetros, os
  parâmetros dos geradores e o tamanho da procura dos agentes, sem código para cada um.
- **Os textos de leitura vêm do backend** (ADR 0030): o "como ler" de cada gráfico é um tema,
  pedido pelo nome, e um teste procura esses nomes no código das duas interfaces.

**10.4, os ecrãs dos dados**:

- **Início**: abre com uma rede neuronal a aprender as luas (a demonstração do catálogo), que
  se reproduz uma vez e fica no último passo; com "reduzir movimento", mostra logo o fim.
  Por baixo, o que ficou aberto da última vez e os dois caminhos.
- **Dados**: os exemplos, um ficheiro (arrastado ou escolhido) e os geradores. Com um dataset
  aberto, ele vem primeiro: os factos, os avisos da validação, as primeiras linhas, os tipos
  das colunas e porquê, e o passo seguinte (à mão ou com os agentes).
- **Explorar**: os números do dataset e uma faixa com uma marca por coluna, na cor do tipo
  inferido; os avisos (os primeiros cinco, e os outros a pedido); as colunas uma a uma (as
  estatísticas, o histograma ou os valores mais frequentes), as correlações, o mapa dos nulos
  (marcas de giz na ardósia) e o alvo (o equilíbrio das classes e as médias de cada feature por
  classe, cada uma na escala da sua coluna).
- **Preparar**: os quatro passos da preparação, por ordem, e o resultado ao lado, que volta a
  correr 350 ms depois do último controlo mexido: as partes da divisão, a linhagem, as classes
  em cada parte (barras de 100%: na divisão estratificada, alinham), os dados antes e depois e
  a projeção 2D (a cor é a classe, a forma é a parte).
- **As escolhas, não a configuração**: o ecrã guarda o que o utilizador escolheu, e o que ele
  não tocou segue as sugestões do backend. Trocar de alvo volta às sugestões do alvo novo. A
  preparação que corre bem fica na sessão (é com ela que se treina); a que falha tira-a.
- **Um id que se perdeu**: os datasets vivem na memória da API. Se ela recomeçar, os ecrãs
  dizem que é preciso carregar o dataset outra vez, em vez de falharem.
- **Um erro que só se via no browser**: as escalas do d3 devolvem `rgb(…)`, e o campo lia as
  cores como `#rrggbb`, por isso pintava tudo a preto. A leitura das cores passou a aceitar os
  dois formatos, com testes.
- **A prova** (`e2e/manual-path.spec.ts`): do início, "Começar à mão" leva aos Dados; o teste
  carrega o Titanic, vê os tipos das colunas, explora (os avisos, as correlações, o alvo),
  prepara (a linhagem) e chega aos Modelos com o filtro na classificação binária, sem os de
  regressão. Um segundo teste troca o alvo para `Fare` e confirma que a tarefa passa a
  regressão.

**10.5, os modelos**:

- A lista mostra só o nome e uma seta, por família, filtrada pela tarefa preparada (o filtro
  fica no endereço, `?tarefa=…`). Ao abrir um modelo, o nome desliza para o título (o mesmo
  `layoutId` do Motion; a rota espera pelo cartão, para os dois nomes existirem no mesmo
  instante).
- A página lê-se como um capítulo: a intuição; o modelo ao vivo num conjunto de exemplo, que
  volta a treinar 300 ms depois de mudar um hiperparâmetro (a fotografia anterior fica à
  vista, esbatida); o que aprendeu (os pesos, a árvore desenhada, a importância das features,
  as tabelas, a rede no mesmo passo do campo); o resto da ficha; e as demonstrações do
  Laboratório.
- "Treinar com estes valores" usa a preparação da sessão e segue para o treino; sem
  preparação, ou com uma de outra tarefa, diz o que falta.
- O backend decide os temas do "Porquê?" de cada modelo (`ModelCard.topics`).
- Os controlos deslizantes não tinham nome para os leitores de ecrã (o rótulo apontava para a
  raiz, e quem tem o foco é o polegar): passaram a ter o nome e o valor escrito, que num
  controlo em passos logarítmicos é o número e não a posição.
- **A prova** (`e2e/models.spec.ts`): os 34 modelos abrem, cada um com o exemplo desenhado, a
  ficha e o que aprendeu; na árvore de decisão, baixar a profundidade máxima de 5 para 2 volta
  a pedir o exemplo, e a animação passa a acabar no passo 2 (antes, no 5).

**10.6, treinar e avaliar**:

- **Ao vivo**, pelo WebSocket: o estado com o ponto amarelo (o agora), o passo, a loss e a
  fronteira no último snapshot, com Pausar, Retomar e Parar.
- **No fim**: o campo passo a passo (com a rede no mesmo passo, numa rede), a loss e as
  métricas de cada passo em gráficos separados (escalas diferentes nunca partilham dois
  eixos), as métricas no treino e no teste, os gráficos da tarefa (a matriz de confusão por
  parte, as curvas ROC e de precisão-recall, os resíduos, a silhueta, o dendrograma, as
  pontuações de anomalia, a variância explicada, onde o modelo erra), a validação cruzada e a
  curva de aprendizagem a pedido, e o que o modelo aprendeu.
- **A gaveta de cada métrica**: o valor em cada parte, o que mede, a fórmula em KaTeX com o
  significado de cada símbolo, a conta deste run (os termos da fórmula, cada um com uma barra
  à escala do maior, a aparecerem um a um, e o resultado no fim) e como se lê. As métricas sem
  termos mostram o gráfico de onde o valor sai (a curva ROC, a silhueta, os grupos).
- O backend ganhou o "como ler" do campo (`TrainingAnimation.how_to_read`) e o nome das
  métricas de cada passo (`Vocabulary.step_metrics`).
- **A prova** (`e2e/training.spec.ts`, com "reduzir movimento" ligado): treina-se a regressão
  logística no Titanic pelo ecrã e abre-se a gaveta de cada métrica. Depois, um run de cada
  tarefa abre as 22, cada uma com a fórmula, a conta do run e o "como se lê". Os cinco runs
  treinam-se um de cada vez, porque a API só guarda os últimos cinco em memória e os outros
  testes também treinam.

**10.7, os agentes**:

- **O pedido**: o alvo, a tarefa, a métrica, os agentes chamados, o tamanho da procura e a
  seed, com o plano do agente de dados ao lado e o fluxograma dos agentes que vão trabalhar.
- **O fluxograma**, em SVG e Motion: Dados, Coordenador, um nó por agente de família, Líder (o
  Vencedor, no fim) e Teste. Só o que acontece agora fica a amarelo: o agente que mede um
  candidato tem o fold em que vai numa barra, e a ligação até ele corre. Os nós medem a
  largura que têm e escolhem a versão mais longa do texto que cabe.
- Por baixo: as fases por ordem, o melhor até agora, os candidatos medidos (o mais recente em
  cima; o lugar de cada um só se decide no fim, pelo backend) e a conversa dos agentes. No
  fim: o vencedor e a sua nota no teste, a procura num gráfico, o melhor de cada agente, o
  quadro completo e a ponte para treinar qualquer candidato à mão.
- **O backend**: o plano de uma procura passou a ter rota (`automl_search_plan`), porque as
  atualizações só trazem o que muda e o fluxograma precisa dos agentes desde o início; e o
  WebSocket passou a mandar uma mensagem também quando só muda o trabalho a decorrer (o
  candidato, o fold, a fase), com dois testes.
- **A prova** (`e2e/agents.spec.ts`): uma procura no Titanic com dois agentes (os modelos
  lineares, rápidos, e as redes neuronais, lentas), 6 candidatos e 3 folds pausa, retoma e
  acaba; outra pára a meio e acaba com o melhor até aí. O fluxograma mostra a triagem e o fim.
  As fases mais curtas que o intervalo do WebSocket (0,25 s), como a do agente de dados ou o
  treino do vencedor, podem começar e acabar entre duas mensagens; por isso o teste
  confirma-as na conversa, que as tem todas. E o fluxograma só aparece quando chega o plano
  da procura: se o plano chegar depois das primeiras mensagens, o fluxograma nasce já numa
  fase. O teste regista também a fase com que ele aparece, e não só as mudanças do atributo
  (sem isso, falhou uma vez, sem a triagem).

**10.8, o resto e o acabamento**:

- **Explicar**: as três perguntas, pela ordem em que se fazem. O que pesa no conjunto (a
  importância por permutação, a pedido, porque volta a prever o teste muitas vezes); como a
  previsão muda com uma feature (a dependência parcial, com as curvas individuais e os traços
  onde há dados); e uma previsão de perto, linha a linha. O "como se lê" dos contributos vem
  com eles (`RowExplanation.how_to_read`), porque depende do método; os dos outros dois
  gráficos são temas (`importance_bars`, `dependence_curves`).
- **Reabrir um run pelo histórico**: quando a API responde `run_not_in_memory` (só guarda os
  últimos runs), o Treinar e avaliar, o Explicar e o ecrã do RL pedem o run ao histórico, como
  o plano previa. Um run guardado volta com o modelo, e com ele os gráficos e o Explicar; só se
  perdeu um run que também não está no histórico. O botão "Guardar experiência", que a
  interface simples tinha na barra lateral, está nos dois ecrãs de treino.
- **Experiências**: a tabela com os filtros no endereço; a comparação de dois a cinco runs (a
  leitura do backend, as métricas com o melhor marcado, as percentagens em pontos de 0 a 100%,
  as escolhas que mudam e as curvas da loss); e a gaveta de cada run, para o abrir no ecrã que
  é o dele, reproduzir, guardar, mudar o nome e apagar (com confirmação). As escolhas da
  preparação passaram a escrever-se com nomes ("Divisão · estratégia: Estratificada",
  "Limpeza: Remover colunas (Cabin)") em vez do JSON da configuração e do nome do campo.
- **Laboratório**: as quatro demonstrações em separadores, com a aberta no endereço (a página
  de um modelo liga-se à que o explica). Os valores de cada controlo estavam nas páginas do
  Streamlit; passaram para o backend (`LaboratoryService.demos`), com o resumo de cada
  demonstração, e as duas interfaces leem-nos (regra 4: de onde a taxa de aprendizagem se
  arrasta até onde explode é uma escolha didática). As demonstrações recusam, com a razão, os
  valores que não podem usar; antes, um grau fora de 1 a 15 dava um `ValueError` em inglês.
  Gráficos novos: as curvas de nível da taça do erro, com o caminho dos pesos por cima, e as
  escalas logarítmicas em x.
- **Reinforcement learning**: o pedido (o ambiente, o agente com a ficha didática, os
  hiperparâmetros e a seed); o treino ao vivo (a recompensa de cada episódio e a política no
  último snapshot); e, no fim, as métricas no treino e no teste, a recompensa e a parte dos
  episódios que chega ao objetivo, em dois gráficos (a interface simples punha-as num só, com
  dois eixos), a grelha episódio a episódio e a tabela Q em quatro grelhas. As paredes, as
  armadilhas e o objetivo têm um símbolo e o nome na legenda, não só uma cor. As métricas de
  cada episódio entraram no registo `step_metrics`, com um teste de contrato.
- **Métricas**: o glossário das 22, com o filtro pela tarefa e uma âncora por métrica (a
  gaveta de cada métrica liga-se a ela). Os símbolos das fórmulas passaram a desenhar o LaTeX do
  seu significado ("do exemplo $i$"), também na gaveta.
- **Um só comando** para trabalhar na interface: `uv run python scripts/dev.py` arranca a API
  e o Vite, e `Ctrl+C` pára os dois; se um parar sozinho (uma porta ocupada), o outro pára
  também. No Windows, o npm corre dentro de um `cmd.exe`: parar só esse processo deixava o
  Vite órfão, por isso pára-se a árvore toda (`taskkill /T`).
- **A acessibilidade**: o axe passa em todos os ecrãs. Os que abrem sem nada antes passam no
  teste da moldura; os outros, nos seus testes, com os dados à vista (um run explicado, a
  comparação aberta, as quatro demonstrações, um agente treinado e o glossário). A auditoria
  encontrou dois problemas, corrigidos em todos os ecrãs:
  - o esbatido do resultado anterior, enquanto o novo chega, baixava o contraste do texto
    abaixo de 4,5:1 (o axe apanhou-o a meio da transição, na página de um modelo). Agora só os
    desenhos se esbatem (a classe `.stale`);
  - no telemóvel, os textos só para leitores de ecrã dentro das tabelas, e o MathML que o
    KaTeX junta às fórmulas, têm posição absoluta e escapavam às caixas que deslizam,
    alargando a página. As caixas passaram a ser posicionadas (`relative`), e uma fórmula larga
    desliza dentro da sua. Um teste confirma que nenhum ecrã desliza para o lado no telemóvel.
- **O guia para mexer na interface** está no `ui/web/README.md`: que ficheiro muda as cores,
  os tipos de letra, os componentes, a moldura, os gráficos e cada ecrã, e as regras do sistema
  visual que dão coerência ao todo.
- **A prova** (`e2e/explain`, `history`, `laboratory`, `reinforcement` e `metrics`): um run
  explicado nas três partes; dois runs comparados, renomeados, guardados, apagados e
  reproduzidos; as quatro demonstrações, com um controlo que muda o resultado; um agente
  treinado, percorrido episódio a episódio e guardado, e outro pausado e retomado; e o glossário
  com as 22 fórmulas e o filtro pela tarefa. Com as etapas anteriores, os 11 ecrãs da interface
  simples têm equivalente na web.
- **A interface Streamlit**: funcionou até ao fim desta etapa, com os textos novos (as duas
  liam os mesmos temas e os mesmos controlos). A decisão sobre ela veio logo a seguir: saiu
  (ADR 0031, na nota seguinte).

**Depois da 10.8, a interface Streamlit sai** (ADR 0031):

- **Porquê**: com a web a fazer tudo o que a simples fazia (os 11 ecrãs têm equivalente, com
  testes), manter as duas era fazer cada ecrã duas vezes, ter duas portas para o núcleo (os
  serviços chamados no mesmo processo, e a API) e instalar o Streamlit e o Plotly só para ela.
- **O que saiu**: `ui/simple/` (os ecrãs, os componentes e os gráficos em Plotly),
  `.streamlit/`, a configuração `ml-lab-ui` do `launch.json` das pré-visualizações, o grupo de dependências
  `ui` (o Streamlit, o Plotly e o que só eles traziam: 19 pacotes) e o que o ruff e o mypy
  tinham só para a interface (a pasta `ui` nos caminhos, a exceção do Plotly).
- **O que ficou**: o backend todo. Uma procura de código morto (o `vulture`, com e sem
  `ui/simple`) não encontrou nada que só servisse a interface simples: os serviços, os
  textos, os temas e os dados dos gráficos servem a API. Mudaram só os textos que nomeavam os
  ecrãs do Streamlit (docstrings, fichas e um aviso do histórico): "o ecrã Avaliar" passou a
  "a avaliação", e "o ecrã Treinar" a "o Treinar e avaliar" ou "o treino passo a passo".
- **Os testes** em Python passaram de 2091 para 2044. Saíram os que usavam o `AppTest` (o
  `test_simple_ui.py` e as partes da interface nos testes de integração de cada fase) e os
  que liam o código do Streamlit. Ficaram os dos serviços, que provam o critério de cada fase,
  e a interface tem os seus testes de ponta a ponta, na web. A regra "a
  interface e a API só importam `services` e `schemas`" passou a ser só da API. A que proíbe o
  núcleo de importar bibliotecas de interface ou de gráficos (o Streamlit e o Plotly
  incluídos) ficou, para não voltarem pela porta das traseiras. A procura dos temas pedidos
  pelo nome (`test_topics_in_use.py`) lê agora só o código da web.
- **Para a ver outra vez**: a última versão com a interface simples está no commit `dcdcc5e`.

**10.9, o redesenho** (ADR 0032 e 0033):

- **O pedido**:
  - separar a interface dos agentes da de testar à mão;
  - uma página de entrada com o nome da app e as duas escolhas, sem barra lateral;
  - cada interface com as suas secções na barra;
  - as cores, a letra e o contraste de uma imagem de referência: preto, cinzentos escuros e
    branco, com muita cor nos gráficos;
  - menos texto à vista, com as explicações atrás de um botão discreto;
  - o espaçamento devido entre cartões, ícones e tabelas.
- **As skills**: instalaram-se as da Vercel (`web-design-guidelines`, `react-best-practices` e
  `composition-patterns`). Nenhuma desenha um aspeto: são regras de qualidade, de desempenho e
  de composição. Por isso:
  - o desenho seguiu o `frontend-design` (os tokens primeiro, medidos na referência);
  - as cores dos gráficos seguiram o `dataviz`, com o validador da paleta;
  - a `web-design-guidelines` ficou para a auditoria final.
- **A entrada e as duas interfaces** (ADR 0033):
  - a entrada (`/`) fica fora da moldura, com um cartão por interface. O do Manual tem a rede
    neuronal a aprender as luas; o dos Agentes, o coordenador a distribuir o trabalho;
  - `NAVIGATION` diz as secções da barra de cada interface. Os ecrãs que só existem numa
    (`ONLY_IN`) trazem a interface deles quando se abrem pelo endereço;
  - a troca fica no topo da barra, e `Ctrl K` abre uma pesquisa de ecrãs, modelos e métricas
    (`cmdk`).
- **O sistema visual** (ADR 0032):
  - a Geist e a Geist Mono substituem a Literata e a Atkinson;
  - os cinzentos são os da referência, com o dos rótulos pequenos um degrau mais claro, para
    chegar aos 4,5:1;
  - a paleta dos dados tem oito cores validadas, com rampas azul e âmbar e uma escala
    divergente laranja e azul;
  - as linhas têm um brilho (uma cópia desfocada por baixo).
- **Menos texto**:
  - `Explain` é o "?" ao lado de cada título de cartão ou de secção, e o "Como se lê" de cada
    gráfico, que deixou de ficar escrito por baixo;
  - `PageExplain` é o "?" do título da página: abre a gaveta com o que se faz no ecrã e as
    ideias por trás (o antigo "Porquê?");
  - as ajudas dos campos dos formulários passaram para o "?" ao lado do nome;
  - os temas continuam a pedir-se pelo nome, e `test_topics_in_use.py` continua a
    encontrá-los.
- **Os ecrãs** passaram para cartões (`Panel`), com os números grandes em `Stat` e o mesmo
  espaço entre cartões. As tabelas têm o cabeçalho em mono pequeno e a linha escolhida um
  degrau acima. Na ficha de cada modelo, as secções passaram a separadores, a começar pela
  intuição.
- **O que a revisão nos ecrãs apanhou**:
  - Com os gráficos todos em mono, os nomes das categorias saíam das margens (no gráfico de
    pontos da comparação, "Recall (sensibilidade)" passava 30 px para fora). Ficaram em mono só
    os eixos.
  - O fluxograma dos agentes rolava na horizontal a 1440 px. A coluna do pedido ficou mais
    estreita (22rem), e o fluxograma passou a caber a partir de 640 px, com a coluna do
    coordenador mais larga para o nome não se cortar.
  - Nas Experiências, a explicação do histórico passou para o "?" ao lado da contagem, e a
    comparação ficou em cartões, com o melhor de cada métrica marcado a branco.
- **Os testes de ponta a ponta** mudaram onde a interface mudou:
  - a entrada já não tem barra;
  - a troca de interface é um grupo de alternância (`radio`);
  - a ficha abre-se por separadores;
  - o tipo de cada coluna tem a razão no "?";
  - a tabela das métricas encontra-se pelo nome.

  Há testes novos: um ecrã de uma só interface traz a barra dela, e a pesquisa salta para um
  ecrã. Os 39 passam, e o axe passa em todos os ecrãs, como antes. Numa das corridas, os dois
  servidores do teste (a API e o `vite preview`) caíram a meio, sem mensagem. Nas seguintes,
  com o `PYTHONFAULTHANDLER` ligado (para um crash nativo deixar rasto), não voltou a
  acontecer. Se voltar, é por aí que se começa.
- **A auditoria final**, com a skill `web-design-guidelines` da Vercel, encontrou e corrigiu:
  - transições `all` nos botões, nos distintivos, nos interruptores e na barra de progresso
    (passaram a listar as propriedades);
  - a falta de um atalho "Saltar para o conteúdo", que agora é o primeiro Tab de cada ecrã;
  - o conteúdo dos separadores sem foco visível;
  - o espaço entre "Ctrl" e "K", que podia partir a linha;
  - os campos soltos sem `name` e sem `autocomplete="off"`;
  - as gavetas e a pesquisa sem `overscroll-behavior: contain`;
  - o nome da app sem `translate="no"` e o título das páginas sem `text-wrap: balance`.

  Ficaram de fora, de propósito: o "Title Case" e as aspas curvas inglesas (em pt-PT usa-se
  maiúscula só no início e «»); as datas com formato fixo (`dd/mm/aaaa hh:mm`, iguais em
  qualquer browser); e os separadores dos Dados, do Explorar e da ficha de cada modelo, que não
  passam para o endereço.

**Depois da 10.9, a página larga e os agentes vivos** (ADR 0034):

- **O espaço:**
  - as páginas largas ocupam o ecrã, com 2 cm de margem de cada lado;
  - o `Panel` passou a secção aberta (um filete por cima), e os números grandes a uma faixa
    (`StatStrip`);
  - a caixa ficou só para o que se escolhe, os avisos e os diálogos;
  - entre colunas, o mesmo espaço largo em todos os ecrãs.
- **O fluxograma dos agentes** segue o movimento de um vídeo de referência:
  - os nós entram coluna a coluna e as ligações desenham-se;
  - na ligação ativa corre uma luz com a cor do agente;
  - nas dos agentes que já trabalharam passa, de vez em quando, uma luz lenta;
  - as etiquetas em pílula mostram as linhas, o fold e o teste.

  Para ver o vídeo, tiraram-se fotogramas com o `ffmpeg`: o X não deixa ler a página sem
  conta. Os nós passaram a HTML (botões), com as colunas laterais de largura fixa.
- **O painel de cada agente:** clicar num nó abre-o ao lado, sem tapar o fluxograma, com o que o
  agente faz agora, os números do seu melhor, os modelos que experimenta, cada candidato (com a
  nota em cada fold) e o que disse. O agente de dados, o coordenador e o líder têm os seus. Usa
  só o que a procura já devolvia: o backend não mudou.
- **A conversa:** cada agente tem a sua cor, a mesma do nó, no ícone e num filete ao lado da
  mensagem.
- **A verificação:**
  - Para ver uma procura a decorrer sem tocar no histórico de quem usa o laboratório, correu-se
    uma API à parte, com os dados numa pasta temporária, e uma interface ligada a ela.
  - Os 56 testes da web e os 39 de ponta a ponta passam sem mudanças: os títulos e os papéis que
    procuram ficaram iguais.

**O histórico visual das trocas** (ADR 0035):

- **O backend:** cada mensagem da conversa diz o que é na troca (`kind`: relatório, anúncio,
  pedido, resposta, decisão ou teste) e a quem se dirige (`recipients`). O coordenador passou a
  deixar na conversa o pedido que faz a cada agente antes de cada candidato. Antes, só se via a
  resposta. Os dois enums novos têm nome no vocabulário, e o tema `exchange_history` explica como
  se lê o histórico.
- **A interface:** um interruptor por cima do fluxograma ("Ao vivo" e "Histórico das trocas")
  repete a procura mensagem a mensagem:
  - cada troca é uma luz que vai de quem fala para quem ouve, pelas ligações (ao contrário,
    numa resposta);
  - a decisão e o teste passam pelo agente do vencedor;
  - o fluxograma mostra a procura como estava nesse momento;
  - por baixo ficam o que se disse e uma linha do tempo, com uma marca por mensagem na cor de
    quem a disse e as fases por baixo. Arrasta-se como um cursor, com o rato ou o teclado;
  - a conversa acompanha o histórico, diz a quem cada mensagem se dirige e mostra os pedidos
    numa linha.
- **A verificação:**
  - Viu-se de ponta a ponta numa API à parte, com os dados numa pasta temporária: 60 mensagens
    numa procura de 24 candidatos, com os pedidos, as respostas, a decisão pelo agente dos
    modelos probabilísticos até ao vencedor, e o teste.
  - Os testes novos do Python verificam a troca (um pedido seguido da resposta por candidato,
    os anúncios para os agentes, a decisão para o vencedor, o fim da conversa). O teste de
    ponta a ponta dos agentes percorre o histórico até à decisão e ao teste.
  - O `pytest` encontrou uma instabilidade antiga em `tests/api/test_training.py`, fora destas
    mudanças: o treino que o teste pausa e para às vezes acaba antes de a paragem chegar
    (falhou 1 vez em 3).

**A barra pelos passos, o espaço e o botão de prosseguir** (ADR 0036):

- **A skill:** para a organização e o espaçamento instalou-se a `impeccable`
  (pbakaus/impeccable), só com o texto revisto (o `layout`, o `distill`, o `typeset`, o `polish`
  e o `quieter`); o lançador e o detetor ficaram de fora, por serem executáveis por rever. O
  método dela (avaliar, fixar uma tese espacial, agrupar pela proximidade) guiou o resto.
- **A barra lateral:**
  - os primeiros passos (os Dados e o Explorar, agora também na interface dos agentes) ficam
    em cima;
  - por baixo, a troca entre o Manual e os Agentes;
  - depois, os passos de cada interface, o "Aprender" e o "Histórico";
  - os passos estão ligados por um trilho, com um ponto por passo (aceso no aberto, cheio nos
    de trás) e uma luz que desce sem parar;
  - letra de 15 px e barra de 18rem.
- **O espaço:** os filetes de separação saíram, e o espaço passou a separar (uma escala de base
  4, com intervalos apertados dentro dos grupos e largos entre eles). Ficaram só as linhas com
  significado.
- **Prosseguir:** um botão próprio (`Proceed`: branco, maior, com a seta), no topo à direita e
  no fim de cada passo (`NextStep`), do Dados ao Explicar, na procura e no RL. Na procura
  terminada, treinar um candidato à mão passou para baixo do quadro, como ação dele.
- **Os gráficos:** no "O que aprendeu", a rede e a árvore ocupam a largura toda, os pesos, a
  importância e as tabelas ficam lado a lado, e um só gráfico de meia largura fica com uma
  largura de leitura.
- **Os testes:** os 56 da web e os 39 de ponta a ponta passam. Estes clicam no primeiro dos
  dois botões de prosseguir de cada página, e o da barra verifica que o Explorar está nas duas
  interfaces.

**As cores por família, a barra que desliza e a lição do reinforcement learning** (ADR 0037):

- **As cores por família:** cada família de modelos tem uma cor fixa (`ui/web/src/lib/families.ts`),
  a mesma nos Modelos, no fluxograma e na conversa dos agentes, nos gráficos da procura e na
  entrada. Antes, a cor de um agente vinha da posição dele no plano e mudava de procura para
  procura. Onze famílias e oito cores: as que só têm modelos não supervisionados repetem a cor
  de uma família que só tem supervisionados, e nunca se encontram no mesmo sítio. As duas
  sequências passaram no validador de paletas do skill de visualização de dados.
- **Os Modelos:** dois paradigmas (a aprendizagem supervisionada e a não supervisionada, com os
  temas novos `supervised` e `unsupervised`) e, em cada um, um cartão por família, na cor dela.
  Cada modelo tem um desenho pequeno, a mexer, do que faz, escolhido pela família e pelas
  tarefas. Com "reduzir movimento", os desenhos ficam parados e completos.
- **A barra lateral:** a secção aberta tem um cartão claro que desliza de uma secção para a
  outra (uma mola do Motion), e o ponto aceso corre pelo trilho. A pesquisa ficou só no
  `Ctrl K` e no cabeçalho do telemóvel.
- **Os Dados:** sem separadores. Os exemplos em cartões e, logo por baixo, uma zona grande para
  largar ou escolher o ficheiro; os dados sintéticos no fim.
- **O Explorar:** seis partes numeradas, todas à vista (o resumo, os avisos, as colunas, as
  correlações, os valores em falta e o alvo), com um índice preso ao topo que diz em que parte
  se está e salta para as outras.
- **O painel de cada agente:** 46rem, com o cabeçalho fixo (o estado do agente numa frase e uma
  fila para saltar para outro agente) e separadores (o resumo, os candidatos e o que disse).
  O `Sheet` do shadcn prendia todos os painéis laterais a 24rem com uma classe de variante; o
  dos agentes usa a variante certa.
- **A lição do RL:** um método novo, `ReinforcementService.lesson` (`GET /reinforcement/lesson`),
  treina um Q-learning no penhasco com uma seed fixa e devolve cinco episódios inteiros, passo
  a passo: o estado, a ação, se explorou, a recompensa, a surpresa e o valor novo da célula, com
  uma frase sobre cada episódio. Não é um run, e calcula-se uma vez. A interface anima-a no topo
  do ecrã: o agente na grelha, a recompensa de cada passo, o valor a espalhar-se a partir do
  objetivo, e o ciclo em quatro cartões (os temas novos `rl_state`, `rl_action`, `rl_reward` e
  `rl_update`). Para saber se uma ação foi ao acaso, o ε-greedy da base dos agentes passou a
  ter um `_decide`, com o mesmo uso do acaso: o treino não muda.
- **A verificação:**
  - Viu-se cada ecrã no browser, com capturas a meio das animações (o cartão da barra a
    deslizar, os desenhos dos modelos em dois momentos do ciclo, a lição em cada episódio).
  - Os testes novos do Python verificam a lição: sempre igual, os passos ligados uns aos outros
    e pagos pela célula onde acabam, o primeiro episódio a explorar em todos os passos, o último
    pelo caminho de 13 passos. O teste de ponta a ponta do caminho à mão usa o índice do
    Explorar.
  - Passam os 56 testes da web, os 39 de ponta a ponta e os 2059 do Python. O axe encontrou
    regiões com o mesmo nome nos Modelos (uma família aparece nos dois paradigmas): os cartões
    das famílias passaram a grupos.
  - O `pytest` encontrou outra instabilidade antiga, fora destas mudanças:
    `test_the_same_seed_gives_the_same_result` falha às vezes no K-means e no t-SNE (diferenças
    na nona casa decimal). Falha também no código anterior a estas mudanças, e passa sempre com
    um só fio (`OMP_NUM_THREADS=1`).

**O contorno colorido, sem números nem brilho, e menos texto à vista** (ADR 0038):

- **A cor é o contorno:** as bolas e as barras coloridas ao lado dos nomes saíram. A cor de uma
  família (ou de como acabou um episódio da lição) passou para o contorno do retângulo: inteiro
  quando está escolhido ou ativo, fraco quando não está. Assim nas fichas dos agentes, no
  fluxograma, na conversa, no painel de cada agente, nos cartões dos Modelos, na página de um
  modelo, no plano do agente de dados, na comparação de runs, na lição e na entrada. O
  `ChipGroup` aceita uma cor por opção.
- **Sem números:** o Explorar, os passos do Preparar e as fases da lição ficam só com os
  títulos (no Preparar, um filete separa os passos).
- **Sem brilho:** o botão de prosseguir destaca-se pelo tamanho e pelo branco, sem a sombra; os
  brilhos da barra, do coordenador e da caixa ativa do fluxograma também saíram.
- **Menos texto à vista:**
  - os subtítulos dos cabeçalhos saíram: o contexto de um run é o primeiro número da faixa
    ("Dados" ou "Ambiente", com a seed como detalhe do run), e no Explicar e na procura vai para
    o "?" da página;
  - o `NextStep` é só o botão;
  - os rótulos ficam curtos, com a explicação no "?";
  - cada métrica tem as tarefas e o sentido no "?";
  - a lição tem a frase de cada episódio e o texto de cada fase nos "?", e os controlos na fila
    do progresso;
  - os gráficos estreitos sem legenda têm o "Como se lê" no "?" do título do bloco.
- **O treino:** o cabeçalho tem dois botões, que só ficam ao lado do título se couberem os dois
  (o nome do modelo deixou de se partir em três linhas). "Experimentar outro modelo" passou para
  o fim, ao lado de "Explicar as previsões".
- **Os painéis laterais:** o "?" de cada página, o de um run e o de uma métrica passaram a ter a
  largura pedida (32rem, 32rem e 36rem), pela variante certa do `Sheet`.
- **A verificação:** viu-se cada ecrã no browser, a 1440 px. Passam os 56 testes da web e os 39
  de ponta a ponta. Três testes mudaram:
  - o dos modelos procura o painel de cada separador da ficha (o título repetido saiu);
  - o "?" ao lado de "Calcular a importância" chama-se "Quanto demora";
  - o separador "Classes em cada parte" do Preparar passou a "Classes", e o teste da troca de
    alvo verifica primeiro que ele existe.

  O backend não mudou.

### Preparação da fase 11: guardar e carregar o modelo

O pedido: uma secção para **guardar o que se preparou e o que os modelos aprenderam**, com uma
pré-visualização do ficheiro antes de ele existir e o que vai lá dentro. O ficheiro tem de abrir
nesta app (noutro computador) ou numa parecida, e tem de haver a opção de **carregar um modelo
já treinado**.

O ficheiro é o pacote do modelo (ADR 0039): tudo o que é preciso para perceber o modelo e para
voltar a prever com ele, sem a fonte dos dados. A fase 12 (Prever) vai usá-lo.

**O que vai no ficheiro.** Um `.mllab` é um ZIP com nomes fixos, descrito em
`docs/formato-mllab.md` para outras apps o saberem ler:

```text
titanic-regressao-logistica.mllab
├── README.md          a ficha do modelo, para pessoas
├── manifest.json      a identidade, as versões e a impressão digital de cada ficheiro
├── manifest.sig       a assinatura do manifesto (diz se foi feito nesta instalação)
├── input.json         as colunas que o modelo espera, como chegam
├── preparation.json   a receita da preparação, o que ela aprendeu no treino e a linhagem
├── run.json           o run: a configuração, as métricas, o que o modelo aprendeu, a loss
├── model.joblib       o modelo treinado, para este laboratório (opcional)
├── model.onnx         a preparação e o modelo, para outras apps (sempre que dá)
├── animation.json     o treino passo a passo, para o rever depois (opcional)
└── examples.csv       algumas linhas do teste, com o real e o previsto (opcional)
```

Um excerto das partes que não se veem noutros ecrãs (valores de exemplo):

- `manifest.json`:

  ```json
  {
    "format": "ml-lab-package",
    "format_version": 1,
    "package_id": "pkg-3f9a1c2e",
    "name": "Titanic — Regressão logística",
    "created_at": "2026-10-29T14:32:05Z",
    "run_id": "run-c36c0f26",
    "seed": 42,
    "dataset": { "name": "Titanic", "source": "sample", "rows": 891, "sha256": "9b1e…" },
    "task_type": "binary_classification",
    "target": "Survived",
    "model": { "name": "logistic_regression", "family": "linear", "implementation": "sklearn" },
    "primary_metric": { "name": "accuracy", "partition": "test", "value": 0.821 },
    "versions": { "ml_lab": "0.1.0", "python": "3.12", "numpy": "…", "scikit-learn": "…" },
    "files": [{ "path": "README.md", "bytes": 2140, "sha256": "41c0…" }]
  }
  ```

- `input.json`, uma entrada por coluna, tirada das linhas de treino antes da preparação:

  ```json
  {
    "target": { "name": "Survived", "classes": ["0", "1"] },
    "columns": [
      { "name": "Pclass", "kind": "numeric", "min": 1, "max": 3, "typical": 3, "can_be_missing": true },
      { "name": "Sex", "kind": "categorical", "categories": ["female", "male"], "typical": "male" },
      { "name": "Age", "kind": "numeric", "min": 0.42, "max": 80, "typical": 28, "missing_in_train": 142 }
    ],
    "left_out": ["PassengerId", "Name", "Ticket", "Cabin"]
  }
  ```

- `preparation.json`, cada transformação com os parâmetros (`params`) e o que aprendeu
  (`learned`), na ordem em que corre:

  ```json
  {
    "cleaning": [
      { "operation": "drop_columns", "params": { "columns": ["PassengerId", "Name", "Ticket", "Cabin"] } }
    ],
    "transforms": [
      { "operation": "impute", "params": { "strategy": "median" }, "learned": { "fill_values": { "Age": 28.0 } } },
      { "operation": "impute", "params": { "strategy": "most_frequent" }, "learned": { "fill_values": { "Embarked": "S" } } },
      { "operation": "one_hot", "learned": { "categories": { "Sex": ["female", "male"], "Embarked": ["C", "Q", "S"] } } },
      { "operation": "standard_scaler", "learned": { "means": { "Age": 29.65 }, "scales": { "Age": 12.71 } } }
    ],
    "features": ["Pclass", "Sex_female", "Sex_male", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S"]
  }
  ```

- `README.md`, a ficha do modelo, escrita pelo backend:
  - o que prevê e a partir de quê;
  - como foi feito: os dados, a divisão, a preparação numa linha por passo, o modelo e os
    hiperparâmetros;
  - como se portou: as métricas no treino e no teste, e a comparação com o palpite mais
    simples ("prever sempre «0» acertava 61,5%");
  - os limites: os intervalos que o modelo viu ("Age de 0,42 a 80: fora disso é extrapolação")
    e os avisos do diagnóstico;
  - o que vai dentro do pacote e como se usa: no ML Lab (carregar o ficheiro), noutra app (o
    `model.onnx`, com um exemplo em Python com o ONNX Runtime) e em Python com o ML Lab.

**O ecrã Guardar** (`/guardar`, e `/guardar/<run>` para um run concreto):

- **Como se chega lá:**
  - na barra, é o último passo das duas interfaces, depois do Explicar;
  - o Explicar passa a prosseguir para "Guardar o modelo";
  - o botão "Guardar" do treino leva aqui, com o run aberto (era o "Guardar experiência");
  - o Guardar prossegue para as Experiências (e, na fase 12, para o Prever).
- **Em cima, duas colunas** num ecrã largo (uma por cima da outra no telemóvel); em baixo, a
  biblioteca e o carregar:

  ```text
  Guardar (?)                                                  [ Guardar o modelo → ]

  O que guardar                       O ficheiro
  Run    [ Regressão logística · Titanic ▾ ]
                                      titanic-regressao-logistica.mllab
  Nome   [ Titanic — Regressão logística ]
                                      24 KB · 10 ficheiros · formato 1
  O modelo, para prever    (?) [■]
  Linhas de exemplo: 5     (?) [■]    README.md         2 KB  │ Titanic — Regressão logística
  O treino passo a passo   (?) [ ]    manifest.json     2 KB  │
                                      input.json        1 KB  │ Prevê Survived a partir de 7
  ⚠ As linhas de exemplo são dados    preparation.json  3 KB  │ colunas de um passageiro.
    reais do Titanic.                 run.json          9 KB  │
                                      model.onnx        2 KB  │ Como foi feito …
                                      …

  Os modelos guardados                          ┌───────────────────────────────────┐
  Titanic — Regressão logística   feito aqui    │  Carregar um modelo treinado       │
     24 KB · hoje, 14:32   Abrir  ↓  Apagar     │  Larga aqui um ficheiro .mllab     │
  Iris — Árvore de decisão        carregado     │  ou clica para o escolher          │
     31 KB · ontem         Abrir  ↓  Apagar     └───────────────────────────────────┘
  ```

  - **O que guardar:** o run (por omissão, o último; a lista tem os runs com o modelo à mão), o
    nome (dá o nome do ficheiro) e as partes opcionais, cada uma com o seu "?". Por baixo, os
    avisos do backend.
  - **O ficheiro:** o nome, o tamanho total, o número de ficheiros e a versão do formato. Por
    baixo, as partes do pacote numa lista, com o tamanho de cada uma. Ao escolher uma parte, o
    conteúdo aparece ao lado:
    - a ficha, desenhada como Markdown;
    - os JSON, formatados e com as secções a abrir e fechar;
    - o CSV, numa tabela;
    - os binários (o modelo e o ONNX), só com o tamanho e o que são.
  - **Os modelos guardados:** a biblioteca, com os pacotes feitos aqui e os carregados. Cada um
    tem a origem, o tamanho e a data; "Abrir" mostra o pacote no mesmo leitor da
    pré-visualização, com o botão para o abrir no Treinar e avaliar.
  - **Carregar um modelo treinado:** uma zona grande, como a do ficheiro nos Dados, onde se larga
    um `.mllab`.
- **Ao mudar uma opção**, a pré-visualização pede-se outra vez, depois de uma pausa curta (como
  no Preparar). Uma parte que entra ou sai da lista anima-se, e o tamanho total acompanha.
- **Guardar** grava o pacote na biblioteca, guarda o run no histórico (como o botão do treino
  fazia) e descarrega o ficheiro. O botão mostra que está a gravar e depois o visto; uma
  notificação diz o nome e o tamanho.
- **Carregar** lê o ficheiro e mostra-o logo no leitor, com a ficha e a avaliação:
  - **feito aqui:** o run entra no histórico e pode abrir-se já;
  - **de outra instalação:** aparece o aviso, antes de o modelo abrir:

    > Este modelo foi feito noutro ML Lab. O modelo vem num ficheiro joblib, e abri-lo corre o
    > código de quem o fez. Só o abras se confias em quem to deu. A ficha, a avaliação e a
    > preparação já estão à vista, e lê-las não tem risco.

    Com "Confio, abrir o modelo", o run entra no histórico. Sem confirmar, o pacote fica na
    biblioteca só para ler;
  - **estragado ou alterado** (as impressões digitais não conferem): lê-se o que der, e o modelo
    não abre;
  - **de um formato mais novo:** recusa-se, com a versão que falta.
- **Os estados:**
  - sem nenhum run, a parte de guardar diz como treinar um, e o carregar continua disponível;
  - com um run só registado (sem o modelo), um aviso leva às Experiências para o reproduzir;
  - um erro da API aparece por baixo do botão;
  - um modelo que não prevê linhas novas (o t-SNE) guarda-se, mas o aviso diz que o pacote é só
    para ler.
- **O que se aprende** fica nos "?", com os temas novos do backend:
  - `model_package`: o que é o pacote, e porque é que a preparação viaja com o modelo (uma
    linha nova tem de passar pelas mesmas transformações, com os valores aprendidos no treino);
  - `model_card`: o que deve dizer uma ficha de modelo;
  - `pickle_safety`: porque é que abrir um modelo de outra pessoa pede confirmação;
  - `onnx_format`: o que é o ONNX e porque é que outras apps o abrem;
  - `memorized_rows`: há modelos que guardam os próprios dados;
  - `package_versions`: porque se registam as versões das bibliotecas.

**Por dentro**, camada a camada, com as regras do `CONTRIBUTING.md`:

- **`core/data`** (o pandas fica aqui):
  - o `Transform` ganha `learned()` e `restore()`, em todas as transformações;
  - as limpezas guardam o que aprenderam (os limites dos extremos), e cada operação diz se vale
    para linhas novas;
  - uma `FittedPreparation` nova (`core/data/serving.py`): o perfil da entrada (das linhas de
    treino em bruto), as limpezas que valem para linhas novas e o pipeline aprendido. Sabe
    transformar uma tabela nova e passar a JSON e voltar;
  - o `Dataset` guarda-a: o `_assemble` monta-a em vez de deitar o pipeline fora;
  - as linhas de exemplo em CSV saem daqui, como texto.
- **`core/models`:** `ModelInfo.keeps_training_rows`, marcado nos modelos que guardam linhas do
  treino.
- **`core/export`** (novo; o ONNX só existe aqui):
  - o grafo da preparação, a partir dos valores aprendidos (imputação, one-hot, escala,
    intervalos, polinomiais);
  - um conversor por modelo, num registo: os do scikit-learn pelo `skl2onnx`, os didáticos e a
    MLP montados a partir do que aprenderam;
  - o grafo final junta os dois, e um teste de contrato compara as previsões do ONNX Runtime com
    as do laboratório.
- **`core/tracking`** (o joblib e os ficheiros ficam aqui):
  - o escritor do pacote: o ZIP, o SHA-256 de cada ficheiro e a assinatura, com a chave em
    `artifacts/package.key`;
  - o leitor: só os nomes conhecidos, com os tamanhos, as impressões digitais e a assinatura
    confirmados, sem extrair nada para o disco;
  - a biblioteca: `artifacts/packages/<id>/`, com o pacote e a origem dele (feito aqui ou
    carregado, e se já se confiou nele);
  - os temas do "?", em `core/tracking/explanations.py`.
- **`core/services`:**
  - um `PackageService` novo (`lab.packages`), com dados simples:
    - `candidates()`: os runs que se podem guardar, ou porque não;
    - `options(run_id)`: as partes opcionais e os valores por omissão;
    - `preview(run_id, request)`: as partes, os tamanhos, o conteúdo de cada uma e os avisos;
    - `save(run_id, request)`: grava o pacote, guarda o run no histórico e devolve o que ficou
      guardado;
    - `saved()`: a biblioteca;
    - `contents(package_id)`: um pacote da biblioteca, no mesmo formato da pré-visualização;
    - `load(filename, content)`: põe um `.mllab` na biblioteca; feito aqui, o run entra logo no
      histórico;
    - `trust(package_id)`: depois do aviso, abre o modelo de um pacote de fora e põe o run no
      histórico;
    - `delete(package_id)`;
    - `file(package_id)`, `@internal`: os bytes, para o descarregar;
  - o `HistoryService` ganha um método `@internal` para adotar um run carregado (a entrada, o
    documento, o modelo e o passo a passo);
  - a ficha (`README.md`) monta-se aqui, a partir do resultado do run;
  - os avisos, com código: `package_real_rows`, `package_memorized_rows`,
    `package_cannot_predict`, `package_without_onnx`, `package_large`,
    `package_without_model`, `package_from_elsewhere`, `package_damaged` e
    `package_other_versions`;
  - os tipos novos: `PackageCandidate`, `PackageOptions`, `PackageRequest`, `PackageFile`,
    `PackagePreview`, `SavedPackage`, e os enums `PackageFileKind` e `PackageOrigin` (com nome
    no vocabulário).
- **A API** (`api/routes/packages.py`):
  - uma rota por método: `GET /packages/candidates`, `GET /packages/runs/{run_id}/options`,
    `POST /packages/runs/{run_id}/preview`, `POST /packages/runs/{run_id}`, `GET /packages`,
    `GET /packages/{package_id}`, `POST /packages/uploads`, `POST /packages/{package_id}/trust`
    e `DELETE /packages/{package_id}`;
  - o descarregar, `GET /packages/{package_id}/file`: a primeira rota que não devolve JSON
    (`application/zip`, com o nome do ficheiro no `Content-Disposition`);
  - o `openapi.json` gerado outra vez, e o guia do contrato em `ui/web/README.md`.
- **A interface** (`ui/web/src/features/save/`):
  - o ecrã, a lista das partes, o leitor de cada uma, a biblioteca, o carregar e o aviso;
  - o ecrã novo em `screens.ts`, com o ícone `Package`;
  - o descarregar é uma ligação normal para a rota do ficheiro.

**Os passos:**

- [x] **11.1 A preparação que se leva** (`core/data`):
  - [x] `learned` e `restore` nas transformações, com um teste de contrato: a restaurada
    transforma igual à original;
  - [x] as limpezas com o que aprenderam e a marca das que valem para linhas novas;
  - [x] a `FittedPreparation`, guardada no `Dataset`;
  - [x] o JSON da preparação e o perfil da entrada;
  - [x] as linhas de exemplo em CSV.
- [x] **11.2 O pacote** (`core/tracking`): o escritor, o leitor com a verificação, a chave da
  instalação e a biblioteca.
- [x] **11.3 O serviço** (`core/services`):
  - [x] guardar: `candidates`, `options`, `preview`, `save`;
  - [x] carregar: `load`, `trust`, e o run adotado pelo histórico;
  - [x] a biblioteca: `saved`, `contents`, `delete`, `file`;
  - [x] a ficha do modelo, os avisos e a marca `keeps_training_rows`;
  - [x] os temas do "?" e o vocabulário.
- [x] **11.4 A API:** as rotas, o descarregar, o carregar, o `openapi.json` e o guia do
  contrato.
- [x] **11.5 O ecrã Guardar:**
  - [x] a rota e a barra;
  - [x] as duas colunas e a pré-visualização ao vivo;
  - [x] a biblioteca e o leitor de um pacote;
  - [x] o carregar, com o aviso;
  - [x] o Explicar e o botão do treino a levar ao Guardar;
  - [x] os estados.
- [x] **11.6 O ONNX** (`core/export`): as dependências, o grafo da preparação, os conversores
  dos modelos, a regra de arquitetura e o teste de contrato.
- [x] **11.7 Os testes e o fecho:**
  - [x] o pacote do Titanic tem as partes todas, e o manifesto confere;
  - [x] cada documento pré-visualizado é igual ao que fica no ZIP;
  - [x] nos 34 modelos (testes de contrato), o pacote faz-se; nos que sabem prever linhas
    novas, a preparação e o modelo lidos do pacote dão as mesmas previsões (ou projeções) do
    run;
  - [x] carregar noutra instalação (outra pasta de dados) pede confirmação; depois dela, o run
    abre no histórico com o modelo;
  - [x] um pacote com um byte mudado nunca abre o modelo;
  - [x] num ficheiro carregado, as linhas de exemplo vêm desligadas;
  - [x] os testes da fronteira, das rotas e do vocabulário passam;
  - [x] o teste de ponta a ponta do ecrã: mudar uma opção, ver a ficha, guardar, descarregar
    (com os cabeçalhos certos), encontrar o pacote na biblioteca e carregá-lo outra vez, com o
    axe;
  - [x] o formato descrito em `docs/formato-mllab.md`, a ADR 0039 aceite, as notas da fase 11 e
    o ✅ no roadmap.

**Pronto quando:** no Titanic, depois de treinar, o ecrã Guardar mostra o ficheiro antes de o
gravar (cada parte, o tamanho e o conteúdo). Guardá-lo descarrega um `.mllab` que abre em
qualquer programa de ZIP. Carregado noutra instalação, mostra o aviso e, depois de confirmado, o
run abre no Treinar e avaliar. O modelo de dentro dá as mesmas previsões do run original, no
laboratório e no ONNX Runtime.

**Fora desta fase:**

- prever com um pacote, e com um modelo ONNX feito noutra app: a fase 12;
- os pacotes dos agentes de reinforcement learning.

**Decidido** (as respostas ao plano):

1. A secção chama-se "Guardar", e o ficheiro, `.mllab`. Abre nesta app e noutras parecidas (o
   formato documentado e o ONNX), e há a opção de carregar um modelo já treinado.
2. As linhas de exemplo vêm ligadas nos datasets de exemplo e nos sintéticos, e desligadas nos
   ficheiros carregados.
3. Um pacote de outra instalação abre depois de um aviso.
4. O botão de guardar do treino leva ao ecrã Guardar, e guardar lá também guarda o run no
   histórico.

### Fase 11

Decisões tomadas ao implementar (a principal tem ADR própria, a 0039):

- **A preparação viaja em JSON, não em pickle.** Cada transformação diz o que aprendeu
  (`learned`) e volta a montar-se a partir disso (`restore`); um teste de contrato confirma, em
  todas, que a restaurada transforma igual à original, e outro passa todos os datasets de
  exemplo e geradores pelo caminho do pacote. Assim só o modelo é pickle, e o pacote sem o
  modelo pode partilhar-se à vontade.
- **A pré-visualização é o ficheiro.** O mesmo código monta a pré-visualização e o ZIP, e um
  teste compara-os byte a byte. Guardar também guarda o run no histórico, como fazia o botão do
  treino.
- **A confiança é por instalação.** O manifesto lista o SHA-256 de cada ficheiro e é assinado
  com HMAC-SHA256 pela chave da instalação: um pacote feito aqui abre logo; um de fora abre a
  ficha, a avaliação e a preparação, e o modelo só depois do aviso; um estragado nunca abre o
  modelo. Uma entrada do ZIP com os bytes estragados (o CRC não confere) lê-se como estragada,
  em vez de dar um erro 500.
- **As partes opcionais vêm do backend** (`PackageOptions.choices`): o título, o que levam, se
  vão por omissão e a razão quando o run não as tem. O `key` de cada uma é o campo do pedido que
  a liga, e o ecrã desenha-se da lista sem saber quais são.
- **O ONNX é confirmado antes de entrar no pacote.** Corre com as linhas de exemplo no ONNX
  Runtime e tem de dar as mesmas previsões do laboratório; se não der, fica de fora com a razão.
  A preparação vai à frente do modelo, em `double` (as mesmas contas do laboratório): uma
  entrada por coluna, os números com `NaN` onde faltam, as categorias em texto com o vazio onde
  faltam. Os modelos do scikit-learn passam pelo skl2onnx em `double` quando ele e o ONNX
  Runtime o aceitam, e em `float` nos outros (as árvores do scikit-learn já comparam em
  `float32`); os didáticos, as redes e os que memorizam linhas montam-se do que aprenderam, com
  as mesmas fórmulas do NumPy. A conversão fica guardada por run, porque a pré-visualização se
  pede a cada opção que muda.
- **O PyTorch não sai das redes.** A MLP dá as camadas densas em NumPy (`dense_layers`), e os
  modelos que memorizam linhas dão-nas (`memory`): o `core/export` monta o grafo sem saber como
  cada um as guarda. O ONNX, o skl2onnx e o ONNX Runtime só existem em `core/export`, com uma
  regra de arquitetura.
- **O aviso do skl2onnx sobre o SVM.** O conversor do SVC ainda lê `probA_` e `probB_`, que o
  scikit-learn 1.9 deu como obsoletos (o nosso SVM calibra à parte e não os usa). O aviso é do
  skl2onnx, não do laboratório: ignora-se só esse, só durante a conversão, e a confirmação das
  previsões garante o resultado.
- **Ainda não passam para ONNX:** as datas (o formato não tem um tipo de data) e o t-SNE (não
  coloca pontos novos). O pacote vai na mesma, e a ficha diz porquê.
- **A mesma seed dá o mesmo k-means e o mesmo t-SNE** (ADR 0040): ao fechar a fase, o teste de
  contrato da seed falhava ao acaso, porque o scikit-learn soma em paralelo numa ordem que muda.
  Esses dois modelos passaram a treinar com uma só thread.
- **O ecrã:** o aviso de um pacote de fora e o leitor abrem um de cada vez (dois modais abertos
  ao mesmo tempo escondiam-se um ao outro dos leitores de ecrã); no teste de ponta a ponta, os
  ficheiros vão pelo conteúdo, porque o Chromium perde um caminho com acentos.

### Fase 13: agentes de IA

O pedido: agentes de IA, com um modelo de linguagem no próprio computador (o Ollama), que se
vejam a pensar, ao lado dos agentes de regras da fase 9 (que ficam como estão). A decisão e as
regras estão na ADR 0042: **a IA propõe e explica; as regras decidem e medem**.

**As etapas**, cada uma com o seu "Pronto quando":

| Etapa | O quê | Pronto quando | Estado |
|---|---|---|---|
| 13.1 — A base | `core/llm/`: a interface `LanguageModel` (a conversa aos bocados, o raciocínio à parte, a saída num JSON Schema e as ferramentas), o cliente do Ollama por HTTP e o `ScriptedModel` para os testes; as definições `ML_LAB_OLLAMA_URL` e `ML_LAB_OLLAMA_MODEL`; `LanguageModelService.status()` e `GET /llm/status`; duas regras de arquitetura | Sem o Ollama, `GET /llm/status` diz como o ligar; com ele, diz se tem o modelo; o cliente passa os testes contra um Ollama de mentira, e o teste com o Ollama a sério passa quando ele está a correr | ✅ |
| 13.2 — O agente de IA | Um agente que propõe hiperparâmetros com o modelo de linguagem, com a forma do `ModelAgent`; o pensamento ao vivo no trabalho em curso e uma mensagem "Pensamento" na conversa; as propostas validadas pelo espaço de procura; as respostas guardadas para reproduzir; no ecrã Agentes, o interruptor, o nó da IA no fluxograma e a comparação com a procura aleatória | No Titanic, com o Ollama aberto, vê-se o agente de IA a pensar antes de cada proposta, e as propostas medem-se como as outras, sem tocar no teste | ✅ (com respostas guardadas; o raciocínio ao vivo, com o Ollama a sério, só na 13.3, depois de corrigido o WebSocket) |
| 13.3 — O analista | Um agente com ferramentas (o quadro, um candidato, comparar dois, o plano dos dados) que responde a perguntas sobre a procura; o ciclo pensar → usar uma ferramenta → ler o resultado → responder, à vista | O analista explica porque ganhou o vencedor, e vê-se cada ferramenta que usou | ✅ (com o `qwen3:8b` a sério, no ecrã e no teste `ollama`) |

Decisões tomadas na etapa 13.1:

- **Sem dependências novas.** O Ollama fala HTTP com NDJSON (um objeto JSON por linha), e a
  biblioteca padrão chega (`urllib`). Assim o código mostra exatamente o pedido que vai para o
  modelo, e é o único pedido à rede do núcleo (regra de arquitetura).
- **O estado não é um erro.** O Ollama não estar a correr é normal: `status()` devolve-o com o
  que fazer (`ollama_not_running`, `model_not_installed`), e a API responde 200. Os erros
  (`LanguageModelError`) ficam para quando se pede uma conversa e ela falha.
- **A temperatura vai sempre a 0** e a seed vai no pedido: a mesma pergunta tende a dar a mesma
  resposta. Não é garantido entre computadores, por isso a etapa 13.2 guarda as respostas.
- **Os testes não precisam do Ollama.** O cliente testa-se contra um servidor HTTP de mentira
  (o pedido, os pedaços, as ferramentas, os erros); os agentes vão usar o `ScriptedModel`. Só o
  teste marcado `ollama` fala com o modelo a sério, e salta-se sem ele
  (`uv run pytest -m ollama`).

Decisões tomadas na etapa 13.2:

- **Um agente a mais, não um agente diferente.** O agente de IA (`core/automl/ai_agent.py`) entra
  na roda da afinação depois dos agentes de família, e na sua vez escolhe um dos finalistas e os
  valores. Os agentes de família continuam a sortear: a IA e a procura aleatória competem na
  mesma procura, com o mesmo orçamento, e a leitura do fim (`ai_comparison`) compara a melhor
  proposta dele com o melhor sorteio da afinação.
- **O espaço de procura valida as propostas.** `Dimension.problem` e `SearchSpace.problems` dizem
  porque é que um valor não serve; `SearchSpace.normalized` passa o `4.0` de um modelo de
  linguagem a `4`. Os agentes de família guardam tudo o que se mediu (`has_tried`), dos dois
  lados: nem a IA repete um sorteio, nem um sorteio repete a IA.
- **Uma recusa não gasta orçamento**, e diz-se porquê na conversa. À terceira seguida, ou se o
  modelo de linguagem falhar a meio, o agente de IA sai e a procura continua sem ele.
- **O pensamento vai ao vivo para o estado da procura**: `SearchLog.start_thinking`, `think` e
  `stop_thinking` guardam o raciocínio que vai chegando, e `AutoMLUpdate.thought` leva-o à
  interface pelo WebSocket. No fim, o raciocínio inteiro fica numa mensagem `THOUGHT`.
- **O que é novo no vocabulário:** `AgentRole.AI`, `ParticipantKind.AI` e
  `MessageKind.THOUGHT`; no candidato, `ai` e `reason`; no plano, `ai_agent` e `ai_model`.
- **No ecrã:** o interruptor "Chamar o agente de IA" pergunta o estado ao `GET /llm/status` (e
  mostra os avisos dele quando o Ollama não está pronto); o nó tracejado no fluxograma, o
  pensamento ao vivo e fechado na conversa, a marca **IA** no quadro e o painel do agente de IA
  com as propostas e a razão de cada uma. O tema "Porquê?" `ai_agent` explica-o.
- **Verificado sem o Ollama:** a procura com respostas guardadas passa pelo mesmo caminho (é o
  `ScriptedModel` a responder). Com o Ollama a sério, falta ver o raciocínio a crescer ao vivo
  (o teste marcado `ollama` cobre o cliente, não o ecrã).

Decisões tomadas na etapa 13.3 (a principal tem ADR própria, a 0053):

- **Um agente com ferramentas, num ciclo à vista.** O analista (`core/automl/analyst.py`) começa
  só com a pergunta: pensa, pede uma ferramenta, lê o que ela devolveu e volta a pensar, até
  responder. Cada passo fica no `AnalystLog`: o raciocínio, cada ferramenta (os argumentos, um
  resumo e o JSON que o modelo leu) e a resposta. Ao fim de seis voltas com ferramentas, a última
  vai sem elas e com o pedido de responder.
- **As ferramentas leem, não medem** (`core/automl/analyst_tools.py`): o quadro, um candidato,
  dois lado a lado e o plano dos dados leem um `SearchFacts`, que o serviço monta do resultado
  (`search_facts`). Do teste, só a nota do vencedor. Uma regra de arquitetura nova confirma que os
  ficheiros do analista não chegam aos modelos, aos dados, à avaliação, ao treino nem ao
  coordenador. Um pedido que não serve volta ao modelo com a razão, e ele corrige-se.
- **O `qwen3:8b` a sério desenhou as ferramentas.** A primeira versão punha tudo no quadro, e o
  modelo, numa procura do Titanic: atribuiu ao vencedor os números da referência; fez contas
  ("0.2", "1.1") que ninguém pediu; comparou o teste do vencedor com o "teste" da referência, que
  não existe; e respondeu só com o quadro. Cada erro deu uma regra:
  - as contas e as leituras vêm das ferramentas, escritas como o ecrã as mostra (a distância, o
    empate na prática com a regra do aviso `close_race`, o desempate, os folds ganhos e o "decora
    o treino" do `Objective`);
  - o quadro é um índice: o vencedor à parte, quanto cada candidato fica atrás do primeiro, e as
    chamadas seguintes já escritas (`compare_candidates(first=3, second=10)`); o treino, o empate
    e os folds ficam nas ferramentas dos pormenores;
  - quem não foi ao teste diz que não foi.

  Com isto, na pergunta principal, o modelo passou a ler o quadro e a comparar o vencedor com o
  segundo, e todos os números da resposta vieram das ferramentas.
- **A resposta passa por uma verificação**: cada número dela tem de estar no que as ferramentas
  devolveram (ou na pergunta). Os que não estão vão para o aviso `unsupported_numbers`; quando estão
  todos, `numbers_checked`. Também há `no_tools_used`, `forced_answer`, `empty_answer` e
  `stopped_early`. A verificação só olha para os números: uma frase errada com números certos
  passa, e é o registo passo a passo que deixa ver de onde veio cada coisa.
- **As respostas guardam-se**: o `RecordedReply` ganha `tool_calls`, e a pergunta
  (`AnalystRequest.replies`) repete uma resposta tal e qual com o `ScriptedModel`, sem o Ollama. Os
  testes do Python, da API e de ponta a ponta perguntam assim; o teste `ollama` pergunta ao modelo
  a sério e depois repete a resposta com o pedido dela.
- **Em segundo plano, como as procuras**: `AutoMLService.ask` lança a pergunta num `TrainingJob`
  próprio, e `question_job`, `question_updates` (e o WebSocket), `question_result` e
  `cancel_question` seguem-na pelo id; `questions` dá as respondidas de uma procura e
  `analyst_options` as perguntas sugeridas (uma regra do domínio: a do agente de IA só aparece se
  ele entrou) e as ferramentas. Sem pausa: parar fecha a ligação ao Ollama, a meio da resposta.
- **O contexto do Ollama vai a 8192 tokens em todos os pedidos** (`num_ctx`): o Ollama carrega os
  modelos com 4096, e a conversa do analista passa disso (e aí o Ollama corta o princípio sem
  avisar). O mesmo valor para o agente de IA, para o modelo não voltar a carregar ao alternarem.
- **O raciocínio ao vivo da 13.2 não chegava ao ecrã.** O WebSocket só mandava uma atualização
  quando havia passos ou mensagens novas, ou mudava o estado, o candidato ou a fase; o raciocínio
  cresce sem nada disso, por isso só aparecia no fim, já fechado na mensagem "Thought". O `thought`
  da procura e o `live` da pergunta passaram a contar (`api/stream.py`, com um teste).
- **No ecrã Agents**, o painel "Ask the analyst" fica logo a seguir ao vencedor: as perguntas
  sugeridas, as ferramentas e, para cada pergunta, a linha do tempo dos passos (o raciocínio ao
  vivo, cada ferramenta com "What it returned", a resposta e a leitura dela). Sem o Ollama mostra o
  estado de `GET /llm/status`, e as respostas que já houver ficam à vista. A pergunta a decorrer
  fica na sessão do cliente (`analystQuestion`): sair do ecrã e voltar segue-a outra vez. Os
  testes de ponta a ponta apontam o Ollama para uma porta fechada; para perguntar, a página finge-o
  pronto e o pedido leva as respostas guardadas.
- **Verificado com o Ollama a sério** (`qwen3:8b`, numa RTX 4060 de portátil): no ecrã, o raciocínio
  cresce ao vivo e, na pergunta "Why did the winner win?", o analista leu o quadro, comparou o #3
  com o #10 e respondeu em cerca de um minuto, com todos os números verificados. O raciocínio do
  agente de IA da 13.2 também cresce agora no ecrã, numa procura do Titanic: com isto, a fase 13
  cumpre o "Pronto quando".

### Orbyte, o site em inglês e a revisão do desenho

Entre as etapas 13.2 e 13.3 vieram dois pedidos sobre a interface:

1. **O nome e a língua (ADR 0043).** O projeto passou a chamar-se Orbyte, o site passa todo para
   inglês, a entrada ficou mais simples (com uma animação para cada interface), a barra lateral
   ficou mais larga, com o Histórico no rodapé, e a conversa dos agentes abre-se em ecrã
   inteiro, como um chat, com a linha do tempo das trocas ao lado.
2. **A revisão do desenho (ADR 0044).** Um só botão de prosseguir por ecrã, à direita do título;
   a escala de letra um degrau acima; a barra alinhada por duas verticais (e certa quando está
   recolhida); as larguras acertadas para a barra nova e as margens de 3 cm.

A análise ao código desta revisão encontrou e corrigiu:

- **A barra recolhida:** os ícones ficavam encostados à direita e cortados pela borda, porque as
  margens novas não tinham versão recolhida; o nome de cada item empurrava o ícone para a
  esquerda do botão.
- **A conversa em ecrã inteiro:**
  - as setas da linha do tempo andavam para trás a partir do segundo toque: contavam da primeira
    mensagem à vista, e o salto deixa a mensagem a um terço do topo, com outras por cima;
  - clicar outra vez no mesmo agente em "Who spoke" voltava à mesma mensagem, pela mesma razão;
  - ao vivo, uma mensagem que chegasse a meio do deslizar até ao fim podia desligar o seguir;
    agora só quem lê, ao subir, deixa de seguir.
- **O fluxograma dos agentes** ficava cortado num portátil (o nó do teste fora do ecrã).
- **As etiquetas dos gráficos** em SVG cresciam com a escala nova, sem as margens crescerem.
- **O salto para o lado** entre uma página curta e uma comprida (a barra de deslocamento).
- **Dois textos esquecidos em português** (o aviso de dados carregados e a dica do mapa de
  nulos), e o ponto da API, na entrada, verde enquanto ainda ligava.

A tradução do backend ficou a meio nesta revisão, e acabou no pedido seguinte.

### A tradução acabada, a identidade visual e a história do treino

O pedido seguinte fechou o que faltava e trouxe uma parte nova no Explicar:

1. **A tradução do backend acabou.** Tudo o que aparece no site está em inglês, venha da
   interface ou do backend: os nomes dos modelos e dos hiperparâmetros, os temas e as
   explicações, os avisos, a conversa dos agentes, as 25 fichas e os números do `formatting.py`
   (ponto decimal e vírgula nos milhares, como o `format.ts`). Fica em português o que só lê quem
   programa ou opera o laboratório: os registos, as docstrings, os comentários, a documentação,
   as ADRs, as descrições do `/docs` da API e as mensagens do `config.py` e do `logging.py`.
   - **Mudou com o texto o que também era chave:** as secções das fichas (`## Intuition`,
     `## Essential maths`…), que os testes de contrato procuram pelo nome, e a coluna das
     previsões no `examples.csv` do pacote, agora `predicted` (`docs/formato-mllab.md`).
   - **Não mudaram os identificadores:** os ids dos mapas do reinforcement learning (`simples`,
     `penhasco`, `labirinto`) ficam, porque os runs registados os guardam; o nome à vista vem de
     `MAP_LABELS`.
2. **O cliente gera-se só no `dev`, no `build` e no `npm run check` (ADR 0046).**
3. **O código carrega-se por ecrã:** o `autoCodeSplitting` do plugin do TanStack Router dá um
   pedaço a cada rota; as bibliotecas ficam em pedaços próprios (`react`, `tanstack`, `ui`), que
   o browser guarda entre versões; e o "?" de cada página (com o Markdown e o KaTeX) carrega-se
   à parte, com `lazy`. Antes, a entrada descarregava 700 kB de fórmulas antes de se mostrar.
4. **As conversas longas:** cada mensagem dos agentes (e cada linha do quadro das trocas) leva a
   utilidade `lazy-paint` (`content-visibility: auto`). Fora de vista não se desenha: uma
   conversa com centenas de mensagens desenha só as que estão à vista.
5. **O estado do Ollama** pergunta-se outra vez a cada 10 segundos: abrir o Ollama com o ecrã dos
   agentes aberto liga o interruptor, sem recarregar a página.
6. **A identidade visual (ADR 0045):** a Unbounded nos títulos, os rótulos em sans e sem
   maiúsculas, os botões sem setas e a barra a 18rem.
7. **A história do treino no Explicar (ADR 0047):** `HistoryService.story` e a rota
   `GET /history/runs/{run_id}/story` contam o run em seis capítulos (com que dados aprendeu,
   como aprendeu, o que aprendeu, como se saiu, onde se engana e o que fazer a seguir), cada um
   com um texto curto e um visual: a divisão das linhas, a curva da loss, umas barras ou a
   matriz de confusão. Conta-se só com o resultado registado, por isso aparece em qualquer run,
   mesmo nos do histórico que já não estão em memória.

A história leu-se em runs reais de cada tarefa (regressão logística, árvore de decisão, MLP, KNN,
regressão linear, k-means, PCA, gradient boosting, Isolation Forest e LOF), e foi assim que se
acertaram as frases que não se percebiam. A leitura das anomalias encontrou um buraco que não era
da história: um detetor que não marcava nada no teste, e por isso falhava as anomalias
verdadeiras, não levava aviso nenhum, e o último capítulo dizia que nada tinha corrido mal. O
avaliador ganhou dois avisos, como os `below_baseline` das outras tarefas:

- `nothing_flagged`: não marcou nada e a referência diz que havia anomalias (sem referência fica
  uma nota, porque não se sabe se havia o que marcar; sem anomalias no teste, não marcar nada é
  acertar);
- `no_anomaly_caught`: marcou linhas, mas nenhuma era uma anomalia verdadeira.

Sem nada marcado não há precisão, por isso o visual de "How well it did" passa a comparar a
fração que marcou com a fração de anomalias da referência.

**A varredura das resoluções.** Os 14 ecrãs abriram-se a 375, 768, 1024, 1280, 1366, 1440, 1920
e 2560 px, cada um num iframe da mesma página, a medir três coisas: se a página desliza para o
lado, que elementos passam da borda e que texto fica cortado sem reticências. Encontrou e
corrigiu:

- **o tablet ao alto ficava com 243 px de conteúdo** (a barra fixa e os 3 cm de cada lado, num
  ecrã de 768): a barra passa a gaveta abaixo de 1024 px, e os 3 cm valem só a partir de 1280 px
  (ADR 0045). A 768, o conteúdo passou a ter 694 px;
- **o texto dos seletores cortava-se sem reticências** (o run no Guardar, a escala no Preparar):
  o valor era um contentor `flex`, onde as reticências não funcionam; passa a bloco com
  `truncate`, e no Guardar um espaço separa a data do nome;
- **o último nome de coluna do mapa de nulos passava 11 px da borda** no telemóvel: a margem
  direita mede-se agora pelo rótulo inclinado (`slantedOverhang`, em `charts/margins.ts`);
- **o fluxograma dos agentes rolava a 1280 px**, com o teste fora do ecrã: encolhe até 0,8, com
  24 px entre colunas.

Uma lição da própria medição: num separador escondido (ou com a janela tapada), o browser não
entrega os `ResizeObserver`. Os gráficos ficam com largura 0 e o fluxograma não encolhe, e a
primeira volta da varredura deu tudo certo por isso. Tem de correr com a página à vista.

**Verificado:** os testes do Python (com os da história, em
`tests/unit/core/services/test_story.py`), o `npm run check` e o ponta a ponta (43 testes,
reescritos para os textos e as rotas em inglês).

### Os modelos por problema, a letra maior e o movimento

O pedido seguinte foi de acabamento (ADR 0048):

1. **Os modelos arrumam-se pelo problema que resolvem.** O backend ganha o enum `Problem` e o
   `TaskInfo.problem` (as duas classificações são o mesmo problema; é uma regra do domínio, por
   isso não fica na interface). O ecrã Modelos mostra o paradigma e, dentro dele, cada problema
   com a sua cor e os modelos em cartões. A cor tem sentido: fria nos supervisionados, quente nos
   não supervisionados (`lib/problems.ts`). Os agentes continuam com a cor da família.
2. **O "suggested to start" saiu da lista**, e o **"API online" saiu da barra e da entrada**: só a
   falha se mostra, com o que fazer.
3. **O botão de recolher a barra** desceu para o fundo, por cima do Histórico.
4. **A letra subiu mais um degrau** (15 a 30 px, e o título entre 36 e 44 px), também nos
   gráficos (1 px), e a barra passou a 19,5rem.
5. **O movimento ganhou uma curva com peso** (`--ease-out-expo`), a omissão do Tailwind e do
   Motion; os menus entram devagar e saem depressa, e cada página entra em cascata.
6. **O Explicar arruma-se em linhas:** cada capítulo da história é uma linha entre filetes, com o
   visual à altura do título, e as três perguntas das previsões seguem o mesmo desenho.

Uma lição dos testes: o axe mede o contraste do texto como ele está no momento. Com as páginas a
acender, media o texto ainda meio transparente e dava falhas de contraste que ninguém vê. Os
testes de acessibilidade esperam agora que as animações de entrada acabem
(`animationsSettled`, em `e2e/helpers.ts`).

### Os agentes com todos os modelos, a conversa por palavras e a barra com os ícones fixos

O pedido seguinte voltou aos agentes (ADR 0049):

1. **Entram todos os modelos da tarefa**, as implementações didáticas incluídas; ficam de fora só
   os que não se podem medir com o objetivo (sem probabilidades, para a ROC AUC e a log-loss).
2. **O pedido escolhe os modelos** (`AutoMLRequest.models`), e o ecrã Agentes arruma-se à volta
   disso: o pedido à esquerda, os modelos por família à direita (com as caixas da família a meio
   quando só alguns estão escolhidos), e por baixo o fluxograma e o plano. "Ask the agents" passou
   para o cabeçalho.
3. **A conversa diz por palavras o que se faz.** O pedido do coordenador diz porquê; a resposta do
   agente diz que hiperparâmetros mudou (com o valor de omissão ao lado), o que o resultado quer
   dizer (`Objective.reading`), se a mudança ajudou e se o modelo decora o treino
   (`Objective.overfits`). Os empates à vista dizem-se empates, e a profundidade 0 lê-se "no
   limit" (`x-labels`).
4. **A barra lateral cresceu** (21rem, a letra a 19 px, os ícones a 22 px), e **os ícones ficam no
   mesmo sítio** com a barra aberta ou recolhida: o centro a 35 px da borda e à mesma altura. O
   trilho dos passos passa agora pelos próprios ícones.

Mediu-se a posição de cada ícone com a barra aberta e recolhida: iguais ao píxel. Antes, um ícone
centrado só na barra recolhida saltava a meio da animação, porque o `justify-content` muda de uma
vez e a largura não.

### A barra que desliza, a página de erro, a troca e uma lição por agente

O pedido seguinte (ADR 0050):

1. **A barra ficou simples:** sem o trilho dos passos (nem a luz que o percorria), só os itens e o
   cartão do ecrã aberto. O trilho passou primeiro para junto da borda, e mesmo aí lia-se como um
   defeito.
2. **Abrir e fechar a barra reorganiza a página uma só vez.** O lugar da barra muda de uma vez, e
   o `AppFrame` faz o conteúdo deslizar de onde estava até lá (FLIP, com a Web Animations API).
   Mediu-se: uma reorganização em vez de uma por fotograma. Os nomes apagam-se e acendem com uma
   pequena espera, e a troca de interface passa de uma forma à outra a desvanecer.
3. **A página de erro é um gráfico que cai**, com o nome do que caiu, a mensagem e "Try again";
   o mesmo desenho serve o 404.
4. **A troca de interface leva ao passo certo** (`landingOf`): com os dados abertos, do Preparar
   vai-se para os Agentes, e não outra vez para os Dados.
5. **A lição do reinforcement learning tem um agente à escolha**: o Q-learning e o DQN. O backend
   grava o treino com um gravador que se junta a qualquer agente episódico (`core/rl/lesson.py`).

Uma lição dos testes: o ecrã do RL passou a ter dois botões "DQN (Deep Q-Network)", o da lição e
o do treino, e o teste de pausar o treino falhava por não saber qual carregar. Os testes procuram
agora o botão dentro do grupo dele.

### A órbita, o Parar, o Laboratório que se explica, o Guardar em passos e os dados de verdade

O pedido seguinte (ADR 0051):

1. **O logótipo é só a órbita**: a elipse num traço, com um satélite e uma falha à volta dele,
   numa só cor; o mesmo desenho no favicon.
2. **O Stop dos agentes para tudo**: não mede mais, não treina o vencedor, não vai ao teste nem
   ao histórico, e o ecrã volta ao pedido. A Pausa continua a interromper sem desistir.
3. **O Models abre com todas as tarefas**; o "Choose the model" do Preparar leva-o já com a
   tarefa preparada (`?task=…`).
4. **O Laboratório tem um guia em cada demonstração** (`DemoGuide`: a ideia, o que experimentar
   e o que fica), à vista, e cada gráfico tem nome. Duas demonstrações novas: **o limiar de
   decisão** (uma regressão logística em dados desequilibrados; cada linha do teste é um ponto,
   e a matriz de confusão e as métricas mudam com o limiar) e **a quantidade de dados** (uma
   árvore sem limite contra uma regressão logística, de 10 a 750 linhas, cinco amostras por
   quantidade).
5. **O Guardar faz-se em três passos numerados** (o run e o nome, o que vai no ficheiro, o
   ficheiro), com o resumo do pacote ao lado; a biblioteca é uma lista, com a faixa de carregar
   por cima.
6. **Os dados de verdade.** Dois ficheiros gerados com os problemas de sempre (151 500 linhas de
   churn em CSV com `;`, Windows-1252, vírgula decimal, datas e duplicados; 40 000 imóveis em
   Excel e em JSON Lines, com preços impossíveis) passaram o percurso todo pelos serviços (do
   carregar aos agentes, a medir cada passo) e pela interface. Nada falhou; o que se corrigiu:
   - o overfitting conta também o erro, descontado o ruído do teste (`memorised`): 100% contra
     92,6% num teste de 30 mil linhas é decorar; 97,5% contra 90% nas 30 do Iris, não;
   - 484 cidades repetidas são "categorias a mais", não "texto livre"; um código com 95% de
     valores diferentes é um identificador, mesmo com linhas repetidas;
   - os erros grosseiros do alvo (dez IQR para lá dos quartis, até 1% das linhas) sugerem-se
     para sair: nos imóveis, o R² passou de ≈ 0 para 0,82;
   - o mínimo, o máximo e os valores muito afastados do perfil vêm da coluna inteira, não da
     amostra de 10 000 linhas;
   - o JSON entra: registos, JSON Lines, os registos numa chave e as colunas, também em `.gz`.
7. **Para o GitHub**: o workflow de CI (`.github/workflows/ci.yml`), o `* text=auto` no
   `.gitattributes`, o README refeito e o último texto em português do site traduzido.

Uma lição dos testes: a regra nova do overfitting, só com a diferença e o erro, apanhava o Iris
(3 erros em 30 linhas de teste) como overfitting, e o teste do serviço de experiências falhou.
Uma diferença fixa não chega com testes pequenos: a regra desconta agora o ruído do próprio
teste, como o de uma proporção.

### O Parquet, o Excel antigo e a licença

O pedido seguinte (ADR 0052):

1. **O Parquet entra, com o `pyarrow`.** O leitor troca os tipos que só o pandas conhece (as
   categorias, os inteiros e os booleanos com nulos, o texto com `<NA>`, as datas com fuso)
   pelos de um CSV, e devolve o índice com nome como coluna.
2. **O Excel antigo (`.xls`) entra, com o `calamine`.** O motor vai sempre dito: o `calamine`
   para o `.xls` e o `openpyxl` de sempre para o `.xlsx`.
3. **A licença é a MIT**, no `LICENSE`, declarada no `pyproject.toml` e no `package.json`; o
   README diz o que ela permite.

Duas lições dos testes:

- o plano era ler o `.xls` com o `xlrd`, mas o pandas 3.1 avisa que esse motor vai sair, e os
  avisos contam como erros: trocou-se pelo `calamine`, o que o pandas recomenda. E com o
  `calamine` instalado, o `.xlsx` também passou a avisar (o motor por omissão vai mudar), por
  isso o motor do Excel ficou sempre explícito;
- o teste das datas com fuso usava uma hora de Lisboa na noite em que os relógios adiantam, e
  a terceira hora "saltava" de 0 para 2. O teste usa agora uma noite de inverno.

O `.xls` dos testes foi gravado pelo próprio Excel, porque nenhuma biblioteca de Python o grava
sem mais uma dependência; fica em `tests/unit/core/data/io/files/`, marcado como binário.

### O analista numa conversa, o quadro inteiro, apagar no History e pronto para outros computadores

O pedido seguinte, já com o analista da etapa 13.3 a responder (ADR 0054):

1. **O analista é uma conversa.** Um cartão ao centro da página, com a altura fixa (78% do ecrã,
   entre 32 e 52 rem), em que só a conversa desliza: em cima, o nome e o modelo de linguagem com
   o estado; no meio, as perguntas à direita e o analista à esquerda; em baixo, a caixa da
   pergunta (Enter pergunta, e o "Stop" toma o lugar do botão enquanto ele trabalha), com as
   sugestões que faltam numa fila que desliza para o lado. Sem perguntas, o meio dá as
   boas-vindas: as sugestões e as ferramentas. Ao vivo, a conversa segue os passos (quem sobe
   deixa de seguir, com o botão "Latest"); na resposta, os passos fecham-se numa linha com as
   ferramentas usadas ("How it got there"), e a resposta fica numa bolha, com a leitura por
   baixo. A página nunca mexe: nada de `scrollIntoView`, e o `overscroll-contain` não passa o
   deslizar à página.
2. **O quadro tem todos os candidatos e todos os modelos.** O `list_candidates` perdeu o `limit`
   (o `qwen3:8b` pedia sempre `limit=10`), cada candidato passou a uma linha de texto, com o
   número, o modelo e a nota juntos, e a lista `models` diz cada modelo uma vez, com o melhor
   candidato dele. Até 40 linhas (`MAX_ROWS`); acima disso, os de baixo saem das linhas mas não
   dos modelos.
3. **Apagar no History, à vista**: o caixote do lixo ao fim de cada linha (presa à direita, para
   se ver num portátil, onde a tabela desliza dentro da caixa) e "Delete the selected" ao lado de
   "Clear the selection", os dois com confirmação.
4. **Pronto para outros computadores**: o README abre com "Começar" (o que instalar em cada
   sistema, o `git clone`, o `uv sync`, o arranque e uma tabela "Se alguma coisa falhar"); o
   `scripts/dev.py` verifica o Node.js e as portas e instala a interface com `npm ci` quando o
   `package-lock.json` muda; o CI ganha o Windows e o macOS; o Prettier aceita os fins de linha
   de cada ficheiro; e as cópias de trabalho locais ficam fora do Git.

**Verificado com o `qwen3:8b` a sério**, numa procura do Titanic com 24 candidatos e 12 modelos:

- "Why did the winner win?": leu o quadro e comparou o #22 (gradient boosting) com o #16, em
  50,8 s, com todos os números verificados; a página não mexeu enquanto ele escrevia, e a
  conversa seguiu os passos até ao fim;
- "Which models did the agents try, and how far behind the winner was each one?": só com o
  quadro, em 35,1 s, disse os 12 modelos com o melhor candidato e a distância de cada um,
  incluindo os do 13.º ao 24.º lugar (a árvore de decisão, a random forest, o SVM, o AdaBoost,
  o KNN e o perceptrão), que o quadro dos dez primeiros nunca lhe mostrava.

Quatro lições:

- **Um teste que treina muitos runs corre no fim.** A API dos testes de ponta a ponta é uma só, e
  guarda em memória só os cinco runs mais recentes (`KEPT_RUNS`). O teste novo de apagar treina
  três, e, a correr ao mesmo tempo que os outros, empurrava para fora da memória os runs que o
  teste do History ia guardar e o do Explain ia explicar ("is no longer in memory"). Passou para
  um ficheiro seu (`e2e/history-delete.spec.ts`), num projeto do Playwright (`no-fim`) que só
  começa quando os outros acabam.
- **O tamanho mede-se, não se adivinha.** O plano eram 60 linhas, a contar 40 tokens cada. Com o
  `qwen3:8b` a contar: as instruções e as ferramentas custam 966 tokens; o quadro de 24
  candidatos, 2611 (as linhas, 1532, uns 65 cada, porque as mudanças dos hiperparâmetros são
  compridas); dez objetos do quadro antigo custavam 1098. Com 60 linhas, a conversa podia passar
  dos 8192 e o Ollama cortava o princípio sem avisar: ficaram 40.
- **Um botão que aparece não pode mexer na página.** O "Delete the selected" começou na linha dos
  filtros, que partia em duas ao marcar o primeiro run e empurrava a tabela: o segundo clique
  caía noutra linha. Foi para junto do "Clear the selection", que já era das ações sobre os
  escolhidos.
- **O Windows tem os seus fins de linha.** Com o `core.autocrlf` ligado, os ficheiros chegam com
  CRLF, e o `npm run format:check` dava 203 ficheiros por formatar sem nenhum estar mal: o
  Prettier passou a aceitar os fins de linha de cada ficheiro, e o repositório continua em LF.

### As dependências da interface sem avisos de segurança

O pedido seguinte (ADR 0055): limpar os 23 avisos do `npm audit` da interface antes de o
repositório ficar público. Vinham todos de quatro pacotes indiretos (`js-yaml`, `minimatch`,
`katex` e `deepmerge`), e o `npm audit fix --force` só sabia descer de versão.

- **Três `overrides`** obrigam a árvore a usar as versões corrigidas: o `js-yaml` 4.3.2 (o gerador
  do cliente fixava a 4.2.0), o `minimatch` 11.0.5 (a correção não chegou ao 10, que o ESLint pede)
  e o KaTeX do próprio projeto, 0.19.1, para o `rehype-katex` e o `remark-math` (`"$katex"`).
- **A ferramenta do shadcn saiu**: trazia o `deepmerge`, sem versão corrigida, e 230 pacotes, para o
  projeto usar só nove variantes e um utilitário da folha de estilos dela. Essas linhas estão
  agora no `globals.css`, e um componente novo acrescenta-se com `npx shadcn@latest add <nome>`.

Ficou o `npm audit` a 0, e o lockfile com 680 pacotes em vez de 910. Para ver que nada mudou:
o ESLint analisa os mesmos 205 ficheiros, o cliente gerado é igual byte a byte, o CSS do build só
perdeu sete propriedades que nenhuma classe usava, e os testes de ponta a ponta passam, com as
fórmulas do Metrics e do Train & evaluate desenhadas pelo KaTeX 0.19.1.

Uma lição: no Windows, o `npm ci` não consegue apagar um `node_modules` com um servidor do Vite a
correr dessa pasta (o ficheiro nativo do `lightningcss` fica preso, `EPERM`). O
`scripts/dev.py` não cai nisto, porque verifica a porta 5173 antes de instalar; quem tiver outro
Vite aberto noutra porta tem de o fechar primeiro.
