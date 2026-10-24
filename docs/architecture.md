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
  de gráficos, nem nada de UI. A API e a futura interface consomem o núcleo, nunca o
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
  fases 7 e 9 (ver [Tecnologias](#tecnologias)).
- Python 3.12+, scikit-learn para todo o ML clássico, PyTorch para deep learning.
- pandas apenas na camada de dados (`core/data/`); o resto do sistema trabalha com NumPy.
- Implementações didáticas em NumPy puro para os algoritmos simples, ao lado das versões do
  scikit-learn.
- Registo de experiências em SQLite com o módulo `sqlite3` da biblioteca padrão, atrás de um
  contrato trocável.
- Interface em duas versões com as mesmas funcionalidades: simples (Streamlit) a par de cada
  fase; aprimorada e personalizada (React) na fase 9.
- Reinforcement learning na fase 8, com ambiente próprio em grelha, sem bibliotecas de RL.
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
| 6 · Memória | **Registar** (run, métricas, configuração) → **Guardar** (modelo treinado e snapshots) → **Comparar** (2 ou 3 runs lado a lado) → **Reproduzir** (mesma configuração e mesma seed) | `core/tracking` · SQLite |
| 7 · Apresentação | **Serviços** (única porta do núcleo) → **Gráficos** (fronteiras, curvas, projeções 2D) → **UI simples** (Streamlit + Plotly, todas as funções) → **UI aprimorada** (React + API, WebSocket) | services · visualization_data · ui |

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

A stack final tem **13 tecnologias** para o backend e a interface simples, e mais **7** que
só entram nas fases 7 e 9 (API e interface aprimorada em React). Todos os tipos de
aprendizagem do plano ficam cobertos. Cada tecnologia tem um único papel; a coluna "Fase"
diz quando a começas a usar.

### Ferramentas de desenvolvimento

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| Python 3.12+ | Linguagem principal do ecossistema de ML | Todo o backend e a interface simples | 0 |
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
| Pydantic v2 | Validação de dados com tipos Python | Hiperparâmetros, configurações de experiência, fichas técnicas; gera os formulários das duas interfaces | 2 |
| SQLite (módulo `sqlite3`) | Base de dados num único ficheiro, incluída no Python | Registo de experiências, runs e métricas | 5 |

Configuração e logs usam só a biblioteca padrão do Python (um módulo de configuração e o
módulo `logging`), sem dependências extra.

### Interface simples

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| Streamlit | Transforma scripts Python em aplicações web, sem HTML nem JavaScript | Todos os ecrãs da interface simples, construídos a par do backend | 0 |
| Plotly | Gráficos interativos (zoom, hover, animação) | Gráficos da interface simples; o Plotly.js dá os mesmos gráficos no React | 1 |

### API e interface aprimorada (fases 7 e 9)

| Tecnologia | O que é | Para que serve no projeto | Fase |
|---|---|---|---|
| FastAPI | Framework web assíncrona baseada em Pydantic, com WebSockets incluídos | Expor os serviços por HTTP e transmitir o treino em tempo real; documentação automática em `/docs` | 7 |
| Uvicorn | Servidor que corre a aplicação FastAPI | Correr a API localmente | 7 |
| React | Biblioteca para interfaces feitas de componentes | Interface aprimorada e personalizada | 9 |
| TypeScript | JavaScript com tipos | Evitar erros; tipos gerados a partir da documentação da API | 9 |
| Vite | Ferramenta de desenvolvimento e build para front-end | Servidor local rápido e build final | 9 |
| Tailwind CSS | CSS por classes utilitárias | Sistema de design próprio (cores, tipografia, modo escuro) | 9 |
| D3.js (+ Plotly.js) | Visualização de baixo nível | Animações feitas à medida: fronteiras a mover-se, redes neuronais, centróides | 9 |

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
| Deep learning | Classificação / Regressão | MLP (rede densa) | PyTorch | Loss, pesos, gradientes, ativações e fronteira por época |
| Deep learning (extensão) | Classificação de imagem | CNN pequena | PyTorch | Filtros e mapas de ativação |
| Reinforcement learning (fase 8) | Controlo | Q-learning tabular, DQN | Didático (NumPy) / PyTorch | Recompensa por episódio, tabela Q, política |

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
a animação montada a partir dos snapshots. O registo de experiências chega na fase 5.

### Regra de dependências

| Camada | Pode importar | Nunca importa |
|---|---|---|
| Interface simples | `services`, `schemas` | Modelos, dados, treino ou base de dados diretamente |
| Interface aprimorada | API (por HTTP e WebSocket) | Nada do Python diretamente |
| API | `services`, `schemas` | Modelos, dados, treino ou base de dados diretamente |
| Serviços | Todo o `core` | `api`, `ui` |
| Núcleo | Fundação, bibliotecas de ML; pandas só em `core/data/` | `api`, `ui`, `services` |
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
│   ├── config.py             # definições: caminhos, base de dados, logs
│   ├── logging.py            # configuração de logs estruturados
│   ├── formatting.py         # números à portuguesa (19,9%, 20 640)
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
│   │   │   │   └── growth.py             # modelos do scikit-learn que crescem passo a passo
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
│   │   │   └── neural/       # redes PyTorch, captura de ativações e gradientes (fase 6)
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
│   │   │   └── evaluation.py # os gráficos do ecrã Avaliar (ROC, resíduos, silhueta…)
│   │   ├── tracking/         # registo de experiências, base de dados, artefactos
│   │   ├── schemas/          # esquemas Pydantic partilhados
│   │   │   ├── hyperparameters.py  # base dos hiperparâmetros (formulários, validação)
│   │   │   ├── experiment.py # ExperimentConfig: dados, preparação, modelo e seed
│   │   │   └── validation.py # erros do Pydantic explicados em português
│   │   ├── services/         # casos de uso: a única porta de entrada do núcleo
│   │   │   ├── datasets.py   # carregar, perfilar e preparar dados (ecrãs 1 a 3)
│   │   │   ├── catalog.py    # catálogo de modelos (ecrã 4)
│   │   │   ├── experiments.py  # treinar (também em segundo plano), avaliar, anunciar (4 a 6)
│   │   │   ├── analysis.py   # depois do treino: gráficos, validação cruzada, explicações (6 e 7)
│   │   │   ├── laboratory.py # as demonstrações do ecrã Laboratório (9)
│   │   │   ├── explanations.py # o painel "Porquê?": os temas de todas as camadas
│   │   │   └── lab.py        # todos os serviços de uma sessão, ligados entre si
│   │   └── rl/               # reservado (fase 8): ambiente em grelha, Q-learning, DQN
│   │
│   └── api/                  # FastAPI: rotas HTTP e WebSocket
│
├── ui/
│   ├── simple/               # INTERFACE SIMPLES (Streamlit + Plotly)
│   │   ├── app.py            # entrada e navegação
│   │   ├── pages/            # um ecrã por etapa (Dados, Explorar, Preparar…)
│   │   ├── components/       # blocos reutilizáveis: cartão de métrica, formulário
│   │   └── charts/           # desenha com Plotly os dados de visualization_data
│   └── web/                  # INTERFACE APRIMORADA (React + TypeScript), fase final
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
  trabalho da interface, para que os mesmos dados sirvam a versão simples e a aprimorada.
- `services/` é a única porta de entrada do núcleo: a interface simples chama-a
  diretamente; a API chama-a para a interface aprimorada.
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
   preenchido.
3. Decorar com `@MODELS.register("extra_trees")`.
4. Escrever `docs/algorithms/extra_trees.md` (intuição, hiperparâmetros, quando usar).
5. Correr `uv run pytest tests/contract` — o modelo é testado sem escrever testes novos.

### Adicionar uma métrica

1. Escrever uma função pura em `evaluation/metrics/<tipo>.py`.
2. Decorar com `@metric("balanced_accuracy", task_types={...}, higher_is_better=True)`.
3. Juntar a explicação em `evaluation/explanations.py`.

A métrica aparece em todos os relatórios dos tipos de tarefa indicados.

### Adicionar uma fonte de dados

Implementar `DataSource.load() -> LoadedTable` (o DataFrame e os seus metadados) e
`DataSource.spec()` num novo ficheiro em `data/io/`, e registar com
`@DATA_SOURCES.register("nome")`. Ex.: uma fonte que lê de uma URL, ou de uma base de dados.
O `Dataset` com arrays NumPy só existe no fim da camada de dados (ADR 0009).

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

A interface é feita em duas versões com **as mesmas funcionalidades**: primeiro uma simples
(Streamlit), que cresce a par do backend em cada fase; no fim, uma aprimorada e bem
desenhada (React), que muda a apresentação mas não o que se pode fazer. Assim validas tudo
cedo e o redesenho final não obriga a repensar o produto.

### Ecrãs (iguais nas duas versões)

Os ecrãs seguem a ordem natural de uma experiência, da esquerda para a direita numa barra de
navegação.

| # | Ecrã | O que o utilizador faz | O que vê | Fase do backend |
|---|---|---|---|---|
| 1 | Dados | Carrega um CSV/Excel, escolhe um dataset de exemplo ou gera um sintético (com controlos de ruído e número de pontos) | Lista de datasets, pré-visualização das primeiras linhas, erros de validação | 1 |
| 2 | Explorar | Navega pelas colunas | Perfil de cada coluna, histogramas, mapa de correlações, mapa de nulos, desequilíbrio de classes, avisos | 1 |
| 3 | Preparar | Escolhe a coluna-alvo, confirma o tipo de tarefa, aplica limpezas e transformações, define a divisão | Antes/depois de cada operação, histórico de passos (linhagem), tamanho de treino/teste | 1, 2 |
| 4 | Modelo | Escolhe um algoritmo do catálogo filtrado pela tarefa e ajusta hiperparâmetros | Cartões por família, ficha didática, formulário gerado da ficha técnica | 2, 4 |
| 5 | Treinar | Inicia, pausa, cancela; percorre o treino passo a passo | Barra de progresso, loss em tempo real, animação dos snapshots com controlos de reprodução, fronteira de decisão / centróides / árvore | 3 |
| 6 | Avaliar | Lê os resultados | Cartões de métricas com explicação, matriz de confusão, ROC e PR, resíduos, silhouette, comparação treino vs. teste, diagnósticos | 2, 4 |
| 7 | Explicar | Escolhe uma previsão ou o modelo inteiro | Importância de features, gráficos de dependência parcial | 4 |
| 8 | Experiências | Filtra o histórico, escolhe 2 ou 3 runs, reproduz um run | Tabela de runs, comparação lado a lado de métricas e gráficos | 5 |
| 9 | Laboratório | Mexe num controlo e vê o efeito imediato | Demos de conceitos: overfitting (complexidade do modelo), taxa de aprendizagem, k do KNN, regularização | 4 |
| 10 | Reinforcement learning | Escolhe ambiente e agente, inicia o treino | Recompensa por episódio, tabela Q, política aprendida | 8 |

### Elementos comuns a todos os ecrãs

- Barra lateral com o estado da experiência atual: dataset, tarefa, modelo, último
  resultado.
- Painel "Porquê?" com a explicação didática do que está no ecrã.
- Avisos destacados quando algo merece atenção (fuga de dados, desequilíbrio, overfitting).
- Botão para guardar a experiência em qualquer momento.

### Simples vs. aprimorada

| Aspeto | Interface simples | Interface aprimorada |
|---|---|---|
| Tecnologia | Streamlit + Plotly (só Python) | React + TypeScript + Vite + Tailwind CSS + Plotly.js/D3 |
| Ligação ao backend | Chama a camada de serviços diretamente, no mesmo processo Python | Através da API (HTTP para pedidos, WebSocket para o treino) |
| Tempo real | Atualização periódica do ecrã de treino | Snapshots chegam em fluxo contínuo |
| Animações | Slider para percorrer snapshots | Animações suaves, transições entre estados |
| Design | Tema padrão do Streamlit com pequenos ajustes | Sistema de design próprio: cores, tipografia, componentes, modo escuro |
| Objetivo | Validar funcionalidades e aprender depressa | Experiência polida para usar e mostrar |
| Quando | Cresce a cada fase, a partir da fase 1 | Fase 9, com a API estável |

### Regras para as duas versões partilharem tudo

1. Nenhuma lógica de ML ou de dados vive na interface; a interface só pede e mostra.
2. Os gráficos são desenhados a partir dos dados preparados por `visualization_data/`: a
   versão simples e a aprimorada recebem exatamente os mesmos números.
3. Os formulários de hiperparâmetros são gerados a partir das fichas técnicas dos modelos,
   nas duas versões.
4. Os textos didáticos vêm do backend (fichas e explicações de métricas), por isso só se
   escrevem uma vez.
5. Cada ecrã da versão simples é o rascunho funcional do mesmo ecrã da versão aprimorada.

## Roadmap por fases

Dez fases (0 a 9). Cada uma termina com algo que funciona, está testado e já se vê na
interface simples. Cada fase introduz poucas tecnologias novas, para as aprenderes uma de
cada vez.

| Fase | Backend | Interface simples (ecrãs que ganha) | Tecnologias novas | Pronto quando |
|---|---|---|---|---|
| 0 — Fundações ✅ | Repositório, configuração, logs, registo genérico, tipos, eventos, seeds; testes de arquitetura | Esqueleto com navegação entre ecrãs vazios | Python, uv, ruff, mypy, pytest, Streamlit | `pytest` e `mypy` passam; a app abre |
| 1 — Camada de dados ✅ | As nove etapas: entrada, validação, perfil, limpeza, tarefa, divisão, transformação, projeção, saída; linhagem | Dados, Explorar, Preparar | pandas, NumPy, Plotly | Carregar o Titanic mostra perfil correto, avisos e a sugestão "classificação binária"; o teste de arquitetura confirma o pandas isolado |
| 2 — Primeiro fluxo completo ✅ | Contrato de modelo, adaptador scikit-learn, regressão logística, avaliador de classificação, serviço de experiências | Modelo, Avaliar (básico) | scikit-learn, Pydantic | No Iris: escolher modelo, treinar e ver métricas na interface; testes de contrato a passar |
| 3 — Eventos e snapshots ✅ | Contexto de treino, gravador de snapshots, regressões didáticas em NumPy, dados da fronteira de decisão | Treinar (com reprodução passo a passo) | — (gradiente descendente, padrão observer) | Ver a fronteira da regressão logística didática a mover-se; resultado final igual ao do scikit-learn |
| 4 — Catálogo clássico ✅ | Todos os modelos supervisionados e não supervisionados; avaliadores de regressão, clustering e anomalias; diagnósticos; validação cruzada; importância por permutação | Avaliar (completo), Explicar, Laboratório | — (cada algoritmo e as suas métricas) | Todos os modelos passam os testes de contrato e têm ficha didática |
| 5 — Memória | Base de dados, registo de experiências, artefactos, comparação e reprodução de runs | Experiências | SQLite (`sqlite3`), joblib | Fechar a app, reabrir e comparar dois runs |
| 6 — Deep learning | Adaptador PyTorch, MLP, captura de ativações e gradientes, early stopping | Treinar mostra as camadas da rede | PyTorch | A MLP resolve "moons" e os snapshots mostram a fronteira a curvar |
| 7 — API | Rotas de dados, catálogo e experiências; treino em background; WebSocket | — (a simples continua a usar os serviços) | FastAPI, Uvicorn | Pela documentação em `/docs`: carregar CSV, lançar treino e receber snapshots |
| 8 — Reinforcement learning (opcional) | Contratos de ambiente, agente e episódio; ambiente em grelha próprio; Q-learning didático; DQN em PyTorch | Reinforcement learning | — | Um agente aprende a atravessar a grelha e o run fica registado |
| 9 — Interface aprimorada | Ajustes à API que a interface pedir | Todos os ecrãs refeitos em React com design próprio | React, TypeScript, Vite, Tailwind CSS, D3.js, Plotly.js | Tudo o que a versão simples faz, com animações suaves e treino em tempo real |

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
| 0005 | Interface simples (Streamlit) primeiro, React no fim | Validar todas as funcionalidades cedo; o redesenho final só muda a apresentação |

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
