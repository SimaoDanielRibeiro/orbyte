# Camada de dados (`core/data/`)

O único sítio do projeto onde o pandas é usado. Os dados entram como ficheiro, passam por
nove etapas e saem como um `Dataset` com arrays NumPy e metadados.

| # | Etapa | Onde | Altera os dados? |
|---|---|---|---|
| 1 | Entrada | `io/`: `FileSource` (CSV, Excel), `SampleSource` (exemplos), `SyntheticSource` | Não |
| 2 | Validação | `validation/`: ficheiro, cabeçalho, tabela e regras por coluna | Não |
| 3 | Perfil | `profiling/`: tipo inferido, estatísticas, correlações, classes | Não |
| 4 | Limpeza | `cleaning/`: receita de operações registadas em `CLEANING` | Sim, com registo |
| 5 | Tarefa | `task/`: sugere o `TaskType` a partir do alvo | Não |
| 6 | Divisão | `splitting/`: aleatória, estratificada ou temporal, com seed | Separa |
| 7 | Transformação | `transforms/`: imputação, one-hot, escala, datas, polinomiais, intervalos | Sim |
| 8 | Projeção | `projection/`: duas colunas ou PCA, para os gráficos | Cópia à parte |
| 9 | Saída | `boundary.py`: matrizes `float64`, alvo codificado | Converte |

`preparation.py` corre as etapas 4 a 9 pela ordem certa, a partir de uma configuração
declarativa; `lineage.py` regista cada operação; `explanations.py` tem os textos didáticos de
cada etapa.

## Regras desta camada

1. **Limpeza antes da divisão, transformação depois.** A limpeza não aprende nada com os
   dados; a transformação aprende parâmetros (a média para imputar, o desvio para escalar) e
   por isso só pode ver o treino. Se visse o teste, seria *data leakage*.
2. **Nunca alterar o original.** Cada etapa devolve um DataFrame novo; o ficheiro carregado
   fica intacto.
3. **Tudo fica registado (linhagem).** Cada operação aplicada ("removidas 12 linhas
   duplicadas") fica em `lineage.py`, para a interface a mostrar e a experiência poder ser
   reproduzida.
4. **Explicar em vez de esconder.** Cada etapa produz avisos didáticos ("30% de nulos na
   coluna cabine").
5. **Datasets grandes:** o perfil e os gráficos usam uma amostra; o treino usa tudo.
6. **Nomes de colunas preservados até ao fim**, para chegarem à explicabilidade e à
   interface.

O pandas não sai daqui: `boundary.py` é a fronteira, e um teste de arquitetura falha se
algum ficheiro fora desta pasta o importar. Esta camada não usa o scikit-learn (ADR 0008).

## Exemplo

```python
from ml_lab.core.data.cleaning.base import CleaningStep
from ml_lab.core.data.io.builtin_source import SampleSource
from ml_lab.core.data.preparation import PreparationConfig, prepare
from ml_lab.core.data.profiling.dataset_profile import profile_dataset
from ml_lab.core.data.splitting.base import SplitConfig, SplitStrategy
from ml_lab.core.foundation.random import SeedManager

table = SampleSource("titanic").load()  # 1. entrada
profile = profile_dataset(table.frame, target="Survived")  # 3. perfil
for notice in profile.all_notices:
    print(notice.level, notice.message)  # "A coluna Cabin tem 77,1% de valores em falta…"

config = PreparationConfig(
    target="Survived",
    cleaning=(
        CleaningStep("drop_columns", {"columns": ["PassengerId", "Name", "Ticket", "Cabin"]}),
    ),
    split=SplitConfig(SplitStrategy.STRATIFIED, test_size=0.2),
)
dataset = prepare(table, config, seeds=SeedManager(42)).dataset  # 4 a 9
dataset.train.X.shape  # (712, 10)
dataset.class_labels  # ("0", "1")
for step in dataset.lineage:
    print(step.stage.label, step.description)
```

## Acrescentar componentes

- **Fonte de dados:** uma subclasse de `DataSource` em `io/`, com `@DATA_SOURCES.register(...)`.
- **Gerador sintético:** uma função `(rng, **parâmetros) -> Generated` com `@generator(...)`.
- **Operação de limpeza:** uma função `(frame, **parâmetros) -> CleaningResult` com
  `@cleaning_operation(...)`.
- **Estratégia de divisão:** uma função com `@SPLITTERS.register(...)` em `splitting/`.

Os testes de contrato (`tests/contract/test_data_components.py`) apanham-nos sozinhos.
