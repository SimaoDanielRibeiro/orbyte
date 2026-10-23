# Serviços (`core/services/`)

Casos de uso: a **única porta de entrada do núcleo**. A interface simples chama os serviços
diretamente; a API chama-os para a interface aprimorada. Cada serviço recebe configurações
validadas e devolve resultados prontos a mostrar, em estruturas simples (sem DataFrames).

- **Pode importar:** todo o núcleo.
- **Nunca importa:** `ml_lab.api` nem a interface.

Como a interface só pode importar `ml_lab.core.services` e `ml_lab.core.schemas`,
`__init__.py` reexporta os tipos de que ela precisa (resultados, configurações, o erro base).

| Serviço | O que faz | Fase |
|---|---|---|
| `datasets.py` · `DatasetService` | Exemplos, geradores, upload, validação, perfil, sugestões de tarefa e de limpeza, preparação | 1 |
| `catalog.py` · `CatalogService` | Os modelos de cada tarefa, com a ficha didática e o JSON Schema dos hiperparâmetros | 2 |
| `experiments.py` · `ExperimentService` | Corre uma experiência (`ExperimentConfig`): prepara, treina, avalia, compara treino e teste e anuncia o run no `EventBus` | 2 |
| `lab.py` · `Lab` | Os serviços de uma sessão, ligados entre si | 1 |

## Exemplo

```python
from ml_lab.core.services import ExperimentConfig, Lab, ModelSpec, Partition, PreparationConfig

lab = Lab.create()
iris = lab.datasets.load_sample("iris")
result = lab.experiments.run(
    ExperimentConfig(
        dataset=iris.source,
        preparation=PreparationConfig(target="species"),
        model=ModelSpec(name="logistic_regression", hyperparameters={"C": 0.5}),
        seed=42,
    )
)
result.primary.formatted  # exatidão no teste: "93,3%"
for metric in result.reports[Partition.TEST].metrics:
    print(metric.display_name, metric.formatted, metric.what)
```
