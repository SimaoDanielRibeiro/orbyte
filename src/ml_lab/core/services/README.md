# Serviços (`core/services/`)

Casos de uso: a **única porta de entrada do núcleo**. A interface simples chama os serviços
diretamente; a API chama-os para a interface aprimorada. Cada serviço recebe configurações
validadas e devolve resultados prontos a mostrar, em estruturas simples (sem DataFrames).

- **Pode importar:** todo o núcleo.
- **Nunca importa:** `ml_lab.api` nem a interface.

Como a interface só pode importar `ml_lab.core.services` e `ml_lab.core.schemas`,
`__init__.py` reexporta os tipos de que ela precisa (resultados, configurações, o erro base).

**A fronteira está pronta para a API (ADR 0019).** Cada método público devolve dados
simples que passam a JSON (`schemas.serialization.to_json`). O teste
`tests/contract/test_service_boundary.py` verifica-o para todos os métodos, sem ser
preciso acrescentar nada. Os métodos que só servem outros serviços (devolvem DataFrames
ou o modelo) levam `@internal`, e as regras de domínio ficam aqui, não na interface.

| Serviço | O que faz | Fase |
|---|---|---|
| `datasets.py` · `DatasetService` | Exemplos, geradores, upload, validação, perfil, sugestões de tarefa e de limpeza, opções de cada tarefa, preparação | 1 |
| `catalog.py` · `CatalogService` | Os modelos de cada tarefa, com a ficha didática e o JSON Schema dos hiperparâmetros | 2 |
| `experiments.py` · `ExperimentService` | Corre uma experiência (`ExperimentConfig`), também em segundo plano: prepara, treina, avalia, compara treino e teste e anuncia o run no `EventBus` | 2 |
| `analysis.py` · `AnalysisService` | Depois do treino: gráficos do Avaliar, validação cruzada, importância, dependência parcial, explicação de uma previsão | 4 |
| `laboratory.py` · `LaboratoryService` | As quatro demonstrações do Laboratório | 4 |
| `history.py` · `HistoryService` | O histórico (ADR 0020): regista cada run sozinho, guarda o modelo e o treino passo a passo a pedido, reabre, compara (`comparison.py`) e reproduz runs | 5 |
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
