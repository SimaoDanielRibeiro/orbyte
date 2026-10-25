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
| `experiments.py` · `ExperimentService` | Corre uma experiência (`ExperimentConfig`), também em segundo plano: prepara, treina, avalia, compara treino e teste e anuncia o run no `EventBus`. Os treinos em segundo plano acompanham-se pelo id do run (`job`, `updates`, `pause`, `resume`, `cancel`, `result`) | 2, 7 |
| `analysis.py` · `AnalysisService` | Depois do treino: gráficos do Avaliar, validação cruzada, importância, dependência parcial, explicação de uma previsão | 4 |
| `laboratory.py` · `LaboratoryService` | As quatro demonstrações do Laboratório | 4 |
| `history.py` · `HistoryService` | O histórico (ADR 0020): regista cada run sozinho, guarda o modelo e o treino passo a passo a pedido, reabre, compara (`comparison.py`) e reproduz runs, de modelos e de agentes | 5, 8 |
| `reinforcement.py` · `ReinforcementService` | Treina agentes num ambiente (`ReinforcementConfig`), também em segundo plano, avalia a política aprendida e anuncia o run (ADR 0024) | 8 |
| `automl.py` · `AutoMLService` | O modo automático (`AutoMLRequest`): o agente de dados decide a preparação, os agentes de modelo e o coordenador procuram o melhor modelo com validação cruzada só no treino, e o vencedor treina como um run normal do `ExperimentService` (ADR 0025). Acompanha-se pelo id da procura, como os treinos, e `train_candidate` treina um candidato à mão | 9 |
| `lab.py` · `Lab` | Os serviços de uma sessão, ligados entre si e a um só `EventBus` | 1 |

Cada método público tem a sua rota na API (`ml_lab/api/routes/<serviço>.py`), com o nome do
método (ADR 0022).

Os textos das interfaces não são um serviço do `Lab`, mas funções: `explain(tema)` e
`topics()` (`explanations.py`), com os temas do painel "Porquê?" de todas as camadas, e
`vocabulary()` (`vocabulary.py`), com o nome a mostrar de cada valor de cada enum e o que muda
com cada tarefa. O frontend pede-os uma vez, ao arrancar (`GET /explanations` e
`GET /vocabulary`, ADR 0026).

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
