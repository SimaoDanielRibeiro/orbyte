# Fundação (`core/foundation/`)

Peças usadas por todas as camadas. Só depende da biblioteca padrão (e do Pydantic, a partir
da fase 2): **nunca importa outra pasta do projeto**.

| Módulo | O que tem |
|---|---|
| `types.py` | `TaskType` (os seis tipos de tarefa) e `Paradigm` |
| `errors.py` | Exceções do domínio; todas herdam de `MLLabError` |
| `registry.py` | `Registry`: componentes por nome, filtro por tipo de tarefa, descoberta automática |
| `events.py` | Eventos de treino (`RunStarted`, `StepCompleted`...) e o `EventBus` |
| `random.py` | `SeedManager`: seeds derivadas por nome a partir da seed raiz |

Os logs da fundação usam `logging.getLogger(__name__)` da biblioteca padrão: como o nome
começa por `ml_lab`, herdam a configuração feita em `ml_lab.logging`.

## Exemplos

```python
from ml_lab.core.foundation.events import EventBus, StepCompleted
from ml_lab.core.foundation.random import SeedManager
from ml_lab.core.foundation.registry import Registry
from ml_lab.core.foundation.types import TaskType

# Registo: cada componente regista-se sozinho e encontra-se pelo nome.
MODELS = Registry("models", task_types=lambda model: model.task_types)


@MODELS.register("logistic")
class Logistic:
    task_types = {TaskType.BINARY_CLASSIFICATION}


MODELS.get("logistic")  # a classe Logistic
MODELS.for_task(TaskType.BINARY_CLASSIFICATION)  # {"logistic": Logistic}

# Eventos: o treino anuncia, quem quiser ouve.
bus = EventBus()
subscription = bus.subscribe(StepCompleted, lambda event: print(event.step, event.loss))
bus.publish(StepCompleted(run_id="run-1", step=1, loss=0.42))
bus.unsubscribe(subscription)

# Seeds: uma raiz por experiência, uma seed independente por componente.
seeds = SeedManager.create(42)
seeds.seed_for("split")  # sempre o mesmo número para a raiz 42
```
