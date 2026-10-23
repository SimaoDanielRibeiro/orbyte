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
| `lab.py` · `Lab` | Os serviços de uma sessão, ligados entre si | 1 |

Serviços previstos: catálogo de modelos e experiências (fase 2).
