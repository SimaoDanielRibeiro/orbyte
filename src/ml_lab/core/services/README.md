# Serviços (`core/services/`)

Casos de uso: a **única porta de entrada do núcleo**. A interface simples chama os serviços
diretamente; a API chama-os para a interface aprimorada. Cada serviço recebe configurações
validadas e devolve resultados prontos a mostrar, em estruturas simples (sem DataFrames).

- **Pode importar:** todo o núcleo.
- **Nunca importa:** `ml_lab.api` nem a interface.

Serviços previstos: datasets (fase 1), catálogo de modelos e experiências (fase 2).
