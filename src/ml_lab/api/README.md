# API (`api/`)

FastAPI: rotas HTTP para dados, catálogo e experiências, e um WebSocket que transmite os
eventos de treino em tempo real. A documentação interativa é gerada automaticamente em
`/docs`, a partir dos esquemas Pydantic. Chega na fase 7.

- **Pode importar:** `ml_lab.core.services` e `ml_lab.core.schemas`.
- **Nunca importa:** modelos, dados, treino ou base de dados diretamente, nem a interface.
