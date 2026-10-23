# Núcleo (`core/`)

O domínio do ML Lab: dados, modelos, treino, avaliação e registo de experiências. O núcleo
não sabe que existe uma interface ou uma API.

- **Pode importar:** a fundação e as bibliotecas de ML (NumPy, scikit-learn, PyTorch). O
  pandas só em `data/`.
- **Nunca importa:** `ml_lab.api`, a interface, nem bibliotecas de interface ou de gráficos.
  Fora de `services/`, também não importa `services/`.

Estas regras são verificadas em `tests/architecture/`.

| Pasta | Responsabilidade | Fase |
|---|---|---|
| `foundation/` | Tipos de tarefa, erros, registo de componentes, eventos e seeds | 0 |
| `data/` | As nove etapas da camada de dados; o único sítio com pandas | 1 |
| `models/` | Algoritmos: um ficheiro = um modelo | 2 a 6 |
| `training/` | Trainer, contexto de treino, snapshots, validação cruzada (o contrato do contexto existe desde a fase 2) | 2 e 3 |
| `evaluation/` | Métricas, avaliadores por tipo de tarefa, diagnósticos | 2 |
| `explainability/` | Importância por permutação e dependência parcial | 4 |
| `visualization_data/` | Dados para gráficos: calcula números, nunca desenha | 3 |
| `tracking/` | Registo de experiências em SQLite e artefactos | 5 |
| `schemas/` | Esquemas Pydantic partilhados | 2 |
| `services/` | Casos de uso: a única porta de entrada do núcleo | 1 e 2 |
| `rl/` | Reinforcement learning (reservado) | 8 |
