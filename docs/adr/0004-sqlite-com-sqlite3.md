# 0004 — SQLite com `sqlite3`

- **Estado:** Aceite
- **Data:** 2026-10-22

## Contexto

O registo de experiências (fase 5) guarda runs, configurações, métricas e caminhos para os
artefactos, e tem de permitir comparar e reproduzir runs depois de fechar a app. O ML Lab
corre localmente, para um só utilizador, sem servidores.

## Decisão

O registo usa SQLite através do módulo `sqlite3` da biblioteca padrão, atrás do contrato
`ExperimentStore`. Há também uma implementação em memória para os testes. A base de dados é
um único ficheiro (por omissão `artifacts/ml_lab.db`, configurável com
`ML_LAB_DATABASE_PATH`).

## Consequências

- Zero dependências extra; a base de dados é um ficheiro fácil de copiar, apagar e
  inspecionar.
- O SQL é escrito à mão, o que também é aprendizagem; não há migrações automáticas, por isso
  `scripts/` terá um utilitário para recriar a base de dados.
- A concorrência do SQLite é limitada, o que chega para um utilizador.
- Trocar para SQLAlchemy + Alembic ou para MLflow é escrever outra implementação do
  `ExperimentStore`, sem mudar quem a usa.
