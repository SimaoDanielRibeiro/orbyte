# 0020 — Registo de experiências: registar sempre, guardar a pedido

- **Estado:** Proposta
- **Data:** 2026-10-24

## Contexto

A fase 5 dá memória ao ML Lab: fechar a app, reabrir e comparar runs. O plano divide a
região "Memória" em dois passos: **registar** (o run, as métricas, a configuração) e
**guardar** (o modelo treinado e os snapshots). Há três forças em jogo:

- um run esquecido perde-se: se só ficassem os runs guardados à mão, o histórico teria
  buracos;
- a animação do treino pesa (de meio megabyte a alguns MB num modelo por gradiente
  descendente), e o modelo treinado também (uma floresta pode ter vários MB). Guardar tudo
  de todos os runs enchia o disco depressa;
- desde a ADR 0019, o resultado de um run (`ExperimentResult`) passa a JSON e volta igual.
  Uma experiência confirmou-o com seis tipos de tarefa.

## Decisão

- **Registar é automático e leve.** O registo subscreve o fim de cada run no `EventBus`
  (`RunFinished`): o `Lab` liga-o, e o serviço de experiências não sabe que ele existe.
  Grava-se uma entrada para o histórico e o resultado em JSON **sem o treino passo a
  passo** (a animação e, desde a fase 6, as camadas de uma rede neuronal em cada snapshot;
  ver a ADR 0021). Um registo que falha só fica nos logs: o treino correu bem e não se
  estraga.
- **Guardar é a pedido** (o botão "Guardar experiência"). Junta o modelo treinado
  (`joblib`) e o treino passo a passo (um JSON por documento, `animation.json` e
  `network.json`) numa pasta por run, `artifacts/runs/<run_id>/`.
- **SQLite com duas tabelas** (ADR 0004):
  - `runs`: uma linha por run, com o documento JSON;
  - `metrics`: uma linha por métrica, para ordenar e filtrar o histórico em SQL sem abrir
    os documentos.
- **O esquema tem versão** (`PRAGMA user_version`). Uma base de dados antiga recebe as
  migrações que lhe faltam ao abrir. Isto substitui o utilitário de recriar a base de
  dados previsto na ADR 0004: para recomeçar do zero, basta apagar o ficheiro.
- **Cada operação abre a sua ligação**, porque o registo é chamado no fio do treino e uma
  ligação do `sqlite3` não se partilha entre fios.
- **Reabrir um run guardado devolve tudo.** O modelo vem do ficheiro e os dados
  preparam-se outra vez a partir da configuração (a mesma seed dá os mesmos dados). O run
  volta à memória do serviço de experiências, e por isso os gráficos do Avaliar e o
  Explicar funcionam depois de fechar a app. Se o modelo não abrir, o resultado abre na
  mesma, com um aviso.
- **Comparar mostra só o que muda:**
  - as métricas lado a lado, com o melhor marcado;
  - as escolhas diferentes (modelo, hiperparâmetros com o título do formulário, seed,
    passos da preparação);
  - as curvas da loss;
  - uma leitura: reproduziu-se, só mudou a seed, mudaram várias coisas.
- **Reproduzir** volta a treinar a configuração guardada. O run novo entra no histórico, e
  a comparação com o original confirma que as métricas são iguais.
- **O contrato `ExperimentStore`** tem duas implementações, em memória e SQLite, testadas
  pelos mesmos testes de contrato. O SQLite e o joblib só se usam em `core/tracking/`, e
  há uma regra de arquitetura que o verifica.
- **Os testes nunca tocam na base de dados verdadeira:** o `conftest.py` aponta as pastas
  e a base de dados para uma pasta temporária.

## Consequências

- O histórico está completo e ocupa pouco. Só os runs que interessam pesam no disco.
- Um run só registado reabre com tudo menos o modelo e o treino passo a passo: as
  métricas, a curva da loss e o que o modelo aprendeu estão lá. Para o resto, reproduz-se.
- Um ficheiro joblib é um *pickle*, e só se abrem os que o ML Lab gravou. Se uma classe de
  modelo mudar de sítio ou de forma, os modelos antigos deixam de abrir (o aviso explica).
  Reproduzir o run recria o modelo.
- Só se guarda um run que ainda esteja em memória (os últimos cinco desta sessão). Um run
  antigo, só registado, já não tem modelo para guardar.
- Trocar o SQLite por MLflow, ou por SQLAlchemy com Alembic, é escrever outro
  `ExperimentStore` e passá-lo ao `Lab.create`.
