# 0024 — Reinforcement learning: um paradigma novo, com a mesma maquinaria de treino

- **Estado:** Proposta
- **Data:** 2026-10-24

## Contexto

A fase 8 traz o reinforcement learning. Pede três coisas:

- os contratos de ambiente, agente e episódio;
- um ambiente em grelha próprio, o Q-learning didático e o DQN em PyTorch;
- o critério de pronto: um agente aprende a atravessar a grelha e o run fica registado.

O reinforcement learning não encaixa no resto do ML Lab, porque não há dataset. Um agente
aprende a jogar, pelas recompensas, e não a prever um alvo. Mas o que está à volta é o mesmo:

- um treino por passos, que se quer observar, pausar e parar;
- fotografias do que se aprendeu;
- métricas;
- um registo, uma comparação e uma reprodução.

Escrever tudo isto de novo para o RL duplicaria as fases 3, 5 e 7.

## Decisão

- **O controlo é um tipo de tarefa (`TaskType.CONTROL`), num paradigma novo
  (`Paradigm.REINFORCEMENT`).**
  - `TaskType.learns_from_data` separa-o das tarefas com dataset.
  - A deteção de tarefas e o catálogo de modelos só usam as tarefas com dataset.
  - O controlo não tem `Evaluator`, porque um agente avalia-se a jogar.
- **Os contratos ficam em `core/rl/`**, com registos e descoberta automática, como os modelos:
  - `Environment` (`ENVIRONMENTS`): discreto e pequeno (os estados e as ações contam-se), e
    sabe como se desenha (`GridLayout`). **Não guarda o estado do episódio**: `step(estado,
    ação, rng)` devolve o estado seguinte, a recompensa e se o episódio acabou. O mesmo
    ambiente serve o treino, a avaliação e o desenho do caminho aprendido, e com `rng=None`
    não há acaso;
  - `Agent` (`AGENTS`): aprende os valores Q. `EpisodicAgent` escreve uma vez o ciclo de
    episódios com ε-greedy, e os agentes só dizem como aprendem com um passo (`_learn`).
    Esgotar os passos não é um estado final, por isso o alvo continua a contar com o futuro.
- **O episódio é o passo do treino.** O agente fala com o mesmo `TrainingContext` dos
  modelos: anuncia cada episódio (a recompensa, a duração, se chegou e o ε) e entrega
  fotografias (a tabela Q e o caminho que faria). Vêm de graça:
  - a pausa e a paragem;
  - os snapshots amostrados;
  - o `JobState` e o `JobRegistry`;
  - a API e o WebSocket da ADR 0022.
- **O treino e o teste têm o mesmo sentido de sempre:**
  - o treino são os últimos 100 episódios do treino, ainda a explorar;
  - o teste são 20 episódios novos com a política aprendida, sem explorar e sem aprender.

  A métrica principal é a taxa de sucesso no teste. A diferença entre os dois é o preço de
  explorar, e o relatório explica-o (`exploration_cost`).
- **O histórico é o mesmo.** O `ReinforcementResult` usa os nomes que o histórico e a
  comparação usam (`reports`, `history`, `main_report`, e `dataset_name` e `model_name`, que
  são o ambiente e o agente):
  - os runs de agentes registam-se, guardam-se, comparam-se e reproduzem-se como os outros;
  - abrem-se com `open_reinforcement`;
  - o que o agente sabia em cada snapshot (`trace`) pesa, e por isso, como a animação, só se
    guarda a pedido (ADR 0020).
- **O DQN é o único sítio do RL com PyTorch.** Tem uma rede com o estado em one-hot, uma
  memória de repetição, uma rede-alvo e a loss de Huber, com a seed passada a geradores
  próprios. A regra de arquitetura do PyTorch passa a incluir `core/rl/agents/dqn.py`.

## Consequências

- Um ambiente novo (um tabuleiro, um labirinto gerado) ou um agente novo (o SARSA, por
  exemplo) é um ficheiro e uma ficha. Os testes de contrato
  (`tests/contract/test_rl.py`) testam-nos sozinhos: se aprendem a grelha simples, se repetem
  com a mesma seed, se param a pedido, se o objetivo se alcança.
- Os ambientes têm de ser pequenos e discretos para se desenharem e caberem numa tabela.
  Ambientes contínuos (o Gymnasium, por exemplo) pediriam outra representação dos estados,
  e ficam como extensão.
- O DQN treina a rede em cada passo do agente: demora segundos onde o Q-learning demora
  décimas. Numa grelha pequena isso não compensa, mas é o que se quer ver: o mesmo algoritmo,
  com uma rede no lugar da tabela.
- Os runs de agentes aparecem no ecrã Experiências com a tarefa "Controlo". Abri-los leva ao
  ecrã Reinforcement learning.
