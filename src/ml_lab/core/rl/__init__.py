"""Reinforcement learning: um agente aprende a agir num ambiente, pelas recompensas (fase 8).

Não há dataset: o agente aprende a jogar. Em cada passo vê o estado, escolhe uma ação e
recebe uma recompensa; com muitos episódios, aprende que ações levam a mais recompensa.

- ``environments/``: os ambientes (a grelha), registados em ``ENVIRONMENTS``;
- ``agents/``: os agentes (Q-learning em NumPy, DQN em PyTorch), registados em ``AGENTS``,
  com o ciclo de episódios que partilham;
- ``evaluation.py``: as métricas de um agente, no treino e com a política aprendida;
- ``explanations.py``: os textos do painel "Porquê?" do ecrã Reinforcement learning.

Ver a ADR 0024.
"""
