"""Registo de experiências: os runs, as suas métricas e os ficheiros dos runs guardados.

- ``store.py``: o contrato ``ExperimentStore``, a ``RunEntry`` e o registo em memória;
- ``sqlite_store.py``: o registo num ficheiro SQLite, que sobrevive a fechar a app;
- ``artifacts.py``: o modelo treinado (joblib) e o treino passo a passo (JSON) de cada run
  guardado;
- ``explanations.py``: os textos do painel "Porquê?" do ecrã Experiências.

Ver a ADR 0020.
"""
