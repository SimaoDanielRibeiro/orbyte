# 0003 — Stack enxuta, extensões atrás de contratos

- **Estado:** Aceite
- **Data:** 2026-10-22

## Contexto

O ML Lab é um projeto para aprender ML e para aprender as tecnologias. Muitas bibliotecas
ao mesmo tempo dispersam a atenção, e várias ferramentas com papéis sobrepostos tornam
difícil perceber qual faz o quê.

## Decisão

A stack tem 13 tecnologias no backend e na interface simples (Python, uv, ruff, mypy,
pytest, NumPy, pandas, scikit-learn, PyTorch, Pydantic, SQLite, Streamlit e Plotly), cada
uma com um único papel e introduzida numa fase concreta. FastAPI, Uvicorn e o ecossistema
React só entram nas fases 7 e 9.

As extensões (XGBoost/LightGBM, SHAP, SQLAlchemy + Alembic, MLflow, pandera, UMAP,
Gymnasium, pre-commit/GitHub Actions, Polars) ficam de fora até haver uma razão para as
usar. Cada uma já tem o seu lugar previsto atrás de um contrato, por isso pode entrar sem
refazer nada.

## Consequências

- Uma ou duas tecnologias novas por fase, aprendidas uma de cada vez.
- Nenhuma porta fica fechada: os contratos (`ExperimentStore`, `LabModel`, `DataSource`...)
  são os pontos de entrada das extensões.
- Algumas coisas são feitas à mão com a biblioteca padrão: configuração e `.env`, logs
  estruturados, registo de componentes.
- As verificações antes de cada push (`ruff`, `mypy`, `pytest`) são manuais até entrar o
  pre-commit ou um CI.
