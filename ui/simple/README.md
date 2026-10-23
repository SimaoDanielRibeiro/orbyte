# Interface simples (Streamlit)

Cresce a par do backend: cada fase dá conteúdo a mais ecrãs. Abre-se a partir da raiz do
projeto com `uv run streamlit run ui/simple/app.py`.

| Ficheiro ou pasta | O que tem |
|---|---|
| `app.py` | Entrada: prepara logs e pastas, navegação no topo, barra lateral |
| `pages/` | Um ecrã por etapa (Dados, Explorar, Preparar…) |
| `components/` | Blocos reutilizáveis: estado da sessão, avisos, formulários, tabelas; `screens.py` é o catálogo dos ecrãs |
| `charts/` | Gráficos Plotly desenhados a partir dos números que os serviços devolvem |

## Regras

1. Nenhuma lógica de ML ou de dados vive aqui: a interface só pede aos serviços e mostra.
2. Do núcleo, só importa `ml_lab.core.services` e `ml_lab.core.schemas` (verificado em
   `tests/architecture/`). Pode usar `ml_lab.config`, `ml_lab.logging` e `ml_lab.formatting`.
3. Os gráficos usam os números calculados no backend, os formulários são gerados a partir de
   JSON Schema (`components/forms.py`) e os textos didáticos vêm do backend, para a interface
   aprimorada reaproveitar tudo.

## Estado

O Streamlit volta a correr o script a cada interação: o estado vive em `st.session_state`
(`components/state.py`: o `Lab` com os serviços, o dataset atual, a última preparação). Os
controlos têm `key` e `persist_state="session"`, para as escolhas sobreviverem à mudança de
ecrã. Cada ecrã está numa função `main()`, que pode terminar cedo (por exemplo, sem dataset).

A barra lateral é desenhada depois do ecrã, para já mostrar o que ele acabou de mudar.

Para acrescentar um ecrã: cria `pages/<slug>.py` e acrescenta o `Screen` correspondente em
`components/screens.py`. Os testes em `tests/integration/` abrem todos os ecrãs e percorrem os
fluxos de cada fase.
