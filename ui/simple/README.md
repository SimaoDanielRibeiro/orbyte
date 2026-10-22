# Interface simples (Streamlit)

Cresce a par do backend: cada fase dá conteúdo a mais ecrãs. Abre-se a partir da raiz do
projeto com `uv run streamlit run ui/simple/app.py`.

| Ficheiro ou pasta | O que tem |
|---|---|
| `app.py` | Entrada: prepara logs e pastas, navegação no topo, barra lateral |
| `pages/` | Um ecrã por etapa (Dados, Explorar, Preparar…) |
| `components/` | Blocos reutilizáveis; `screens.py` é o catálogo dos ecrãs |
| `charts/` | Desenha com Plotly os dados de `core/visualization_data` (fase 1) |

## Regras

1. Nenhuma lógica de ML ou de dados vive aqui: a interface só pede aos serviços e mostra.
2. Do núcleo, só importa `ml_lab.core.services` e `ml_lab.core.schemas` (verificado em
   `tests/architecture/`). Pode usar `ml_lab.config` e `ml_lab.logging`.
3. Os gráficos usam os números de `visualization_data`, os formulários vêm das fichas
   técnicas dos modelos e os textos didáticos vêm do backend, para a interface aprimorada
   reaproveitar tudo.

O Streamlit volta a correr o script a cada interação: o estado vive em `st.session_state` e
os recursos pesados em `st.cache_resource`.

Para acrescentar um ecrã: cria `pages/<slug>.py` e acrescenta o `Screen` correspondente em
`components/screens.py`. O teste `tests/integration/test_simple_ui.py` abre todos os ecrãs.
