# Testes da API

A partir da fase 7: endpoints HTTP e WebSocket. Têm o marcador `api` e correm antes de cada
push; o ciclo rápido exclui-os com `uv run pytest -m "not integration and not api"`.
