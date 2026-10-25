# Testes da API

Endpoints HTTP e WebSocket (fase 7), com o `TestClient` do FastAPI. Têm o marcador `api` e
correm antes de cada push; o ciclo rápido exclui-os com
`uv run pytest -m "not integration and not api"`.

- `test_contract.py`: cada método público dos serviços tem a sua rota, documentada em
  português, e as respostas passam pelo `to_json`;
- `test_errors.py`: o código HTTP de cada categoria de erro;
- `test_training.py`: treinar em segundo plano, acompanhar, pausar, parar e o WebSocket;
- `test_datasets_and_history.py`: dados, catálogo, análise, laboratório e histórico;
- `test_reinforcement.py`: treinar um agente de reinforcement learning e o histórico;
- `test_texts.py`: o vocabulário e os temas do "Porquê?", que o frontend pede ao arrancar;
- `test_openapi_file.py`: o contrato guardado para o frontend (`ui/web/openapi.json`) é o que
  a API serve;
- `test_automl.py`: pedir aos agentes, seguir a conversa (também pelo WebSocket, sem perder
  uma mensagem) e treinar um candidato à mão.

O critério da fase 7 (carregar um CSV, treinar e receber os snapshots) está em
`tests/integration/test_phase7_api.py`.
