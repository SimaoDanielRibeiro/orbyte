# API (`api/`)

FastAPI: os serviços do ML Lab por HTTP, para a interface web (fase 10). A
documentação interativa é gerada automaticamente em `/docs`, a partir dos tipos que os
serviços devolvem (ADR 0022).

- **Pode importar:** `ml_lab.core.services` e `ml_lab.core.schemas` (e `config`, `logging`,
  `process`).
- **Nunca importa:** modelos, dados, treino ou base de dados diretamente, nem a interface.

## Abrir

```bash
uv run python -m ml_lab.api
```

A API fica em <http://127.0.0.1:8000> e a documentação em <http://127.0.0.1:8000/docs>. O
endereço, a porta e as origens do CORS mudam-se nas definições (`ML_LAB_API_*`, ver
`.env.example`).

## Como está feita

| Ficheiro | O que faz |
|---|---|
| `app.py` | `create_app`: a aplicação, com um `Lab` por processo, o CORS e os erros |
| `routing.py` | `ServiceRouter`: respostas pelo `to_json`, resumos das docstrings, ids `<serviço>_<método>` |
| `errors.py` | O código HTTP de cada categoria de erro e o corpo `ApiError` |
| `requests.py` | Os corpos dos pedidos que juntam vários valores |
| `stream.py` | O WebSocket: o mesmo `updates` dos serviços, de 0,25 em 0,25 s |
| `openapi.py` | O OpenAPI com as respostas completas: nos tipos que só se recebem, todos os campos são obrigatórios (ADR 0026) |
| `routes/` | Uma rota por método público de cada serviço, com o nome do método, e os textos (`/vocabulary`, `/explanations`) |

Para expor um método novo de um serviço, escreve a rota em `routes/<serviço>.py` com o nome
do método. `tests/api/test_contract.py` falha enquanto faltar. Depois, volta a gerar o
contrato do frontend (`uv run python scripts/export_openapi.py`):
`tests/api/test_openapi_file.py` falha enquanto `ui/web/openapi.json` estiver desatualizado.
O guia do contrato, para quem faz o frontend, está em `ui/web/README.md`.

## Treinar e acompanhar o treino

1. `POST /experiments` com a configuração começa o treino em segundo plano e responde logo
   (202) com o estado: o `run_id`, o estado e o progresso.
2. Para acompanhar, há duas maneiras de receber o mesmo `TrainingUpdate` (o estado, os passos
   novos e a fronteira do último snapshot):
   - `GET /experiments/{run_id}/updates?after_step=N&after_snapshot=M`, dizendo o que já se
     tem;
   - o WebSocket `/experiments/{run_id}/stream`, que envia mensagens
     `{"kind": "update", "data": <TrainingUpdate>}` até o treino terminar, ou
     `{"kind": "error", "data": <ApiError>}` se o treino não existir.
3. `POST .../pause`, `.../resume` e `.../cancel` controlam o treino.
4. `GET /experiments/{run_id}/result` dá o resultado quando o estado for `finished` (antes
   disso, responde 409).

Os agentes de reinforcement learning (`/reinforcement`) seguem-se da mesma maneira.

## Pedir aos agentes (o modo automático)

1. `GET /automl/options?task_type=...` diz o que se pode pedir: as métricas, os agentes e o
   esquema do tamanho da procura (ou porque é que a tarefa fica no modo manual).
2. `POST /automl/plan` com um `AutoMLRequest` mostra o que o agente de dados vai decidir,
   sem começar.
3. `POST /automl` começa a procura em segundo plano e responde logo (202) com o estado; o
   `run_id` do estado é o id da procura (`search-...`). Cada candidato medido é um passo.
4. `GET /automl/{search_id}/updates?after_step=N&after_message=M` traz os candidatos e as
   mensagens novas da conversa dos agentes, e o melhor até agora. O WebSocket
   `/automl/{search_id}/stream` envia o mesmo, sozinho.
5. `GET /automl/{search_id}/result` dá o quadro, a conversa e o `run_id` do vencedor, que é
   um run normal (`/experiments/{run_id}/result`, `/history/runs/{run_id}`).
6. `POST /automl/{search_id}/candidates/{number}/train` treina um candidato à mão; o treino
   acompanha-se nas rotas de `/experiments`.

## Erros

O corpo é sempre `{"code": ..., "message": ..., "problems": [...]}`. O `code` é estável (o
nome do erro em `snake_case`), e a `message` diz em português o que fazer.

| Categoria do erro | Código | Exemplo |
|---|---|---|
| `TrainingStateError` | 409 | `training_in_progress`: o resultado ainda não existe |
| `DataError`, `ValueError` | 422 | `hyperparameter`, `column_not_found`, `invalid_request` |
| `LookupError` | 404 | `run_not_found`, `dataset_not_found`, `job_not_found`, `search_not_found`, `candidate_not_found` |
| `TrackingError` | 500 | O registo de experiências falhou |
| outro `MLLabError` | 400 | `model`: a pergunta não serve para esta tarefa |
