# 0022 — A API: rotas finas por cima dos serviços, e o treino acompanhado pelo mesmo pedido

- **Estado:** Proposta
- **Data:** 2026-10-24

## Contexto

A fase 7 põe os serviços do ML Lab a falar HTTP, para a interface aprimorada (fase 10, React)
os usar. Desde a ADR 0019, tudo o que os ecrãs precisam já existe como métodos públicos dos
serviços, que devolvem dados simples e passam a JSON. Faltava decidir quatro coisas:

- **como evitar que a API e os serviços se afastem:** cada método novo num serviço tem de
  chegar ao frontend sem ninguém se esquecer da rota;
- **como acompanhar um treino:** o treino corre num fio de execução próprio (ADR 0014). O
  HTTP é feito de pedidos soltos, e o frontend quer ver o treino a acontecer;
- **como os erros viram códigos HTTP:** os serviços lançam subclasses de `MLLabError` com
  mensagens em português (regra 8 do `CONTRIBUTING.md`);
- **onde fica o estado:** o ML Lab é local e de um só utilizador.

## Decisão

- **Uma rota por método público dos serviços, com o nome do método.** O id de cada operação
  no OpenAPI é `<serviço>_<método>` (`datasets_samples`, `experiments_start`). É o nome que o
  cliente gerado para o React vai ter. Um teste de contrato (`tests/api/test_contract.py`)
  falha se algum método público ficar sem rota; as exceções estão numa lista, com a razão
  (`experiments.run`: a API treina sempre em segundo plano).
- **As rotas não têm lógica.** Chamam o serviço e devolvem o que ele devolve. O
  `ServiceRouter` faz três coisas por todas:
  - passa a resposta a JSON pelo `to_json` (ADR 0019): os NaN viram `null` e o NumPy vira
    números;
  - tira o resumo da rota em `/docs` da primeira linha da docstring, em português;
  - deixa o FastAPI descrever cada resposta pelo tipo que a rota anuncia.
- **Os treinos ficam nos serviços.** O `ExperimentService` guarda os treinos que lança num
  `JobRegistry` e ganha `job`, `pause`, `resume`, `cancel`, `updates` e `result`. A API não
  guarda estado nenhum. A interface simples passou a usar o mesmo `updates`.
- **Acompanhar o treino é sempre o mesmo pedido.** `updates(run_id, after_step,
  after_snapshot)` devolve o estado, os passos novos e a fronteira do snapshot novo (um
  `TrainingUpdate`). Há duas maneiras de o receber:
  - `GET /experiments/{run_id}/updates`, perguntando de vez em quando;
  - o WebSocket `/experiments/{run_id}/stream`, que pergunta ao serviço de 0,25 em 0,25 s e
    envia só o que mudou.

  O WebSocket pergunta em vez de receber os eventos do `EventBus`, porque passar eventos de
  um fio de execução para o ciclo assíncrono do servidor pede filas e cuidado. Perguntar
  também junta numa só mensagem os milhares de passos que um modelo rápido dá por segundo.
- **O código HTTP vem da categoria do erro**, pela ordem:
  - `TrainingStateError` → 409;
  - `DataError` e `ValueError` → 422;
  - `LookupError` → 404;
  - `TrackingError` → 500;
  - outro `MLLabError` → 400.

  O corpo é sempre `{code, message, problems}`. O `code` é o nome do erro em `snake_case`
  (`run_not_found`). Um pedido com a forma errada dá `invalid_request`, com uma frase em
  português por problema. Para as categorias ficarem certas, entram erros novos:
  - `DatasetNotFoundError` e `RunNotInMemoryError` (404);
  - `JobNotFoundError` (404);
  - `TrainingInProgressError` e `TrainingFailedError` (409).
- **Um `Lab` por processo.** É criado com a aplicação, e os testes passam o seu.
- **A API só escuta neste computador.** O endereço por omissão é `127.0.0.1` e não há
  autenticação. O CORS deixa entrar o servidor do Vite (fase 10); tudo isto se muda nas
  definições.
- **Dependências:**
  - o FastAPI;
  - o Uvicorn, com o `websockets`, porque o Uvicorn sozinho não aceita WebSockets;
  - o `python-multipart`, para os uploads;
  - o `httpx2`, só nos testes, porque é o que o `TestClient` do Starlette 1.9 pede.

## Consequências

- O frontend gera os tipos e as funções a partir de `/openapi.json`, com nomes iguais aos dos
  serviços. A fase 10 só tem de desenhar.
- Um método novo num serviço obriga a escrever a rota; o teste lembra-o.
- O progresso chega com até 0,25 s de atraso, o que chega para um ecrã.
- Os treinos em curso vivem na memória do servidor: reiniciá-lo perde-os. Os runs que
  terminaram estão no histórico (ADR 0020).
- Dois separadores do navegador partilham os datasets e os treinos. Um servidor para várias
  pessoas precisaria de sessões e de autenticação, o que fica fora do âmbito.
