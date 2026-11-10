# Como trabalhar no Orbyte

Laboratório de machine learning feito para aprender, fase a fase. Chama-se Orbyte (antes ML Lab;
o pacote Python continua `ml_lab`, ADR 0043). O plano completo está em `docs/architecture.md`;
as decisões e as suas razões, em `docs/adr/`.

## Como trabalhar

- Escreve em português de Portugal (documentação, ADRs, docstrings, comentários, registos e
  mensagens de commit); os nomes no código ficam em inglês. Docstrings no estilo Google.
- **O site está todo em inglês** (ADR 0043): tudo o que a interface mostra, venha dela ou do
  backend — nomes a mostrar (`label`), avisos, erros, temas, explicações, a conversa dos agentes,
  a história do treino, as fichas de `docs/algorithms/` e os números do `formatting.py`. Os
  identificadores não se traduzem (ids, códigos dos avisos, os mapas `simples`/`penhasco`).
- O projeto é para aprender: explica o porquê das escolhas, não só o quê.
- Trabalha fase a fase, pelo roadmap de `docs/architecture.md`. Uma fase só termina
  quando o critério "Pronto quando" se cumpre, `ruff`, `mypy` e `pytest` passam e as
  fichas didáticas dos algoritmos novos estão em `docs/algorithms/`.
- Cada decisão que não seja óbvia fica numa ADR em `docs/adr/` (o modelo está no
  `README.md` dessa pasta) e nas notas da fase em `docs/architecture.md`.
- Os commits vão em ramos `feat/...` ou `fix/...`, no formato Conventional Commits.

## Comandos

```bash
uv sync                                          # instalar
uv run ruff format .                             # formatar
uv run ruff check .                              # lint
uv run mypy                                      # tipos (strict)
uv run pytest                                    # todos os testes
uv run pytest -m "not integration and not api"   # ciclo rápido
uv run python scripts/dev.py                     # a API e a interface juntas
uv run python -m ml_lab.api                      # só a API (documentação em /docs)
uv run python scripts/export_openapi.py          # o contrato do frontend (ui/web/openapi.json)
```

A interface é a web (fase 10, em `ui/web`); a simples, em Streamlit, saiu (ADR 0031). Dentro
de `ui/web`, com a API a correr:

```bash
npm install                                      # instalar
npm run dev                                      # a interface (http://localhost:5173)
npm run check                                    # verificar: gera o cliente, lint, tipos e testes
npm run e2e                                      # ponta a ponta (API numa pasta temporária)
```

Os avisos contam como erros nos testes: resolve-os em vez de os silenciar.

## Arquitetura (as regras têm testes em `tests/architecture/`)

- `core/` nunca importa `api/` nem bibliotecas de interface ou de gráficos;
  `core/foundation/` não importa nada do projeto; o pandas só existe em `core/data/`.
- A API só importa `ml_lab.core.services` e `ml_lab.core.schemas` (e `config`, `logging`,
  `formatting`, `process`); a interface web só fala com a API, por HTTP e WebSocket.
- Os componentes registam-se (`MODELS`, `METRICS`, `EVALUATORS`, `GENERATORS`, `AGENTS`,
  `ENVIRONMENTS`…). Um algoritmo novo é um ficheiro novo mais a sua ficha, e os testes de
  contrato testam-no sozinho. Ver "Manutenção: como adicionar componentes" em
  `docs/architecture.md`.
- O PyTorch só existe nas redes (`core/models/neural/`, `torch_adapter.py` e o DQN em
  `core/rl/agents/dqn.py`); o SQLite e o joblib só em `core/tracking/`.

## O backend tem de ficar pronto para o frontend

A interface é a web, em React (fase 10), que fala com o núcleo só através da API HTTP
(fase 7, FastAPI); a simples, em Streamlit, saiu quando a web ficou com tudo (ADR 0031).
**Todo o código do backend escreve-se a pensar nisso**, para que a interface só tenha de
desenhar (ADR 0019):

1. **Os serviços são a única porta.** Tudo o que um ecrã precisa existe como método
   público de um serviço em `core/services/`. Se a interface precisa de algo novo, cria-se
   no serviço, não na página.
2. **Os serviços devolvem dados simples.** Dataclasses `frozen` ou modelos Pydantic com
   texto, números, booleanos, `None`, enums, tuplos e dicionários. Nunca DataFrames,
   matrizes do NumPy, modelos treinados ou objetos com estado. Os números vão em bruto,
   com o que é preciso para os mostrar (por exemplo, `as_percent`).
3. **Tudo passa a JSON.** `tests/contract/test_service_boundary.py` exige que cada
   método público devolva um tipo que o Pydantic sabe descrever, e resultados reais que
   passam a JSON estrito e voltam a validar contra o próprio tipo. Para serializar, usa
   `ml_lab.core.schemas.serialization.to_json` (os NaN passam a `null`). Um método que só
   serve outros serviços (devolve DataFrames ou o modelo) leva `@internal`. As propriedades
   (`@property`) não vão no JSON: o que o frontend precisa de saber fica num campo.
4. **As regras de domínio ficam no backend.** A interface não decide nada que seja do
   domínio: que divisões existem para uma tarefa, o que significa o estado de um modelo,
   qual é a classe anómala. Pede-o a um serviço (por exemplo
   `DatasetService.task_options` ou `ExperimentResult.learned`). A interface decide só a
   apresentação: a disposição, os separadores e as cores.
5. **Os textos didáticos vêm do backend:** fichas, explicações das métricas (com a fórmula e
   os termos dela num run), temas do painel "Porquê?", avisos (`Notice`, com um código
   estável) e o nome a mostrar de cada valor (o `label` de cada enum, que o frontend recebe
   no vocabulário, `GET /vocabulary`; `tests/contract/test_vocabulary.py` falha se um enum
   devolvido não tiver nome). Assim só se escrevem uma vez. Os textos novos vão para o
   backend. O "como se lê" de cada gráfico também é um tema, sem cores (as cores são da
   interface), e a interface pede-o pelo nome escrito por inteiro
   (`tests/contract/test_topics_in_use.py` falha se não existir; ADR 0030).
6. **O trabalho longo corre em segundo plano**, com um estado serializável
   (`TrainingJob.state()` devolve um `JobState`). O serviço que o lança guarda-o num
   `JobRegistry` e expõe `job`, `pause`, `resume`, `cancel`, `updates` e `result` pelo id
   do run: é isso que a API devolve a quem acompanha o treino (ADR 0022).
7. **Identificadores estáveis, em texto:** o id do dataset, o id do run, o nome do
   modelo no registo, o código de cada aviso. É com eles que o frontend volta a pedir as
   coisas.
8. **Os erros são subclasses de `MLLabError`**, com uma mensagem em português que diz o
   que fazer. A API traduz a categoria num código HTTP, por isso escolhe a subclasse certa:
   `TrainingStateError` → 409, `DataError` e `ValueError` → 422, `LookupError` → 404,
   `TrackingError` → 500, outro `MLLabError` → 400 (`api/errors.py`).
9. **Cada método público de um serviço tem a sua rota** em `api/routes/<serviço>.py`, com
   o nome do método (`tests/api/test_contract.py` falha enquanto faltar).
10. **O contrato fica guardado** em `ui/web/openapi.json`: depois de mudar um serviço, uma
    rota ou um tipo devolvido, volta a gerá-lo (`scripts/export_openapi.py`;
    `tests/api/test_openapi_file.py` avisa). O guia do contrato está em `ui/web/README.md`
    (ADR 0026).
