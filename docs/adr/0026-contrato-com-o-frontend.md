# 0026 — O contrato com o frontend: tipos exatos, um vocabulário e o OpenAPI guardado

- **Estado:** Proposta
- **Data:** 2026-10-25

## Contexto

A fase 10 refaz a interface em React, por cima da API (ADRs 0019 e 0022). Antes de começar,
uma revisão do contrato pela perspetiva de quem o vai consumir, com os tipos TypeScript
gerados do OpenAPI, encontrou cinco problemas:

1. **Tipos em duplicado.** O OpenAPI tinha `ExperimentConfig-Input` e
   `ExperimentConfig-Output` (e o mesmo para `PreparationConfig`, `CleaningStep` e
   `AutoMLRequest`). A causa: as dataclasses da camada de dados herdam a configuração do
   Pydantic do modelo onde aparecem. O `CleaningStep` recusava campos a mais dentro de um
   `ExperimentConfig` e aceitava-os dentro de uma `CleaningSuggestion`; daí, dois esquemas.
2. **Respostas com campos "opcionais" que vêm sempre.** O FastAPI marca como opcional, nas
   respostas, todo o campo com valor por omissão (`history`, `animation`…), mas o `to_json`
   manda sempre todos. No TypeScript, seriam `?` e `??` por todo o lado.
3. **Textos que não chegavam ao frontend.** Os enums viajam pelo valor
   (`"binary_classification"`); os nomes em português estão no `label` de cada um, que o
   Streamlit lê e o JSON não leva. O frontend teria de os escrever outra vez, contra a regra
   5 do `CONTRIBUTING.md`.
4. **Regras de domínio em propriedades.** Qual é a métrica principal de um run
   (`ExperimentResult.primary`) e se os dados podem seguir (`ValidationReport.is_usable`)
   eram propriedades do Python, fora do JSON: o frontend teria de as decidir.
5. **Sem contrato fixo.** Para gerar os tipos, o frontend precisava da API a correr, e uma
   mudança no backend não deixava rasto do lado do contrato.

## Decisão

1. **As dataclasses dos pedidos têm a sua própria configuração**, dada na fronteira
   (`schemas/experiment.py`, com o `with_config` do Pydantic): `extra="forbid"` em
   `SourceSpec`, `PreparationConfig`, `CleaningStep`, `SplitConfig`, `TransformConfig` e
   `ProjectionConfig`. A camada de dados continua sem conhecer o Pydantic (ADR 0011), e cada
   tipo tem um só esquema, em qualquer contexto.
2. **O OpenAPI promete o que a API manda** (`api/openapi.py`): nos tipos que só aparecem em
   respostas, todos os campos são obrigatórios. Os tipos que também se enviam ficam como
   estão, porque num pedido um campo com valor por omissão pode mesmo faltar.
3. **O vocabulário** (`services/vocabulary.py`, `GET /vocabulary`): o nome a mostrar de cada
   valor de cada enum que a API devolve, pelo nome do tipo, e o que muda com cada tarefa
   (o paradigma, a métrica principal, o modelo sugerido, se os agentes trabalham nela). Os
   enums que não tinham `label` passaram a ter (`Paradigm`, `NoticeLevel`, `CellKind`,
   `AgentRole`). `GET /explanations` dá todos os temas do "Porquê?" de uma vez.
4. **As regras de domínio que estavam em propriedades viajam**: `ValidationReport.is_usable`
   passou a campo (calculado sempre a partir dos avisos); a métrica principal está no
   vocabulário. O que é só uma conta com campos que já vêm (a parte de nulos, o relatório
   principal) fica documentado no guia do frontend.
5. **O contrato fica guardado** em `ui/web/openapi.json` (`scripts/export_openapi.py`), com
   um teste que falha quando fica desatualizado. O guia do contrato está em
   `ui/web/README.md`.

Os testes de contrato guardam tudo isto: um esquema por tipo e com nome simples; as
respostas completas; cada enum devolvido com um nome para cada valor; e cada resultado real
a voltar a validar contra o próprio tipo depois de passar a JSON (um NaN que virasse `null`
num campo `number` mentia aos tipos TypeScript).

## Consequências

- O frontend gera os tipos a partir de um ficheiro, sem a API a correr, e os tipos dizem a
  verdade: nada opcional que venha sempre, nenhum tipo em duplicado.
- Os textos (os nomes dos valores, os temas do "Porquê?", as fichas, os avisos) continuam a
  escrever-se uma vez, no backend, e o frontend pede-os ao arrancar.
- Um enum novo, ou um valor novo num enum, não passa os testes sem nome no vocabulário.
- Uma mudança num serviço, numa rota ou num tipo obriga a voltar a gerar o
  `openapi.json`. É de propósito: a diferença no Git mostra o que muda para o frontend.
- Os corpos dos pedidos recusam campos desconhecidos em todo o lado (antes, uma `SourceSpec`
  sozinha aceitava-os).
- Ficou de fora, como opção: comprimir as respostas (a API corre no mesmo computador, e
  comprimir 2 MB custa mais tempo do que poupa) e documentar os WebSockets no próprio OpenAPI
  (o FastAPI não o faz; estão descritos no guia).
