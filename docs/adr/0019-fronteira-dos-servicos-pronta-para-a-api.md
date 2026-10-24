# 0019 — A fronteira dos serviços pronta para a API: dados simples, JSON verificado

- **Estado:** Proposta
- **Data:** 2026-10-24

## Contexto

A interface simples (Streamlit) chama os serviços diretamente, no mesmo processo. A
interface aprimorada (fase 9, React) vai chamá-los através de uma API HTTP (fase 7,
FastAPI). Para essa mudança ser só "pôr rotas à frente dos serviços", o que os serviços
devolvem tem de passar a JSON e ter tipos descritos, de onde saem o OpenAPI e os tipos
em TypeScript.

A revisão do código no fim da fase 4 encontrou três problemas:

- só o `EvaluationReport` sabia passar a JSON (`to_dict`), e cada estrutura nova teria
  de ter a sua conversão;
- o estado de um modelo (`LabModel.state()`, um dicionário) era interpretado à mão em
  três sítios, dois deles na interface (a forma dos pesos, "na binária, um só vetor: o da
  classe positiva");
- algumas regras de domínio estavam na interface (que divisões existem para uma tarefa,
  o que a coluna de referência quer dizer) e o React teria de as repetir.

As alternativas para a serialização eram converter todas as dataclasses em modelos
Pydantic, escrever um conversor próprio, ou usar o `TypeAdapter` do Pydantic, que
descreve e serializa dataclasses normais. Uma experiência mostrou que o `TypeAdapter` já
descrevia todos os tipos devolvidos pelos métodos públicos, exceto os que devolvem
DataFrames ou o modelo treinado.

## Decisão

- **`core/schemas/serialization.py`** é o único sítio que passa resultados a JSON:
  - `to_json` usa o `TypeAdapter` do Pydantic. Os NaN e os infinitos passam a `null`
    (o JSON não os tem, e uma loss que explodiu chegaria como `NaN`, que o navegador
    rejeita), e os valores do NumPy em campos livres passam a números;
  - `json_schema` descreve um tipo.

  As dataclasses ficam como estão: o FastAPI aceita-as como modelos de resposta.
- **`@internal`** marca os métodos que só servem outros serviços (`prepared`, `folds`,
  `table`, `artifacts`). A API não os expõe.
- **Teste de contrato da fronteira** (`tests/contract/test_service_boundary.py`):
  - todos os métodos públicos do `Lab` têm de devolver um tipo que o Pydantic descreve;
  - resultados reais de várias tarefas têm de passar a JSON estrito.

  Um método novo fica testado sem escrever testes.
- **O treino vivo tem um retrato:** `TrainingJob.state()` devolve um `JobState`
  (estado, progresso, passos previstos, erro). É o que a API devolve a quem pergunta pelo
  progresso; o `TrainingJob` fica do lado do servidor.
- **O estado do modelo lê-se num só sítio:** `visualization_data/learned.py` transforma-o
  numa `LearnedView`, que fica no resultado como `ExperimentResult.learned`. Tem:
  - os pesos com o nome de cada saída;
  - as importâncias e as regras da árvore;
  - as tabelas com nomes nas linhas e nas colunas;
  - as propriedades com o nome em português.

  A explicação de uma previsão usa os mesmos pesos.
- **As regras de domínio do ecrã Preparar vêm do backend:** `task_options(tarefa, alvo)`
  diz que coluna fica como referência, o que ela permite comparar e que divisões existem
  (a primeira é a sugerida).
- **Os eventos levam os relatórios inteiros** (`to_json(report)`), e o `to_dict` deixa de
  existir.
- A regra fica escrita em `CONTRIBUTING.md`, para todo o código novo a seguir.

## Consequências

- Na fase 7, cada rota pode devolver o resultado do serviço tal como está, e o OpenAPI
  descreve-o sozinho. O frontend gera daí os tipos.
- Um campo novo que não passe a JSON (uma matriz do NumPy, um DataFrame) faz falhar o
  teste de contrato logo, e não só quando a API o tentar enviar.
- As propriedades calculadas (`MetricResult.formatted`, `ExperimentResult.primary`) não
  vão no JSON, porque só os campos vão. O frontend formata os números ele próprio, e é
  por isso que eles vão em bruto, com o que é preciso para os mostrar (`as_percent`).
- As legendas dos gráficos ainda estão nas páginas do Streamlit. Mudam-se para o backend
  quando os ecrãs forem refeitos, já com o desenho novo à frente.
