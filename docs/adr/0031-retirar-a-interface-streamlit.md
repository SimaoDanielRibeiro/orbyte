# 0031 — A interface web é a única: a interface Streamlit sai

- **Estado:** Proposta
- **Data:** 2026-10-28

## Contexto

A ADR 0005 decidiu começar por uma interface simples, em Streamlit e Plotly, que crescesse a
par do backend, e refazer a interface no fim em React. A fase 10 refez os 11 ecrãs na web, e
o critério da etapa 10.8 cumpriu-se: cada ecrã da interface simples tem o seu equivalente,
provado pelos testes de ponta a ponta. Ficou combinado decidir então o que fazer à interface
simples.

Manter as duas tinha custos:

- **Tudo duas vezes.** Cada texto didático e cada controlo novo tinham de chegar às duas
  interfaces (o teste dos temas procurava os nomes nos dois códigos), e cada fluxo testava-se
  duas vezes: com o `AppTest` do Streamlit e com o Playwright. Os testes de integração de
  cada fase misturavam os serviços com os ecrãs do Streamlit.
- **Duas portas para o backend.** A interface simples chamava os serviços no mesmo processo;
  a web passa pela API. As regras de arquitetura tinham de vigiar as duas.
- **Dependências que só ela usava.** O Streamlit e o Plotly, num grupo próprio do uv, que o
  `uv sync` instalava sempre.

E o trabalho muda de sítio: daqui para a frente, é na interface web que se mexe para a pôr
como se quer. A interface simples deixava de ser usada, mas cada mudança no backend
continuava a ter de a manter a funcionar.

## Decisão

A interface web (`ui/web`) passa a ser a única interface do ML Lab. Saem:

- a pasta `ui/simple/` (as páginas, os gráficos em Plotly e os componentes), a configuração
  do Streamlit (`.streamlit/`) e o seu arranque no `launch.json` das pré-visualizações;
- o grupo de dependências `ui` (o Streamlit e o Plotly) e o que só existia para ele na
  configuração do ruff e do mypy;
- dos testes de integração das fases, só as partes que abriam os ecrãs do Streamlit
  (`AppTest`). Os testes dos serviços ficam; o que as outras partes provavam na interface,
  provam-no agora os testes de ponta a ponta da web.

Do backend não sai nenhum serviço, rota ou tipo: procurou-se o código que ficava sem uso sem
a interface simples, e não havia nenhum (o que ela usava, a API também usa). As regras que o
`CONTRIBUTING.md` dá ao backend continuam: a API é a única porta, e os textos e as regras de domínio
ficam no backend. Esta ADR substitui a 0005.

## Consequências

- Uma só interface para manter, testar e documentar: um texto ou um controlo novo escreve-se
  uma vez e chega a um sítio.
- O `uv sync` instala menos (sem o Streamlit, o Plotly e o que eles trazem), e as regras de
  arquitetura ficam mais simples: só a API chega ao núcleo, e só pelos serviços e pelos
  esquemas.
- A interface já não chama os serviços no mesmo processo: tudo passa pela API, como o contrato
  com o frontend previa (ADR 0026). Para a usar, a API tem de estar a correr
  (`uv run python scripts/dev.py` arranca as duas).
- Perde-se uma interface escrita em Python, mais curta e mais fácil de ler para quem aprende.
  Fica no histórico do Git: a última versão completa está no commit `dcdcc5e`, no ramo
  `feat/interface-web`.
- Os testes de ponta a ponta da web passam a ser a única prova de que os ecrãs funcionam.
