# 0041 — A procura dos agentes fica aberta, e a comparação arruma-se por grupos

- **Estado:** Proposta
- **Data:** 2026-11-01

## Contexto

Três coisas na interface web atrapalhavam quem a usa:

1. **Os Agentes esqueciam a procura.** Quem corria os agentes, ia ao Treinar e avaliar ver o
   vencedor e voltava aos Agentes, via outra vez o formulário do pedido. Para rever a procura
   tinha de carregar em "Ver a última procura". O id da procura já estava guardado na sessão;
   só não era o que o ecrã mostrava.
2. **A comparação de runs ficava baralhada.** Com runs de tarefas diferentes (uma regressão e
   uma classificação), a tabela das métricas enchia-se de traços. Em "O que muda", os
   hiperparâmetros de um modelo apareciam a meio das escolhas da preparação, porque as linhas
   iam pela ordem em que surgiam em cada run.
3. **Os eixos cortavam os números grandes.** A margem da esquerda dos gráficos era fixa (48 ou
   60 px). Uma loss de "12 000 000 000" não cabia, e o nome do eixo ("Loss") caía por cima
   dos números.

## Decisão

- **Uma procura aberta é o ecrã Agentes.** Com um id de procura na sessão, a rota `/agentes`
  passa logo para `/agentes/$searchId` (como o Explicar e o Guardar fazem com o run aberto). O
  pedido só volta a aparecer com o botão **"Nova procura"**, que esquece a procura. Abrir outro
  dataset também a esquece, e o mesmo acontece a uma procura que a API já não tem (senão os
  Agentes voltavam sempre ao aviso).
- **O grupo de cada escolha vem do backend.** `ConfigDifference` ganha `group` (o enum
  `ChoiceGroup`: o run, a preparação, o ambiente, os hiperparâmetros, com nome no vocabulário)
  e `name` (o nome dentro do grupo). As diferenças vêm ordenadas por grupo. Dizer a que parte do
  run pertence uma escolha é uma regra de domínio (regra 4 do `CONTRIBUTING.md`), por isso a interface
  não a deduz do texto.
- **As métricas arrumam-se por quem as tem** ("Em comum", "Só A", "Só B e C"). Isto é só
  apresentação, por isso fica na interface. No gráfico de pontos entram só as métricas que mais
  de um run tem. Por cima de tudo fica um cartão por run, com a cor, o modelo, os dados e a
  métrica principal.
- **A margem da esquerda mede-se pelos números do eixo** (`leftMargin`, em
  `components/charts/margins.ts`). As marcas de y calculam-se antes do desenho, porque dependem
  só do domínio e não da altura.

## Consequências

- Sai-se e volta-se aos Agentes sem perder nada. O preço é que, para fazer outro pedido, é
  preciso carregar em "Nova procura", e as ligações para `/agentes` (a do Explorar, por exemplo)
  levam à procura aberta e não ao pedido.
- O JSON da comparação tem dois campos novos, e o `openapi.json` foi gerado outra vez. O campo
  `label` continua igual, e as leituras do backend ("só a seed muda") continuam a usá-lo.
- Os gráficos de linhas e de dispersão ajustam a margem sozinhos. Os outros gráficos com margem
  fixa (o contorno, o campo de decisão, o dendrograma) só mostram valores pequenos e ficaram
  como estavam.
