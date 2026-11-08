# 0053 — O analista: um agente com ferramentas que fazem as contas, e uma resposta verificada

- **Estado:** Proposta; mudada pela [0054](0054-o-analista-numa-conversa-o-quadro-inteiro-e-pronto-para-outros-computadores.md)
  (o quadro leva todos os candidatos, um por linha, e o painel passa a uma conversa)
- **Data:** 2026-11-08
- **Segue:** a [0042](0042-agentes-de-ia-com-o-ollama.md) (a IA propõe e explica; as regras decidem
  e medem), na etapa 13.3

## Contexto

A etapa 13.3 junta à procura dos agentes um **analista**: um agente de IA que responde a
perguntas sobre uma procura acabada ("Why did the winner win?"), usando ferramentas (o quadro, um
candidato, dois lado a lado, o plano dos dados). É a outra maneira de pôr um modelo de linguagem
a trabalhar: o agente de IA da afinação (13.2) recebe tudo num texto e responde num formato fixo;
o analista começa só com a pergunta e decide o que precisa de ler.

As forças em jogo:

- **a resposta tem de vir da procura**, e não do que o modelo sabe de machine learning em geral;
  o analista não pode medir nada nem ver o teste para lá do que o resultado já mostra;
- **um modelo de 8B engana-se de maneiras previsíveis.** Antes de fechar o desenho, o analista
  correu com o `qwen3:8b` a sério, numa procura do Titanic, e:
  - num quadro de dez linhas, atribuiu ao vencedor os números da referência;
  - fez contas que ninguém pediu ("a diferença é 0,2"), e erradas;
  - comparou o teste do vencedor com um "teste" da referência, que não existe;
  - com tudo no quadro, respondeu só com ele, e leu mal o que lá estava;
- **a conversa do analista é maior** do que a do agente de IA: as respostas das ferramentas e o
  raciocínio de cada vez. O Ollama carrega os modelos com um contexto de 4096 tokens, e quando a
  conversa passa disso corta o princípio sem avisar;
- **os testes não podem depender do Ollama**, e o laboratório funciona sem ele (ADR 0042);
- **cada vez do modelo demora segundos** (uma pergunta inteira, meio minuto a um minuto numa
  RTX 4060): o ecrã não pode ficar parado, e tem de se ver o agente a trabalhar.

## Decisão

1. **Um agente com ferramentas, num ciclo à vista**: pensa → pede uma ferramenta → lê o
   resultado → volta a pensar → responde (`core/automl/analyst.py`). Cada passo fica num
   `AnalystLog` (o raciocínio, cada ferramenta com os argumentos, o resumo e o JSON que o modelo
   leu, e a resposta). Ao fim de seis voltas com ferramentas, a última vai sem elas e com o
   pedido de responder: um modelo pode ficar a pedir ferramentas sem fim.
2. **As ferramentas leem, não medem** (`core/automl/analyst_tools.py`). Leem um `SearchFacts`, que
   o serviço monta do resultado da procura e da conversa dos agentes; do teste, só a nota do
   vencedor. Uma regra de arquitetura confirma que os dois ficheiros do analista não chegam aos
   modelos, aos dados, à avaliação, ao treino nem ao coordenador. Um pedido que não serve (um
   candidato que não existe) volta ao modelo com a razão, como o resultado da ferramenta.
3. **As ferramentas fazem as contas e as leituras, e o quadro é um índice.** A distância entre
   dois candidatos, o empate na prática (a distância menor do que o desvio padrão do melhor, a
   regra do aviso `close_race`), o desempate, os folds que cada um ganhou e o "decora o treino"
   (a regra do `Objective`) vêm escritos como o ecrã os mostra. O quadro traz o vencedor à parte,
   quanto cada candidato fica atrás do primeiro e as chamadas seguintes já escritas
   (`compare_candidates(first=3, second=10)`); o treino, o empate e os folds ficam nas ferramentas
   dos pormenores. Quem não foi ao teste diz que não foi. Cada uma destas escolhas corrigiu um dos
   erros do `qwen3:8b`, e com elas ele passou a usar duas ferramentas na pergunta principal.
4. **A resposta passa por uma verificação.** Cada número da resposta tem de estar no que as
   ferramentas devolveram (ou na pergunta): os que não estão aparecem num aviso
   (`unsupported_numbers`); se estão todos, diz-se (`numbers_checked`). Também se avisa uma
   resposta sem ferramenta nenhuma (`no_tools_used`), uma resposta forçada (`forced_answer`) e
   uma pergunta parada (`stopped_early`). A IA explica; as regras medem o que ela disse.
5. **As respostas guardam-se, para repetir e para testar.** O `RecordedReply` ganha as ferramentas
   pedidas (`tool_calls`), e a pergunta (`AnalystRequest`) aceita as respostas de uma pergunta
   anterior (`replies`): com elas, o `ScriptedModel` responde outra vez tal e qual, sem o Ollama.
   A resposta traz sempre o pedido com as respostas que usou. Os testes do Python, da API e de
   ponta a ponta usam respostas guardadas; um teste com o marcador `ollama` corre o analista com o
   modelo a sério.
6. **Em segundo plano, como as procuras** (ADR 0022). O `AutoMLService` ganha `ask`, `questions`,
   `question_job`, `question_updates`, `question_result`, `cancel_question` e `analyst_options` (as
   perguntas sugeridas para a procura, que são uma regra do domínio, e as ferramentas), e cada
   um a sua rota, com o WebSocket da pergunta. Não há pausa: um modelo a meio de uma resposta não
   se pausa; parar fecha a ligação, e o Ollama deixa de escrever.
7. **O contexto do Ollama vai a 8192 tokens em todos os pedidos** (`num_ctx`), e não só nos do
   analista: com outro valor, o Ollama voltava a carregar o modelo de cada vez que o agente de IA
   e o analista se alternassem. Num `qwen3:8b` (Q4_K_M), o modelo e 8192 tokens cabem numa gráfica
   de 8 GB.
8. **O pensamento que cresce vale uma mensagem no WebSocket.** O WebSocket só mandava uma
   atualização quando havia passos ou mensagens novas, ou mudava o estado, o candidato ou a fase:
   o raciocínio do agente de IA (13.2), que cresce sem nada disso, só aparecia no fim. Agora o
   `thought` da procura e o `live` da pergunta também contam.

No ecrã Agents, depois de a procura acabar, o painel "Ask the analyst" fica logo a seguir ao
vencedor: as perguntas sugeridas, as ferramentas e, para cada pergunta, a linha do tempo dos
passos, com o raciocínio ao vivo. Sem o Ollama, mostra o estado de `GET /llm/status` e o que fazer.
A pergunta a decorrer fica na sessão do cliente, para se voltar a ela depois de sair do ecrã.

## Consequências

- **Fica mais fácil** ver a diferença entre dois desenhos de agentes de IA, lado a lado: o que
  recebe tudo e responde num formato (13.2) e o que decide o que ler (13.3).
- **As respostas ficam mais honestas do que o modelo**: as contas não são dele, e um número
  inventado vê-se. Mas a verificação só olha para os números. Uma frase errada com números
  certos passa ("a regressão logística teve um treino mais alto, 80.8% contra 85.8%"), e o
  registo passo a passo é o que deixa ver de onde veio cada coisa.
- **O desenho das ferramentas é para modelos pequenos**: um modelo maior talvez lesse bem um
  quadro com tudo. As lições ficam nas docstrings, para se voltar a medir com outro modelo.
- **Mudar uma ferramenta não muda o contrato com o frontend**: o JSON que ela devolve é texto
  para o modelo, e o painel mostra-o como está. Só o resumo e o nome a mostrar de cada
  ferramenta são campos.
- **Os testes de ponta a ponta apontam o Ollama para uma porta fechada**, e dão o mesmo com ou
  sem ele aberto no computador; para perguntar, a página finge-o pronto e o pedido leva as
  respostas guardadas.
