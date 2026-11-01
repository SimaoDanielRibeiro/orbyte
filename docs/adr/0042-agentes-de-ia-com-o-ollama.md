# 0042 — Agentes de IA com o Ollama, ao lado dos agentes de regras

- **Estado:** Proposta
- **Data:** 2026-11-01
- **Muda:** a [0025](0025-agentes-de-automl-deterministas.md) (os agentes de regras ficam; juntam-se outros, de IA)

## Contexto

Os agentes do modo automático (ADR 0025) são regras e medições, sem modelos de linguagem:
decidem sempre da mesma forma e explicam cada decisão. Isso continua certo para o que fazem.
Mas o projeto é para aprender, e falta ver como trabalha um **agente de IA**: um modelo de
linguagem que lê o que já se mediu, pensa e propõe, e que se pode ver a pensar.

As forças em jogo:

- **a avaliação tem de continuar honesta**: um modelo de linguagem pode inventar números ou
  preferir o que "soa bem". Não pode medir candidatos, ver o teste nem escolher o vencedor;
- **a mesma seed tem de dar o mesmo resultado**, e um modelo de linguagem não o garante: mesmo
  com a temperatura a 0, a resposta pode mudar de um computador para outro;
- **corre tudo no computador**: o laboratório não depende de serviços na nuvem nem de contas;
- **é opcional**: sem o modelo de linguagem, o laboratório funciona como antes;
- **os testes não podem depender** de um modelo de 5 GB nem de uma gráfica;
- **sem tecnologias novas desnecessárias** (como na ADR 0025).

## Decisão

1. **O Ollama, no próprio computador**, fala-se por HTTP com a biblioteca padrão (`urllib`),
   sem dependências novas. É o único pedido à rede do núcleo, e só o `core/llm/` o faz (há uma
   regra de arquitetura). O endereço e o modelo são definições (`ML_LAB_OLLAMA_URL`,
   `ML_LAB_OLLAMA_MODEL`). Por omissão, o modelo é o `qwen3:8b`: pensa à parte, usa ferramentas,
   escreve bem em português e cabe numa gráfica de 8 GB.
2. **Uma interface, `LanguageModel`** (`core/llm/base.py`), que responde aos bocados (`Chunk`).
   O raciocínio chega à parte (`thinking`), antes da resposta (`content`): é isso que deixa a
   interface mostrar o agente a pensar, ao vivo. Pode pedir-se uma resposta num JSON Schema
   (saída estruturada) e oferecer ferramentas ao modelo. O resto do laboratório só conhece esta
   interface; o cliente do Ollama só se importa no `core/llm/` e nos serviços.
3. **Os testes usam o `ScriptedModel`**: um `LanguageModel` com as respostas escritas à mão,
   que as dá aos bocados como um modelo a sério e guarda o que lhe perguntaram. O cliente do
   Ollama testa-se contra um servidor HTTP de mentira; um teste com o Ollama a sério (marcador
   `ollama`) salta-se quando ele não está a correr.
4. **O estado é um serviço, não um erro.** `LanguageModelService.status()` diz se o Ollama
   responde, que modelos tem e se tem o configurado, e o que falta fazer, em avisos com um
   código estável (`ollama_not_running`, `model_not_installed`). A interface usa-o para ligar
   ou desligar os agentes de IA.
5. **A IA propõe e explica; as regras decidem e medem.** O agente de IA (etapa 13.2) é **um
   agente a mais** na afinação, ao lado dos agentes de família, e não os substitui: na sua vez,
   escolhe um dos finalistas e os valores dos hiperparâmetros, lendo os resultados de todos os
   candidatos. Assim, a IA e a procura aleatória competem na mesma procura, com o mesmo
   orçamento, e no fim compara-se a melhor proposta dele com o melhor sorteio. O coordenador
   continua a medir com validação cruzada, só no treino. Uma proposta que não serve (fora do
   espaço de procura, de um modelo que não é finalista, ou já medida) é recusada com a razão e
   não gasta orçamento; à terceira recusa seguida, ou se o modelo de linguagem deixar de
   responder, o agente de IA sai da afinação e a procura continua sem ele. O teste continua a ver
   só o vencedor. A temperatura vai sempre a 0.
6. **Para repetir uma procura, guardam-se as respostas.** Cada resposta do modelo (o raciocínio
   e o JSON) fica no pedido do resultado (`AutoMLRequest.ai_replies`). Correr esse pedido outra
   vez usa o `ScriptedModel` com as respostas guardadas, e não pergunta ao Ollama: a mesma seed
   volta a dar o mesmo, mesmo que o modelo hoje respondesse outra coisa.

As etapas da fase 13 estão nas notas da implementação, em `docs/architecture.md`.

## Consequências

- **Fica mais fácil** ver a diferença entre um agente de regras e um agente de IA, lado a lado,
  na mesma procura e com a mesma forma (`propose`, `remember`, `report`).
- **O laboratório continua a correr sem o Ollama**, e os testes também.
- **Passa a ser preciso** ter o Ollama aberto e o modelo descarregado para ver os agentes de IA
  (`ollama pull qwen3:8b`, uma vez). Numa máquina sem gráfica corre, mas devagar.
- **Uma procura com IA é mais lenta**: cada proposta espera pelo modelo (segundos, numa gráfica).
- **Fica uma porta aberta**: outro modelo, ou outro servidor, entra com uma classe nova que
  cumpra o `LanguageModel`, sem mexer nos agentes.
