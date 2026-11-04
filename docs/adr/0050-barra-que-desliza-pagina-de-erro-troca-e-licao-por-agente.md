# 0050 — A barra que desliza sem reorganizar a página, a página de erro, a troca que leva ao passo certo e uma lição por agente

- **Estado:** Proposta
- **Data:** 2026-11-04
- **Muda:** a [0049](0049-agentes-com-todos-os-modelos-escolhidos-e-por-palavras.md) (o trilho dos
  passos sai do meio dos ícones) e a [0037](0037-cores-por-familia-barra-que-desliza-e-licao-do-rl.md)
  (a lição do reinforcement learning deixa de ser só do Q-learning)

## Contexto

O pedido:

- corrigir a barra lateral (recolhida, estava estranha);
- melhorar as transições e o reajuste da página ao abrir e fechar a barra;
- uma página de erro como um gráfico com a linha a descer, a dizer o que aconteceu e "try
  again", simples e eficaz;
- no Manual, já com os Dados e o Explorar feitos, trocar para os Agentes a partir do Preparar
  devia levar à secção Agentes, e não outra vez aos Dados;
- ver o *timelapse* de um agente a aprender, como em "How an agent learns", para cada tipo de
  agente de reinforcement learning.

O diagnóstico:

- **O trilho partido.** O trilho dos passos passava pelo meio dos ícones (ADR 0049), tapado por um
  disco à volta de cada um. Com a barra recolhida sobravam só tracinhos soltos entre os ícones, que
  pareciam um defeito.
- **A página a redesenhar-se a cada fotograma.** A barra animava a largura do lugar dela, e por
  isso o conteúdo mudava de largura em cada fotograma da animação: mais de trinta reorganizações
  em meio segundo, e os gráficos (os que se desenham pixel a pixel, também) a redesenhar-se a cada
  uma. A troca de interface mudava de forma de uma vez a meio, e as contagens sumiam de repente.
- **A página de erro era só texto.**
- **A troca levava sempre aos Dados**, mesmo a quem já os tinha aberto.
- **A lição era só do Q-learning**: o DQN aprende de outra maneira (com uma rede e lotes de uma
  memória), e não havia como o ver.

## Decisão

**O trilho dos passos sai: a barra fica simples**, só com os itens (o ícone e o nome) e o cartão
claro por trás do ecrã aberto. Primeiro passou-se o trilho para junto da borda, fora da coluna dos
ícones (uma linha acesa até ao passo aberto, com um ponto e uma luz a percorrê-la); vista no
browser, também parecia um defeito, e o pedido foi tirá-la de vez. A ordem dos passos lê-se pela
própria lista.

**Ao abrir e fechar a barra, a página reorganiza-se uma só vez** (a técnica FLIP):

- o lugar da barra muda de uma vez, e o conteúdo fica logo com a largura final (os gráficos
  redesenham-se uma vez);
- o `AppFrame` mede onde o conteúdo estava antes e onde ficou, e faz deslizar a diferença com um
  `transform` (a Web Animations API, em 500 ms, com a mesma curva da barra), que não volta a
  reorganizar nada;
- a moldura corta o que passa da borda enquanto desliza (`overflow-x: clip`, que não estraga o
  `sticky`).

Mediu-se: o conteúdo muda de tamanho uma vez (antes, a cada fotograma), e desliza de 68 para
336 px sem barra de deslocamento. Na própria barra:

- os nomes, o rótulo "Learn", a marca e as contagens apagam-se depressa ao fechar (150 ms) e
  acendem um nada depois de abrir (100 ms de espera, 300 ms);
- a troca de interface tem as duas formas sempre lá, uma por cima da outra, e passa de uma à
  outra a desvanecer (a aberta tem uma largura mínima: fica cortada, não espremida);
- as cores dos botões ao passar o rato mudam em 200 ms, e só a largura acompanha a barra.

**A página de erro é um gráfico que cai**: a linha sobe aos poucos, como um treino a correr bem,
e no ponto mais alto cai a pique, a vermelho, até ao fundo, onde fica um ponto a pulsar com o
nome do que caiu ("API offline", "error", "404"). Por baixo, o título, a mensagem e "Try again"
(volta a pedir tudo; reinicia a fronteira do erro), e uma ligação para a entrada. O mesmo desenho
serve o endereço que não existe. As animações são CSS, e com "reduzir movimento" aparece tudo já
desenhado.

**A troca de interface leva ao passo certo** (`landingOf`, em `screens.ts`): num ecrã que a outra
interface também tem, fica-se; num que não tem (o Preparar, os Agentes, o Laboratório), vai-se
para o primeiro passo da outra (os Agentes, ou o Preparar), se os dados já estiverem abertos; sem
dados, para os Dados, onde as duas começam.

**A lição tem um agente à escolha** (`GET /reinforcement/lesson?agent=…`): o Q-learning e o DQN.
O backend grava o treino de verdade, com um gravador que se junta ao agente por herança
(`_Recording`) e toma nota de cada passo e de cada episódio sem mudar o que o agente faz; por isso
serve qualquer agente episódico novo. A surpresa de cada passo calcula-se como o agente a vê
antes de aprender (no Q-learning, é o erro que corrige a tabela). A lição diz:

- que tema explica a correção do Q (`update_topic`: `rl_update`, para uma tabela;
  `rl_update_network`, para uma rede);
- a frase do último cartão (`update_rule`: "Q moves α = 0.5 of it", ou "the network takes a
  training step on 32 steps drawn from its memory");
- as frases de cada episódio à maneira do agente (a rede começa com pesos ao acaso, não a zeros).

A do DQN treina 300 episódios (uns 6 s, a primeira vez; depois fica guardada). Entretanto, a lição
de antes fica à vista, esbatida, com uma linha que diz o que se está a treinar.

## Consequências

- Um elemento novo na barra recolhida não pode depender de a largura animar: o lugar dela muda
  de uma vez.
- Uma página com algo `fixed` dentro do conteúdo vê-o deslizar com ele durante meio segundo (o
  `transform` faz dele a referência); no fim, o `transform` sai.
- Um agente de reinforcement learning novo tem lição sem mudar a lição: chega ser episódico. Se
  precisar de outra frase para a correção do Q, acrescenta-se em `_update_rule`.
- A primeira vez que se abre a lição do DQN numa API acabada de arrancar, demora uns segundos.
