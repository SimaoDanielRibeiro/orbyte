# 0038 — O contorno colorido, sem números nem brilho, e menos texto à vista

- **Estado:** Proposta
- **Data:** 2026-10-29

## Contexto

Depois da ADR 0037, o pedido seguinte foi:

- tirar as bolas coloridas e passar a cor para o contorno do retângulo;
- tirar a numeração (01, 02…) e deixar só o título de cada coisa;
- tirar o brilho exagerado dos botões brancos;
- arrumar melhor: tirar os subtítulos e os textos que não fazem falta, e pôr no "?" o que
  precisar de explicação.

Ao rever os ecrãs apareceram as causas por baixo do pedido:

- **A bola é uma marca a mais.** Uma bola ao lado de um nome compete com ele e, numa lista,
  parece um marcador. No painel de cada agente havia bolas nas fichas dos agentes, uma barra
  colorida à esquerda dos cartões dos modelos e um brilho à volta da cor. O fluxograma, a
  conversa e a comparação dos runs repetiam o padrão.
- **Os números repetiam a ordem da página.** O índice e as partes do Explorar, os passos do
  Preparar e as fases da lição do RL tinham "01", "02"… em mono antes do título. A ordem já se
  lê de cima para baixo.
- **O brilho do botão de prosseguir** (uma sombra branca por baixo, ADR 0036) esbatia-lhe os
  contornos e puxava o olhar mesmo quando o passo ainda não estava pronto.
- **Os subtítulos repetiam o contexto.** O treino tinha "Titanic, classificação binária" por
  baixo do título, e três botões ao lado partiam o nome do modelo em três linhas. O Explicar, a
  procura dos agentes e o treino de um agente de RL faziam o mesmo. Cada métrica tinha uma
  linha em mono (as tarefas e para que lado é melhor), e cada secção da ficha de um modelo
  repetia o nome do separador por baixo dele.
- **Os "?" soltos.** Um gráfico estreito sem legenda (a matriz de confusão, os pesos) ficava
  com o "?" do "Como se lê" sozinho numa linha por cima do desenho, longe do título.

## Decisão

**A cor de uma coisa é o contorno do retângulo dela.** O contorno inteiro quando está escolhida
ou ativa; fraco (a cor com transparência) quando não está. O nome fica sempre escrito ao lado,
por isso a cor nunca é a única pista (para quem não distingue as cores). Passou a ser assim:

- **nos agentes:** as fichas para escolher os agentes, as caixas do fluxograma (o contorno
  acende com o estado: à espera, a trabalhar, já mediu), o avatar de cada mensagem da conversa,
  quem fala no histórico das trocas e, no painel de cada agente, a fila para trocar de agente,
  o cartão de cada modelo e a família do líder;
- **nos modelos:** o cartão de cada família, na lista, e a ficha da família, na página do
  modelo;
- **no plano do agente de dados:** o cartão de cada agente de modelo;
- **na comparação de runs:** a ficha de cada run e o cabeçalho da sua coluna;
- **na lição do RL:** a ficha de cada episódio, a azul se chegou ao objetivo e a laranja se
  caiu (a escala do valor das células);
- **na entrada:** o agente de cada família, no desenho do cartão dos Agentes.

O `ChipGroup` aceita uma `color` por opção, para os ecrãs não repetirem a regra.

**Sem numeração.** O índice e as partes do Explorar, os passos do Preparar e as fases da lição
ficam só com o título. A ordem é a da página. No Preparar, os passos ficam separados por um
filete, uma das linhas com significado da ADR 0036 (separa os passos de uma sequência).

**Os botões brancos sem brilho.** A variante `proceed` destaca-se pelo tamanho, pelo peso e pelo
branco sobre o escuro, sem sombra. Pelo mesmo motivo, saíram os brilhos à volta do cartão da
barra lateral, das barras do coordenador e da caixa ativa do fluxograma. O ponto do trilho e a
sombra do agente da lição ficaram mais fracos. O ponto verde a pulsar do treino ao vivo ficou:
diz "agora".

**Menos texto à vista.** Um título diz o que é; o resto vai para um número, para o "?" ou sai:

- **o contexto de um run** passa a ser um número da faixa: "Dados" no treino (com a tarefa) e
  "Ambiente" no RL. A seed fica como detalhe do run, e a faixa volta a ter quatro números;
- **o cabeçalho do treino** fica com dois botões (guardar e explicar). Os botões só ficam ao
  lado do título se houver espaço para os dois, e "Experimentar outro modelo" passa para o fim,
  junto do passo seguinte;
- **no Explicar e na procura dos agentes**, o modelo, os dados e o objetivo vão para o "?" da
  página;
- **o fim de cada passo** (`NextStep`) é só o botão: o nome dele já diz o que vem a seguir;
- **a entrada** fica com o nome da app e os dois cartões, sem a frase por baixo do nome e sem a
  fila dos passos de cada interface (estão na barra);
- **nos Dados**, o número de avisos só leva uma nota quando os dados não podem seguir, e a
  explicação da seed dos dados sintéticos vai para o "?";
- **os rótulos dos campos** ficam curtos ("Coluna-alvo", "Os agentes", "Dividir em
  intervalos"), com a explicação no "?";
- **cada métrica** tem as tarefas e o sentido ("quanto maior, melhor") no "?", em "Onde se
  usa";
- **a ficha de um modelo** não repete o nome da secção por baixo do separador;
- **a lição do RL:**
  - o agente, o ambiente e os episódios ficam no "?" do título;
  - a frase de cada episódio fica no "?" ao lado das fichas dos episódios;
  - o texto de cada fase do ciclo fica no "?" do seu cartão;
  - os controlos (reproduzir e a velocidade) passam para a fila do progresso, como num leitor
    de vídeo;
- **as indicações de uso** ("Clica num agente…", "com muitas features, demora um pouco") vão
  para o "?" do sítio a que se referem;
- **a linhagem do Preparar** perde o subtítulo, e o "?" dela fica ao lado dos separadores do
  resultado. Para caber na mesma linha, o separador "Classes em cada parte" passa a "Classes";
- **um gráfico estreito sem legenda** (a matriz de confusão, os pesos, a importância, a árvore,
  a rede) tem o "Como se lê" no "?" do título do bloco, não num "?" solto por cima do desenho.

**Os painéis laterais com a largura certa.** O "?" de cada página (32rem), o de um run (32rem) e
o de uma métrica (36rem) passam a usar a variante do `Sheet` que a ADR 0037 encontrou
(`data-[side=right]:sm:max-w-*`). Até aqui ficavam presos a 24rem.

## Consequências

- **O que substitui:**
  - da ADR 0036: o brilho do botão de prosseguir e a frase do `NextStep`;
  - da ADR 0037: o fio de luz e o brilho dos cartões das famílias, as partes numeradas do
    Explorar e a exceção à ADR 0032 da lição (os textos das quatro fases estavam à vista).
- **Uma regra para o que vier:**
  - uma coisa nova com cor leva-a no contorno (`borderColor`: a cor inteira quando está
    escolhida, `withAlpha(cor, ~0.4)` quando não está) e o nome escrito;
  - uma linha nova de contexto vai para um número da faixa ou para o "?", não para baixo do
    título.
- **Ler as explicações custa um clique.** O que saiu da página continua ao alcance do teclado:
  cada "?" é um botão com nome ("Explicar: …"). A frase de cada episódio da lição deixou de se
  anunciar sozinha a quem usa um leitor de ecrã; ouve-se ao abrir o "?".
- **Os testes de ponta a ponta:**
  - o dos modelos procura o painel de cada separador da ficha, e não o título repetido;
  - o "?" ao lado de "Calcular a importância" chama-se "Quanto demora", para o nome não conter o
    do botão;
  - o da troca de alvo no Preparar procura o separador "Classes" pelo nome exato, e verifica
    primeiro que ele existe com o Survived (antes, só verificava que desaparecia).
- **O backend não mudou:** os textos novos do "?" são os temas e os campos que já existiam.
