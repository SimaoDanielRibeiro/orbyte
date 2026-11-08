# 0054 — O analista numa conversa com o quadro inteiro, apagar no histórico e o projeto pronto para outros computadores

- **Estado:** Proposta
- **Data:** 2026-11-08
- **Muda:** a [0053](0053-o-analista-um-agente-com-ferramentas.md) (o quadro deixa de mostrar só os
  dez primeiros, e o painel do analista passa a uma conversa)

## Contexto

Depois de a etapa 13.3 pôr o analista a responder com o `qwen3:8b` a sério, o pedido seguinte
juntou quatro coisas:

1. **O painel do analista parecia um registo, e não uma conversa.** Estava encostado à esquerda,
   numa página larga; as perguntas sugeridas e as ferramentas ocupavam o topo; e cada resposta
   esticava a página, que tinha de se deslizar toda para acompanhar o analista. O pedido: ao
   centro, organizado, como uma conversa a sério, num espaço fixo em que só a conversa desliza.
2. **O analista só via os dez primeiros.** A ferramenta do quadro (`list_candidates`) tinha um
   argumento `limit`, com dez por omissão, e o `qwen3:8b` pedia sempre `limit=10`: numa procura
   de 24 candidatos, o analista explicava o vencedor sem ver 14 deles, e sem saber de alguns
   modelos. O pedido: que receba todos.
3. **Apagar um run estava escondido** na gaveta de cada run, no History.
4. **O projeto vai para o GitHub, para outras pessoas o correrem nos seus computadores.** O
   README dizia o que instalar, mas não como, nem o que fazer quando falha; e o CI só corria em
   Linux, enquanto quem descarrega o projeto usa sobretudo o Windows e o macOS.

As forças em jogo no quadro: a conversa inteira do analista tem de caber nos 8192 tokens do
Ollama (ADR 0053), e o orçamento de uma procura vai até 200 candidatos; um objeto de JSON por
candidato (oito linhas cada um) custava uns 100 tokens, e era nesse formato que o `qwen3:8b`
trocava os números de um candidato com os de outro.

## Decisão

**O quadro leva todos os candidatos, um por linha, e todos os modelos.**

- `list_candidates` deixa de ter argumentos: devolve sempre o quadro inteiro (até 40 linhas,
  `MAX_ROWS`). Um `limit` que venha de uma resposta guardada não conta.
- Cada candidato é uma linha de texto, com o número, o modelo e a nota juntos:
  `2nd: #3 Decision tree; 92.0% ± 2.0%, 1.0 points behind the first; its default settings;
  Round 1 · screening`. Custa uns 65 tokens (o objeto custava uns 100), e um modelo pequeno não
  tem como atribuir a um candidato a nota de outro, porque estão na mesma linha.
- Uma secção nova, `models`, diz cada modelo uma vez: quantos candidatos teve e o melhor deles
  (`Decision tree: 2 candidates, the best is #4 with 93.0% ± 2.0%, the winner`). Responde a "que
  modelos entraram?" sem ler o quadro todo, e é o que garante que nenhum modelo fica de fora.
- Acima de 40 candidatos (só com orçamentos grandes), os de baixo saem das linhas mas nunca dos
  modelos, e o quadro diz quantos ficaram de fora e que o `get_candidate` lê qualquer um. O 40
  mediu-se com o `qwen3:8b`: as instruções e as ferramentas custam 966 tokens, e o quadro de uma
  procura do Titanic com 24 candidatos e 12 modelos, 2611 (as linhas, 1532); com 40 linhas, a
  conversa começa perto dos 4900, e ainda cabem duas ferramentas de pormenor e o raciocínio.

**O analista é uma conversa, num cartão de altura fixa ao centro da página.** Em cima, o nome e
o modelo de linguagem com o estado (um ponto verde quando está pronto); no meio, a conversa, que é
a única parte que desliza (com `overscroll-behavior: contain`, chegar a uma ponta não passa o
deslizar à página); em baixo, a caixa da pergunta (Enter pergunta, Shift+Enter muda de linha), que
dá lugar ao "Stop" enquanto o analista trabalha, com as perguntas sugeridas que ainda não se
fizeram por cima. Sem perguntas, o meio dá as boas-vindas: as sugestões e as ferramentas.

- As perguntas ficam à direita, numa bolha; o analista à esquerda, com o ícone do agente de IA.
- Enquanto trabalha, os passos chegam à vista (o raciocínio a crescer, cada ferramenta com o que
  devolveu) e a conversa segue-os; quem sobe para ler deixa de seguir, com um botão "Latest" para
  voltar ao fim. Nunca se pede à página que mexa (nada de `scrollIntoView`).
- Quando responde, os passos fecham-se numa linha com as ferramentas que usou ("How it got
  there"), e a resposta fica numa bolha, com a leitura do backend por baixo. Os passos abrem-se
  com um clique, como antes, com o JSON que o modelo leu.

**Apagar está na tabela do History**: um caixote do lixo ao fim de cada linha, e "Delete the
selected" para os runs marcados. Os dois pedem confirmação (a dos escolhidos diz quais são), e
usam a rota de sempre (`DELETE /history/runs/{run_id}`), uma vez por run: não é preciso um
método novo no serviço. A gaveta do run continua a ter o seu "Delete the run".

**Pronto para outros computadores:**

- o README abre com "Começar": o que instalar, com os comandos de cada sistema (o uv, que traz o
  Python; o Node.js 24; o Ollama, opcional), o `git clone`, o `uv sync` e o arranque, e uma tabela
  "Se alguma coisa falhar" com o que aparece e o que fazer;
- o `scripts/dev.py` verifica antes de arrancar o que costuma falhar e diz o que fazer: o Node.js
  antigo, uma porta ocupada (uma API esquecida noutro terminal), e as dependências da interface,
  que instala com `npm ci` (as versões exatas do `package-lock.json`) da primeira vez e sempre que
  o `package-lock.json` mudar depois delas (um `git pull`). As do backend, o `uv run` já as acerta
  com o `uv.lock`;
- o CI ganha o job "Outros sistemas": no Windows e no macOS (Apple Silicon), o `uv sync --frozen`,
  os testes rápidos, o `npm ci`, o `npm run check` e o build. Corre a cada push no `main` com o
  repositório público, e à mão sempre: num repositório privado, cada minuto do macOS conta por
  dez;
- o Prettier aceita os fins de linha de cada ficheiro (`endOfLine: "auto"`): no Windows, com o
  `core.autocrlf` ligado (o que o instalador do Git escolhe), os ficheiros chegam com CRLF, e o
  `npm run format:check` dava 200 ficheiros "mal formatados" sem nada mal formatado. O
  repositório guarda LF na mesma, pelo `.gitattributes`;
- o `.gitignore` deixa de fora as cópias de trabalho locais.

## Consequências

- **O analista vê a procura inteira**, e explica o vencedor também com o que ficou para trás.
  Custa mais contexto: o quadro de 24 candidatos tem 2611 tokens, contra os 1098 dos dez objetos
  de antes (24 objetos teriam 2414, sem a lista dos modelos); a conversa continua a caber nos
  8192, com folga para o raciocínio.
- **O formato das linhas é uma aposta num modelo pequeno**, que se mediu com o `qwen3:8b` numa
  procura a sério (as notas da fase, em `docs/architecture.md`). Com outro modelo, volta a
  medir-se.
- **O painel tem uma altura fixa** (78% do ecrã, entre 32 e 52 rem): num ecrã pequeno, a conversa
  fica mais curta, mas a página nunca salta enquanto o analista escreve.
- **Apagar ficou à mão de semear**, por isso as duas maneiras pedem confirmação; os runs
  escolhidos continuam limitados a cinco (o máximo da comparação).
- **O macOS e o Windows passam a estar testados no CI**, mas só nos testes rápidos: os de
  ponta a ponta continuam só no Linux. O Mac com Intel e o Windows em ARM ficam de fora, porque o
  PyTorch deixou de ter versões para eles; o README diz isso.
