# 0036 — A barra pelos passos, o espaço em vez dos filetes e o botão de prosseguir

- **Estado:** Proposta
- **Data:** 2026-10-28
- **Mudada por:** a [0044](0044-um-botao-de-prosseguir-letra-maior-barra-alinhada.md): o botão
  de prosseguir fica só no topo, à direita do título (o `NextStep` saiu)

## Contexto

Depois da página larga e das secções abertas (ADR 0034), o pedido seguinte foi:

- explorar o dataset também na interface dos agentes;
- mais organização e o espaçamento certo, com uma skill própria para isso;
- menos linhas horizontais a separar as coisas;
- os botões de prosseguir mais salientes e bem colocados;
- os gráficos mais bem arrumados pelas secções, e tudo a seguir uma ordem;
- uma barra lateral mais personalizada, com letra maior e mais premium:
  - em cima, os Dados e o Explorar, iguais nas duas interfaces;
  - por baixo deles, a troca entre o Manual e os Agentes;
  - depois, as secções de cada interface;
  - os primeiros passos com uma animação constante;
  - depois os modelos e as métricas, e por fim o histórico.

Para a organização e o espaçamento instalou-se a skill `impeccable` (de pbakaus/impeccable),
só com o texto revisto: as referências `layout`, `distill`, `typeset`, `polish` e `quieter`. O
método dela guiou o trabalho:

1. Responder às perguntas da avaliação (ordem de leitura, agrupamento, ritmo, estrutura,
   densidade).
2. Fixar uma tese espacial.
3. Agrupar pela proximidade antes de acrescentar contentores.

O diagnóstico: cada secção tinha um filete por cima, um título pequeno e o mesmo intervalo.
Tudo pesava o mesmo, e eram as linhas que agrupavam, não o espaço. O botão de prosseguir mudava
de sítio e de peso de ecrã para ecrã.

## Decisão

**A barra segue os passos.**

1. **Os primeiros passos** (`FIRST_STEPS`: os Dados e o Explorar) são das duas interfaces e
   ficam em cima. O Explorar deixou de ser só do Manual.
2. **A troca** entre o Manual e os Agentes fica por baixo deles, porque é aí que os caminhos se
   separam.
3. **Os passos de cada interface** (`NAVIGATION[path][0]`) continuam os primeiros:
   - o Manual: Preparar, Modelos, Treinar e avaliar, Explicar;
   - os Agentes: Agentes, Treinar e avaliar, Explicar.
4. **"Aprender"** (os modelos, as métricas, o laboratório e o RL) e, por fim, **"Histórico"**.

Os passos estão ligados por um trilho, com um ponto por passo:

- **aceso** no passo aberto;
- **cheio** nos de trás;
- **vazio** nos que faltam.

Pelo trilho desce uma luz, sem parar (`rail-light`). O segundo troço parte a meio do ciclo, como
se a luz continuasse depois da troca. Com "reduzir movimento", a luz não corre.

A letra dos itens passou para 15 px, e a barra para 18rem. Os filetes que separavam o cabeçalho
e o rodapé saíram.

**O espaço separa; as linhas não.**

- **Os grupos:** o `Panel` deixou de ter filete, e a faixa de números (`StatStrip`) perdeu as
  linhas horizontais (ficam os separadores verticais entre números).
- **O ritmo:** os intervalos seguem uma escala de base 4, com contraste entre o apertado e o
  largo:
  - 12 a 16 px dentro de um grupo;
  - 40 a 48 px entre blocos;
  - 64 a 80 px entre secções.

  Há sempre mais espaço por cima de um título do que por baixo.
- **Os títulos dos blocos** passaram de cinzento a branco médio, porque, sem a linha, são eles
  que ancoram o grupo.
- **As linhas que ficaram** têm significado: as das tabelas, o total de uma conta, as etapas da
  linhagem e o eixo da linha do tempo do histórico.

**Prosseguir é sempre igual e está sempre nos mesmos sítios.**

- O `Button` ganhou a variante e o tamanho `proceed`: branco, maior, com a seta e um brilho por
  baixo. `Proceed` é o botão para um ecrã do fluxo; os ecrãs com o id de um run usam o mesmo
  aspeto.
- O botão aparece no topo, à direita, e outra vez no fim da página, num `NextStep`: uma frase
  sobre o que vem a seguir e o botão, sem caixa nem filete.
- Os passos e os seus botões:
  - Dados → "Explorar este dataset";
  - Explorar → "Preparar os dados" (Manual) ou "Pedir aos agentes" (Agentes);
  - Preparar → "Escolher o modelo";
  - modelo → "Treinar com estes valores";
  - Treinar e avaliar → "Explicar as previsões";
  - Explicar → "Comparar nas Experiências";
  - procura → "Abrir o vencedor";
  - RL → "Treinar o agente".
- Na procura terminada:
  - treinar um candidato à mão passou a ser uma ação por baixo do quadro, e não uma secção à
    parte;
  - "Ver o plano da procura" é um botão;
  - o fim da página leva ao vencedor.

**Os gráficos ocupam o espaço pelo que precisam.** No "O que aprendeu":

- a rede e a árvore ocupam a largura toda;
- os pesos, a importância e as tabelas ficam lado a lado;
- um só gráfico de meia largura fica com uma largura de leitura, em vez de esticar as barras.

## Consequências

- **O que substitui:** a ordem da barra substitui a da ADR 0033 (que tinha o Explorar só no
  Manual), e o "filete por cima" da ADR 0034 sai.
- **Os testes de ponta a ponta:**
  - os botões de prosseguir aparecem duas vezes por página, por isso os testes clicam no
    primeiro;
  - o teste da barra verifica que o Explorar está nas duas interfaces e que o Preparar só está
    no Manual.
- **Ficou de fora:** os executáveis da skill `impeccable` (o lançador e o detetor), por serem
  código por rever. As verificações mecânicas fizeram-se à mão.
