# 0056 — O Node.js 24.16 no mínimo, e as actions do CI em dia

- **Estado:** Proposta
- **Data:** 2026-11-09
- **Muda:** a [0054](0054-o-analista-numa-conversa-o-quadro-inteiro-e-pronto-para-outros-computadores.md)
  (o projeto pronto para outros computadores pedia o Node.js 24; passa a pedir o 24.16)

## Contexto

Depois de o repositório ir para o GitHub, apareceram duas coisas.

**O servidor dos testes de ponta a ponta caía sem aviso.** No Windows, em cerca de uma corrida do
`npm run e2e` em cada três, o `vite preview` (o build da interface, na porta 5174) morria a meio
sem escrever nada. Todos os testes seguintes falhavam com `ERR_CONNECTION_REFUSED`, de 15 a 39 de
uma vez. No CI, em Linux, nunca aconteceu. A investigação, por ordem:

1. o registo do Windows não tinha nenhum crash da aplicação, e o processo também não chegava a
   correr o `process.on("exit")`: uma sonda carregada com `--import` registava as saídas normais
   dos outros processos, e nenhuma destas;
2. um observador dos processos (pelo WMI) mostrou que nada os terminava por fora: nenhum
   `taskkill` estava a correr quando morriam;
3. numa das corridas, quem morreu foi um *worker* do Playwright, e o Playwright disse o código:
   `3221226505`, ou seja, `0xC0000409` (`STATUS_STACK_BUFFER_OVERRUN`). É a falha nativa que o
   Windows dá quando uma verificação de segurança da memória falha, e o processo morre na hora,
   sem mensagem;
4. um supervisor à volta do `vite preview`, que é o único que vê o código de saída quando o
   processo morre assim, apanhou o mesmo `0xC0000409` no servidor.

Os dois processos têm uma coisa em comum: abrem muitas ligações TCP seguidas. O proxy do Vite passa
à API cada pedido da interface, e o *worker* pede à API pelo `request` do Playwright. É a
[issue #63620 do Node.js](https://github.com/nodejs/node/issues/63620): no Windows, até ao 24.15,
o caminho do `uv_tcp_connect` (no libuv) escrevia fora do sítio na pilha quando havia muitas
ligações seguidas, e o processo ia abaixo sem aviso. A correção chegou no 24.16
([#62561](https://github.com/nodejs/node/pull/62561)). O computador onde se trabalha tinha o
24.15.0.

**O CI avisava que o `astral-sh/setup-uv@v6` corre no Node 20**, que o GitHub Actions
descontinuou. O `actions/upload-artifact@v4` também corria no Node 20, mas como só corre quando
um teste falha, o aviso não chegava a aparecer.

## Decisão

**O Node.js mínimo passa a ser o 24.16**, em todos os sistemas e não só no Windows. Uma regra só é
mais fácil de dizer e de verificar, e quem instala hoje a versão LTS já tem a 24.22.

- O `engines` do `ui/web/package.json` (e do `package-lock.json`) passa a `>=24.16`.
- O `scripts/dev.py` compara a versão inteira, a principal e a secundária, como números
  (`(24, 9)` vem antes de `(24, 16)`, e o texto `"24.9"` viria depois de `"24.16"`). Com uma
  versão anterior, recusa-se a arrancar e diz o que fazer.
- O README pede o 24.16, e a tabela "Se alguma coisa falhar" diz porquê.

O CI já usava o 24 mais recente (`node-version: 24`).

**As actions do CI passam às versões atuais:**

- o `actions/checkout`, o `actions/setup-node` e o `actions/upload-artifact` passam à v7, pela
  versão principal, que o GitHub mantém;
- o `astral-sh/setup-uv` passa à v10.3.3, pela versão inteira. Desde a v8, deixou de publicar as
  etiquetas que se movem (`@v10`), porque uma etiqueta que se move pode passar a apontar para
  outro código sem ninguém dar por isso, e as versões inteiras são imutáveis.

Nenhuma das mudanças que quebram compatibilidade, entre a v5 e a v7 das actions do GitHub e
entre a v6 e a v10 do `setup-uv`, nos afeta. A cache do uv está ligada à mão (`enable-cache:
true`), tal como a do npm (`cache: npm`), e não se usam os eventos (`pull_request_target`,
`workflow_run`, `release`) em que a v10 desliga a cache automática.

## Consequências

- **Quem tem o Node.js 24.15 ou anterior tem de o atualizar** antes de arrancar o Orbyte, e o
  `scripts/dev.py` diz-lho logo.
- **Um processo do Node que morre sem dizer nada vê-se pelo código de saída**, do lado de quem o
  lançou: `0xC0000409` é uma falha nativa, e não um erro do projeto.
- **O `setup-uv` já não se atualiza sozinho** dentro da v10: cada versão nova escreve-se à mão,
  ou pelo Dependabot, se um dia se ligar.
