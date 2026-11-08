# 0055 — As dependências da interface sem avisos de segurança: três overrides, e a ferramenta do shadcn só quando é precisa

- **Estado:** Proposta
- **Data:** 2026-11-08
- **Segue:** a [0054](0054-o-analista-numa-conversa-o-quadro-inteiro-e-pronto-para-outros-computadores.md)
  (o projeto pronto para outros computadores) e a
  [0027](0027-interface-web-com-o-cliente-gerado.md) (os componentes do shadcn são código nosso)

## Contexto

O repositório vai ficar público no GitHub, e o Dependabot mostra os avisos de segurança das
dependências. O `npm audit` da interface (`ui/web`) dava 23: 6 baixos, 13 moderados e 4 altos.
Pareciam muitos, mas vinham todos de quatro pacotes, e nenhum deles era uma dependência direta:

| Pacote | Aviso | Quem o trazia | Versão corrigida |
|---|---|---|---|
| `js-yaml` 4.2.0 | Alto: três maneiras de um YAML gastar CPU sem fim | o gerador do cliente (`@hey-api/openapi-ts` → `@hey-api/json-schema-ref-parser`, que o fixa em `4.2.0`) | 4.3.2 |
| `minimatch` 10.2.6 | Moderado: um padrão que se multiplica (negação de serviço) | o ESLint, o typescript-eslint e a ferramenta do shadcn | só a 11.0.1 e as seguintes |
| `katex` 0.16.47 | Baixo: uma poluição de protótipo que passa por cima das restrições do `trust` | o `rehype-katex` e o `remark-math`, que pedem `^0.16` (o projeto usa o 0.19.1) | 0.18.2 |
| `deepmerge` 4.3.1 | Baixo: poluição de protótipo no `mergeObject` | a ferramenta de linha de comandos do shadcn | nenhuma: a 4.3.1 é a última |

As versões mais recentes de quem os trazia ainda pedem as versões antigas, e o `npm audit fix
--force` propunha descer de versão: o `shadcn` para a 1.0.0, o `@hey-api/openapi-ts` para a 0.97.0
e o ESLint para o 9. Isso partia a interface e não era uma correção.

Dos 23 avisos, só os quatro do `katex` estavam no que chega ao browser; os outros eram de
ferramentas que só correm no computador de quem programa (o gerador do cliente, o lint e a
ferramenta do shadcn).

## Decisão

**Três `overrides` no `package.json`**, que obrigam toda a árvore a usar uma versão:

- `"js-yaml": "4.3.2"`: uma versão menor acima da 4.2.0, dentro da mesma API;
- `"minimatch": "11.0.5"`: a correção não chegou ao 10, por isso é uma versão principal acima da
  que o ESLint e o typescript-eslint pedem. A API que eles usam não mudou;
- `"katex": "$katex"`: o `rehype-katex` e o `remark-math` passam a usar o KaTeX do próprio
  projeto (o `$` aponta para a versão da dependência direta, 0.19.1), e não uma cópia antiga só
  deles. A função que eles chamam (`renderToString`) é a mesma.

**A ferramenta do shadcn sai das dependências.** O pacote `shadcn` é a linha de comandos que
copia um componente novo para `src/components/ui/`, e não tem correção para o `deepmerge`. O
projeto só o usava por uma folha de estilos (`@import "shadcn/tailwind.css"`), e dela só por nove
variantes dos estados do Radix (`data-open:`, `data-checked:`, `data-horizontal:`…) e pelo
utilitário `no-scrollbar`. Essas linhas passaram para o `globals.css`, com a origem e a licença
(MIT). O resto da folha (o `scroll-fade`, o `shimmer` e as animações do acordeão) não se usava.
Para acrescentar um componente, corre-se a ferramenta quando é precisa:
`npx shadcn@latest add <nome>`.

## Consequências

- **O `npm audit` dá 0 avisos**, e o lockfile passou de 910 para 680 pacotes (a ferramenta do
  shadcn trazia 230). Instala-se mais depressa, em todos os computadores.
- **Nada mudou no que se vê**, e confirmou-se ponto a ponto: o ESLint analisa os mesmos 205
  ficheiros e não encontra nenhum problema; o cliente gerado é igual, byte a byte; o CSS do build
  é igual, tirando sete propriedades do `scroll-fade` e do `shimmer`, que nenhuma classe usava;
  e os testes de ponta a ponta das fórmulas (Metrics e Train & evaluate) passam com o KaTeX
  0.19.1.
- **Os overrides têm de se rever.** Saem quando quem os traz os corrigir: o
  `@hey-api/json-schema-ref-parser` deixar de fixar o `js-yaml` 4.2.0, o ESLint e o
  typescript-eslint passarem ao `minimatch` 11, e o `rehype-katex` e o `remark-math` aceitarem o
  KaTeX 0.18.2 ou mais recente. `npm ls js-yaml minimatch katex` mostra quem pede o quê. Até lá, um
  override esconde do npm um conflito de versões: se um destes pacotes deixar de funcionar com a
  versão forçada, quem o diz são os testes (o lint, o gerador e as fórmulas).
- **As variantes copiadas atualizam-se à mão**, como os próprios componentes (ADR 0027). Se o
  `npx shadcn add` voltar a juntar o `@import "shadcn/tailwind.css"` e o pacote ao
  `package.json`, tiram-se outra vez.
