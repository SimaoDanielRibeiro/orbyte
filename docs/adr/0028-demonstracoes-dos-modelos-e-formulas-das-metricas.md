# 0028 — A página de cada modelo e a explicação de cada métrica: demonstrações e fórmulas

- **Estado:** Proposta
- **Data:** 2026-10-25

## Contexto

A interface aprimorada (fase 10) pede duas coisas que o backend ainda não dava:

1. **Uma página por modelo que explique como ele funciona, com exemplos visuais**, e com os
   hiperparâmetros a mostrar o efeito de cada um. Para isso é preciso ver o modelo a
   trabalhar, mas um run normal vai para o histórico e precisa de um dataset carregado. Se a
   página treinasse um run de cada vez que se mexe num controlo, o histórico enchia-se de
   lixo e o ecrã Dados de conjuntos que ninguém carregou.
2. **Cada métrica explicada com a fórmula e uma animação com os números do run** (na
   precisão, os VP e os FP da matriz de confusão a entrar na fração). As fórmulas não
   existiam no backend, e calcular os termos no frontend obrigava-o a decidir regras de
   domínio: qual é a classe positiva na binária, como se faz a média na multiclasse, como se
   evita o 0/0 (regra 4 do `CONTRIBUTING.md`).

A ficha didática também só vinha inteira, num texto, e a página quer pôr cada secção no seu
lugar (a Intuição ao lado do exemplo, a Matemática essencial com as fórmulas).

## Decisão

**Demonstrações** (`services/demos.py`, `CatalogService.demo`):

- Cada modelo treina num **conjunto de exemplo pequeno da sua tarefa**: as luas e os
  círculos na binária, três nuvens na multiclasse, uma onda e uma reta na regressão, nuvens e
  luas sem rótulo no clustering, anomalias, e nuvens em 5 dimensões na redução. Os conjuntos
  são uma decisão didática, por isso estão no backend; o primeiro de cada tarefa é o de
  omissão. A regressão ganhou um gerador novo, a **onda** (um seno com ruído), porque numa
  reta uma árvore ou uma rede mostram pouco do que fazem.
- A demonstração devolve **o mesmo que um run**: o campo em cada fotografia, a loss, o que
  aprendeu, a rede, o mapa, os relatórios e os gráficos. A interface usa os mesmos
  componentes nas duas coisas.
- **Não é um run.** Treina num `ExperimentService` só seu, com um bus de eventos sem ninguém
  a ouvir (quem regista é um observador do bus), e com um `DatasetService` também só seu:
  nada vai para o histórico, e os dados não aparecem no ecrã Dados. É o mesmo que o
  Laboratório já fazia.
- É **síncrona e guardada**: as 34 demonstrações de omissão demoram 4,6 s ao todo (a rede
  neuronal é a mais lenta, com 2 s), e as últimas 32 ficam em memória, por isso voltar a um
  valor já visto é imediato.
- O cartão do modelo ganhou **a ficha por secções** (`sections`, com uma chave estável como
  `matematica-essencial`), **as demonstrações do Laboratório que se aplicam**
  (`laboratory`: o k no KNN, a regularização na Ridge e no Lasso…) e **os conjuntos de
  exemplo** (`examples`). O `LearnedView` ganhou **a árvore como estrutura** (`tree`), para a
  desenhar, com os mesmos números das regras.

**Fórmulas e termos** (`evaluation/explanations.py`, `Metric.breakdown`):

- Cada métrica tem a **fórmula em LaTeX** e o significado de cada símbolo, ao lado do que já
  tinha (o que mede e como se lê). `CatalogService.metrics` dá tudo isto, e pede-se uma vez.
- As métricas que são uma conta com quantidades que se contam (14 das 22) dão os **termos da
  fórmula** com os números dos dados avaliados (`MetricResult.terms`): os VP e os FP na
  precisão, as somas dos quadrados no R²… Na multiclasse vêm por classe (`group`), porque a
  métrica é a média delas. Cada termo calcula-se ao lado da conta da métrica, com as mesmas
  regras.

## Consequências

- A página do modelo e a gaveta de cada métrica só desenham: as regras (que exemplo, que
  classe é a positiva, como se faz a média) ficam no backend.
- Os testes de contrato guardam o resto: cada modelo tem demonstração e a ficha com as seis
  secções; cada métrica tem fórmula, cada símbolo explicado está na fórmula, e cada termo é um
  símbolo dela. Os testes da aritmética confirmam que os termos dão o valor da métrica.
- Uma demonstração usa os hiperparâmetros pedidos: se alguém pedir 5 000 épocas a uma rede,
  espera por elas. Os conjuntos pequenos tornam isso raro.
- Um modelo novo continua a ser um ficheiro e a sua ficha: a demonstração e a página saem
  sozinhas, desde que a ficha tenha as secções do modelo.
