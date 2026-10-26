# 0030 — Os textos de leitura vivem no backend, e as interfaces pedem-nos pelo nome

- **Estado:** Proposta
- **Data:** 2026-10-26

## Contexto

A regra 5 do `CONTRIBUTING.md` diz que os textos didáticos se escrevem uma só vez, no backend, e
deixou uma exceção com prazo: as legendas dos gráficos que ainda estavam nas páginas do
Streamlit mudavam-se quando os ecrãs fossem refeitos. As etapas 10.4 a 10.7 refazem os ecrãs
Dados, Explorar, Preparar, Modelos, Treinar e avaliar e Agentes, e cada gráfico novo precisa
de dizer como se lê. Havia quatro tipos de texto em falta:

- **o "como ler" de cada gráfico** (a matriz de correlação, a matriz de confusão, a
  linhagem…), escrito nas páginas do Streamlit, às vezes com as cores dessa interface ("a
  linha preta", "a linha vermelha");
- **o "como ler" do campo de decisão**, que depende do que os valores são (uma
  probabilidade, uma classe, um grupo, uma pontuação);
- **os temas do "Porquê?" da página de cada modelo**, que mudam de modelo para modelo (uma
  rede explica-se com a retropropagação, uma árvore com o desenho da árvore);
- **os nomes das métricas de cada passo** do treino (`accuracy`, `val_loss`, `oob_score`), que
  os modelos reportam pelo nome técnico.

## Decisão

1. **O "como ler" de cada gráfico é um tema**, como os do "Porquê?", na camada a que
   pertence: os dos dados em `core/data/explanations.py` (`correlation_matrix`, `missing_map`,
   `lineage`…), os da avaliação em `core/evaluation/topics.py` (`confusion_matrix`,
   `train_vs_test`, `roc_pr_chart`…), o mapa da redução no treino e o treino de um candidato à
   mão nos agentes. Escrevem-se sem as cores de nenhuma interface. As duas interfaces leem-nos
   do backend, e as páginas do Streamlit perderam as suas cópias.
2. **A interface web pede os temas pelo nome** (`<TopicText topic="lineage" />`), e pede-os
   todos de uma vez ao arrancar. Um nome que deixe de existir não daria erro (o texto só não
   aparecia), por isso `tests/contract/test_topics_in_use.py` procura os nomes no código das
   duas interfaces e falha se algum não existir.
3. **O que depende dos dados viaja com eles**, como o `how_to_read` dos pesos: a animação do
   treino traz o seu `how_to_read`, escolhido pelo que os valores são.
4. **O backend decide que temas explicam cada modelo** (`ModelCard.topics`): os do desenho do
   treino, os do que o modelo aprendeu e os das demonstrações do Laboratório que lhe dizem
   respeito. Um teste confirma que cada cartão tem temas, e que todos existem.
5. **As métricas de cada passo têm um registo** (`core/training/step_metrics.py`), com o nome a
   mostrar, se são percentagem e se são uma loss, e o vocabulário leva-o ao frontend. Um teste
   confirma que cada métrica que um modelo reporta está no registo.

## Consequências

- Um gráfico novo traz o seu tema: escreve-se uma vez, no backend, e as duas interfaces
  mostram o mesmo texto.
- O "Porquê?" e as legendas partilham o mesmo formato (`Explanation`): um tema de leitura
  também pode aparecer no painel, e vice-versa.
- Os textos ficam neutros quanto ao desenho: uma interface que mude as cores (a web pinta a
  fronteira a giz, o Streamlit a preto) não os contradiz.
- Os temas passaram de 53 para 76. O teste que procura os nomes lê o código das interfaces
  com expressões regulares: um tema pedido de outra forma (montado numa variável) escapa-lhe,
  por isso a interface web pede sempre os temas com o nome escrito por inteiro, exceto os que
  vêm já do backend (`ModelCard.topics`), que o seu próprio teste verifica.
