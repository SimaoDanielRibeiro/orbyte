# 0047 — A história do treino no ecrã Explicar: capítulos com texto e um visual

- **Estado:** Proposta
- **Data:** 2026-11-03

## Contexto

O pedido: no Explicar, uma explicação com palavras e também visual, completa mas simples, do que
aconteceu no treino.

O Explicar só respondia ao que pesa nas previsões (a importância por permutação, a dependência
parcial e uma previsão de perto). Faltava a pergunta anterior, a de quem acabou de treinar: o
que é que aconteceu? Os números estavam todos noutros sítios (a divisão no Preparar, a loss e as
métricas no Treinar e avaliar), mas soltos, e sem uma leitura que os ligasse.

O que pesa na decisão:

- os textos didáticos vivem no backend (ADR 0030): a interface não escreve frases sobre o domínio;
- o Explicar também abre runs do histórico que não foram guardados, e esses não têm o modelo em
  memória: a parte das previsões fica bloqueada, mas a história tem de se contar na mesma;
- funciona para qualquer tarefa: as tarefas sem alvo também treinam e também se saem bem ou mal.

## Decisão

**Um serviço novo, `HistoryService.story(run_id)`**, devolve a história do run (`TrainingStory`):
um resumo numa frase e seis capítulos, pela ordem em que se pergunta:

1. **com que dados aprendeu**: o dataset, a tarefa, a divisão (com as linhas de cada parte) e a
   preparação passo a passo;
2. **como aprendeu**: o resumo do modelo e, se treinou em passos, quantos, a loss do princípio ao
   fim e se desceu depressa ou ainda descia no fim;
3. **o que aprendeu**: os pesos, a importância das features, as componentes, a rede ou os
   vizinhos, conforme o modelo;
4. **como se saiu** em linhas que nunca viu, contra o palpite mais simples (a classe mais comum, ou
   a média) e contra o próprio treino (é aqui que aparece o overfitting);
5. **onde se engana**: o erro mais comum da matriz de confusão, ou o tamanho dos erros na regressão;
6. **o que fazer a seguir**: os avisos do run, que já dizem o que mudar.

Cada capítulo tem um texto curto, com os números que contam a negrito, e no máximo um visual, que
o backend escolhe e a interface só desenha: a divisão das linhas numa barra, a curva da loss
(com a validação, se houver), umas barras (as métricas comparadas, os pesos, a importância) ou a
matriz de confusão.

**Conta-se só com o resultado registado.** Se o run ainda estiver em memória, a preparação conta-se
com os números de cada passo (a linhagem); se não, pelo que a configuração pediu. Por isso a
história aparece em qualquer run, e a página abre com ela.

**No ecrã**, os capítulos são uma sequência numerada, ligada por um traço, com o texto à esquerda
e o visual à direita (um por baixo do outro num ecrã estreito). Por baixo, com o seu título,
continua o que pesa nas previsões, como estava.

O serviço fica no histórico porque é ele que sabe abrir um run de onde estiver (da memória ou do
registo).

## Consequências

- Quem acaba um treino tem a leitura completa num só sítio, por palavras e por imagens, e as
  tarefas sem alvo deixam de ter um Explicar vazio.
- Os textos novos são do backend, como os outros, e testam-se nos testes da história
  (`tests/unit/core/services/test_story.py`) e no contrato da fronteira (o JSON vai e volta).
- Um modelo novo conta-se sozinho pelo que já tem (o resumo, o estado aprendido, os passos); se
  aprender algo que não se resume em pesos nem em importâncias, a história diz que o resto está no
  Treinar e avaliar, e o modelo pode ganhar uma frase própria em `story.py`.
