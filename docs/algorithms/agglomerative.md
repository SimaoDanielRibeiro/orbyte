# Clustering hierárquico (aglomerativo)

## Intuição

Começa com cada ponto sozinho no seu grupo e, passo a passo, **junta os dois grupos mais
próximos**, até sobrar um só. O histórico das junções forma uma árvore, o **dendrograma**:
cortá-lo a uma certa altura dá um certo número de grupos. Em vez de uma divisão, dá uma
hierarquia inteira: espécies dentro de géneros, géneros dentro de famílias.

## Matemática essencial

O algoritmo é sempre o mesmo; muda a forma de medir a distância entre dois grupos $A$ e $B$
(a **ligação**):

- **simples**: a menor distância entre um ponto de $A$ e um de $B$ (segue cadeias de pontos);
- **completa**: a maior distância (prefere grupos compactos);
- **média**: a média das distâncias entre todos os pares;
- **Ward**: o aumento da variância dentro dos grupos se $A$ e $B$ se juntarem (dá grupos
  redondos e equilibrados, como o k-means).

Em cada passo, junta o par com a menor distância. A altura de cada junção no dendrograma é
essa distância. Para ter $k$ grupos, desfazem-se as últimas $k - 1$ junções.

O laboratório guarda o dendrograma no estado do modelo. Para pontos novos (que o
scikit-learn não prevê), usa o grupo do ponto de treino mais próximo.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_clusters` | Em quantos grupos se corta o dendrograma | 3 |
| `linkage` | A distância entre grupos: `ward`, `complete`, `average` ou `single` | `ward` |

## Quando usar

- Quando interessa a **estrutura em vários níveis**, e não só uma divisão.
- Com **poucos dados** (até alguns milhares de linhas), para ver o dendrograma e escolher o
  número de grupos olhando para as alturas das junções.

## Armadilhas

- **Memória:** guarda a distância entre todos os pares de pontos ($n^2/2$). O laboratório
  recusa mais de 5000 linhas de treino.
- **Decisões irreversíveis:** uma junção má no início nunca se desfaz.
- **A ligação simples encadeia:** dois grupos ligados por uma fila de pontos juntam-se cedo
  demais.
- **Escala das features:** as distâncias misturam todas as features. Padroniza.

## O que observar no treino

Treina de uma vez. No ecrã **Treinar e avaliar** vês os grupos no plano e, na avaliação, o
**dendrograma**: as últimas 30 junções, com a linha do corte que dá os grupos escolhidos. Uma
junção muito mais alta do que as outras é um sítio natural para cortar.
