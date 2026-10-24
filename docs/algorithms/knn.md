# K vizinhos mais próximos (KNN)

## Intuição

"Diz-me com quem andas, dir-te-ei quem és." O KNN não aprende pesos nem regras: guarda os
exemplos de treino e, para classificar um ponto novo, procura os k exemplos mais parecidos
com ele e dá a classe que aparece mais vezes entre eles.

É o exemplo clássico de um modelo **baseado em instâncias**: o treino é instantâneo (só
guardar), e o trabalho todo fica para a previsão.

## Matemática essencial

Para um ponto $x$:

1. calcula a distância a cada exemplo de treino. A **euclidiana** é a linha reta,
   $d(x, z) = \sqrt{\sum_j (x_j - z_j)^2}$; a de **Manhattan** é a soma das diferenças,
   $d(x, z) = \sum_j \lvert x_j - z_j \rvert$, como andar por quarteirões;
2. fica com os $k$ exemplos de menor distância;
3. conta os votos de cada classe. Com pesos uniformes, cada vizinho vale 1; com pesos por
   distância, vale $1/d$ (um vizinho à distância zero leva o voto todo).

A "probabilidade" de cada classe é a sua fração dos votos. A versão didática do laboratório
(`knn_numpy`) faz estas três contas em NumPy, aos blocos de linhas, para não encher a
memória.

## Hiperparâmetros

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `n_neighbors` | Quantos vizinhos votam (o $k$). Poucos: fronteira irregular, que segue o ruído; muitos: fronteira lisa, que pode apagar detalhes | 5 |
| `weights` | `uniform` (todos os votos valem o mesmo) ou `distance` (os mais próximos valem mais) | `uniform` |
| `metric` | A distância: `euclidean` (linha reta) ou `manhattan` (soma das diferenças) | `euclidean` |

## Quando usar

- Como referência simples e honesta, sobretudo com poucas features e muitos exemplos.
- Quando a fronteira entre as classes é irregular e não se sabe que forma tem.

## Armadilhas

- **Escala das features:** a distância mistura todas as features. Uma feature em milhares
  (o rendimento) esmaga uma em unidades (a idade). Padroniza sempre na preparação.
- **k pequeno decora o treino:** com $k = 1$, cada ponto de treino acerta em si próprio
  (100% no treino) e cria uma "ilha" à sua volta (overfitting). O ecrã Laboratório mostra o
  efeito de k.
- **Muitas features:** em muitas dimensões, todos os pontos ficam mais ou menos à mesma
  distância uns dos outros (a "maldição da dimensionalidade"), e "o mais próximo" deixa de
  querer dizer "o mais parecido".
- **Previsões lentas:** cada previsão compara o ponto com todos os exemplos de treino.
- **Empates:** com k par, duas classes podem empatar. Fica a de menor número.

## O que observar no treino

O KNN treina de uma vez (só guarda os dados). No ecrã **Treinar** vês a fronteira final: com k
pequeno aparecem ilhas à volta de pontos isolados; com k grande, a fronteira fica lisa. Compara
`knn` com `knn_numpy` nos mesmos dados: as previsões são iguais.
