# 0040 — Uma só thread nos estimadores que somam em paralelo, para a mesma seed dar o mesmo

- **Estado:** Proposta
- **Data:** 2026-10-29

## Contexto

O laboratório promete que a mesma seed dá o mesmo resultado: a mesma divisão, o mesmo modelo e
o mesmo "o que aprendeu". O teste de contrato `test_the_same_seed_gives_the_same_result` treina
cada modelo duas vezes com a mesma seed e exige o mesmo estado, em JSON, até ao último dígito.

Ao correr a bateria toda para fechar a fase 11, esse teste falhou umas vezes sim, outras não, em
dois modelos do scikit-learn:

- **o k-means**, na inércia (`...907` contra `...908`, no 16.º algarismo);
- **o t-SNE**, na divergência KL (no 8.º algarismo).

A causa é a mesma nos dois. O scikit-learn faz as somas mais pesadas em paralelo, com OpenMP:
o k-means junta os centróides e a inércia de cada bocado dos dados, e o t-SNE (Barnes-Hut) o
gradiente. Cada thread soma o seu bocado, e os resultados juntam-se pela ordem em que as threads
acabam, que muda de uma corrida para a outra. Em vírgula flutuante a soma não é associativa:
`(a + b) + c` e `a + (b + c)` podem diferir no último bit. Num computador com 4 núcleos, 60
treinos do mesmo k-means deram 3 inércias diferentes.

A diferença não se vê num gráfico, mas quebra a promessa: dois runs iguais no histórico
apareciam com números diferentes, e o teste falhava ao acaso (um teste que falha ao acaso deixa
de proteger).

## Decisão

**Os estimadores que somam em paralelo numa ordem que muda treinam com uma só thread de
OpenMP.** O adaptador do scikit-learn (`core/models/adapters/sklearn_adapter.py`) tem a marca
`single_thread`; com ela ligada, o `fit` corre dentro de `threadpool_limits(limits=1,
user_api="openmp")`, do `threadpoolctl` (a biblioteca que o próprio scikit-learn usa para o
mesmo fim, e que passa a ser uma dependência declarada). Ligam-na o k-means e o t-SNE.

O limite é do processo todo, por isso um cadeado faz esperar um treino marcado enquanto outro
está a meio: assim nenhum repõe o número de threads enquanto o outro ainda conta com ele.

## Alternativas consideradas

- **Arredondar o estado** (a inércia, os centróides) antes de o mostrar: esconde a diferença
  quase sempre, mas não sempre (um valor perto da casa do arredondamento continua a saltar), e
  deixa o modelo diferente por baixo.
- **Uma só thread para todos os modelos do scikit-learn:** resolve tudo de uma vez, mas tira o
  paralelismo aos que já são deterministas (o gradient boosting por histogramas soma pela mesma
  ordem em todas as corridas), sem ganho nenhum.
- **`OMP_NUM_THREADS=1` só nos testes:** o teste passava, e a API continuava a dar resultados
  diferentes para a mesma seed.

## Consequências

- O k-means e o t-SNE dão exatamente o mesmo com a mesma seed (60 treinos seguidos, um só
  resultado), e o teste de contrato deixa de falhar ao acaso.
- Os dois treinam mais devagar num computador com muitos núcleos. Nos tamanhos do laboratório
  (milhares de linhas) são milésimos de segundo no k-means; o t-SNE já era o mais lento, e fica
  um pouco mais.
- Um modelo novo do scikit-learn que some em paralelo numa ordem que muda liga a mesma marca;
  o teste de contrato é quem o denuncia.
