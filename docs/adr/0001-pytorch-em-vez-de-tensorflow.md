# 0001 — PyTorch em vez de TensorFlow

- **Estado:** Aceite
- **Data:** 2026-10-22

## Contexto

O ML Lab precisa de deep learning: uma MLP na fase 6, uma CNN pequena como extensão e o DQN
do reinforcement learning na fase 8. O objetivo principal do projeto é didático: cada
algoritmo tem de mostrar o que acontece durante o treino (loss, pesos, gradientes e
ativações, passo a passo), não só o resultado final.

Os dois frameworks dominantes são o PyTorch e o TensorFlow/Keras. Manter os dois duplicaria
adaptadores, testes e coisas para aprender, sem acrescentar aprendizagem.

## Decisão

O PyTorch é o único framework de deep learning do projeto. O TensorFlow/Keras não entra.

## Consequências

- O loop de treino é explícito (forward, loss, backward, step): cada iteração é um ponto
  natural para anunciar um passo ao contexto de treino e tirar snapshots.
- Os hooks do PyTorch permitem capturar ativações e gradientes de cada camada sem alterar o
  modelo.
- A conversão entre arrays NumPy (`float64`) e tensores acontece só dentro do
  `TorchAdapter`; o resto do sistema nunca vê tensores.
- O PyTorch é uma dependência pesada: só entra na fase 6, quando é preciso.
- Quem quiser comparar com Keras terá de o fazer fora do projeto.
