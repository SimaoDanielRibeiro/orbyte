# 0021 — Redes neuronais em PyTorch: um ciclo de treino próprio, que se pode observar

- **Estado:** Aceite
- **Data:** 2026-10-24

## Contexto

A fase 6 traz o deep learning: uma rede densa (MLP) em PyTorch. Tem de cumprir o mesmo
contrato `LabModel` que os outros 32 modelos (a mesma seed dá o mesmo resultado, há
snapshots, pára a pedido, o estado passa a JSON). Também tem de mostrar o que se passa
dentro da rede: os gradientes e as ativações de cada camada. Há quatro forças em jogo:

- o PyTorch tem um gerador aleatório global. Se a inicialização dos pesos e a ordem dos
  lotes dependessem dele, dois treinos com a mesma seed podiam dar redes diferentes;
- o ciclo de treino escreve-se à mão (é a razão da ADR 0001). É aí que se podem
  fotografar as camadas, mas também é aí que estão as armadilhas: parar a tempo,
  treinar um alvo com valores grandes e obter probabilidades que somem mesmo 1;
- fotografar os tensores (os pesos, a ativação de cada exemplo em cada neurónio) pesaria
  dezenas de MB por run;
- o resto do sistema só fala NumPy.

## Decisão

- **Um adaptador, `TorchModel`** (`models/adapters/torch_adapter.py`), escreve uma vez o
  ciclo de treino que todas as redes partilham:
  - mini-lotes baralhados;
  - Adam ou SGD com momento;
  - a penalização L2 (`weight_decay`);
  - um passo é uma época.

  Uma rede concreta só diz que camadas tem (`build_network`). O PyTorch só se importa aí e
  em `models/neural/`, e há uma regra de arquitetura que o verifica. Fora daí só circula
  NumPy.
- **A seed passa por um `torch.Generator` próprio**, criado a partir da seed do modelo.
  Inicializa os pesos (He na ReLU, Glorot nas outras ativações) e baralha os lotes. Nada
  usa o gerador global do PyTorch, por isso o mesmo run dá a mesma rede, observado ou não.
  Os testes de contrato confirmam-no.
- **O early stopping guarda os pesos da melhor época.** Uma parte do treino (10%, por
  omissão) fica de lado para validação. O treino pára quando a loss na validação não
  melhora há `patience` épocas (uma descida menor do que 10⁻⁴ não conta), e a rede volta
  aos pesos da melhor época. Um aviso diz como o treino acabou:
  - parou cedo;
  - fez as épocas todas;
  - havia linhas de menos para validar, e o early stopping ficou desligado.
- **Na regressão, o alvo treina-se padronizado** (média 0, desvio 1), e as previsões
  voltam à escala original. Com um alvo longe de zero (preços de casas, por exemplo), os
  primeiros gradientes seriam enormes e o treino instável.
- **As probabilidades calculam-se em float64.** O softmax em float32 dá somas como
  0,9999999, e o scikit-learn avisa, na log-loss, que aquilo não são probabilidades.
- **Cada snapshot leva um resumo das camadas, não os tensores.** Por camada, guarda o
  tamanho (norma) do gradiente dos pesos, em média na época. Nas camadas escondidas, mede
  numa amostra de até 256 linhas de treino:
  - a ativação média de cada neurónio;
  - que neurónios estão parados (o desvio padrão da saída abaixo de 10⁻³, ou seja, dão
    sempre o mesmo valor);
  - um histograma das ativações, com 20 intervalos.

  `visualization_data/network.py` junta os snapshots num `NetworkTrace`, pronto a desenhar.
  A proporção de neurónios parados conta-se aí, neurónio a neurónio.
- **O rasto da rede segue a regra da animação (ADR 0020):** fica fora do registo
  automático e só se guarda a pedido, em `network.json`. Numa MLP por omissão (duas camadas
  de 16) são uns 75 KB por run. Numa de quatro camadas de 128 passam de 800 KB, quando o
  resto do documento tem uns 25 KB.

## Consequências

- Uma rede nova (uma convolucional, por exemplo) é uma classe com o seu `build_network`. O
  ciclo, o early stopping e as fotografias vêm do adaptador.
- O ecrã Treinar mostra o mesmo para qualquer rede: o diagrama com as ativações, os
  gradientes por camada, os neurónios parados e os histogramas, ao longo do treino.
- Com early stopping, a rede aprende com menos 10% das linhas do treino. Para a comparar
  com outro modelo exatamente nas mesmas linhas, desliga-se.
- O PyTorch é a dependência mais pesada do projeto. No Windows, o PyPI dá a versão só
  para CPU, que chega para redes deste tamanho. No Linux, o PyPI dá a versão com CUDA,
  muito maior; para a evitar, aponta-se o uv para o índice CPU do PyTorch.
- Um modelo guardado com o joblib leva a rede dentro. Se a classe `TorchModel` mudar de
  forma, as redes guardadas antes deixam de abrir (o aviso da ADR 0020 explica-o), e
  reproduzir o run recria-as.
