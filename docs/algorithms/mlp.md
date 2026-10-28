# Rede neuronal densa (MLP)

## Intuição

Uma regressão logística soma as features com pesos e decide de que lado de uma **reta** está
cada exemplo. Uma rede neuronal faz isso muitas vezes e encadeado: cada **neurónio** da
primeira camada é uma pequena regressão logística, a segunda camada combina as respostas da
primeira, e assim por diante. Entre camadas há uma **ativação** não linear, que dobra o espaço:
com ela, a rede desenha fronteiras **curvas**, como a que separa as duas luas.

Ninguém diz à rede o que cada neurónio deve fazer. Os pesos começam ao acaso e o treino
ajusta-os todos ao mesmo tempo, passo a passo, para a loss descer. Os neurónios acabam por se
especializar: uns reagem a um lado do plano, outros a outro, e a saída junta-os.

O nome **MLP** vem de *multilayer perceptron*: várias camadas de perceptrões, mas com
ativações suaves (que deixam calcular gradientes), em vez do degrau do perceptrão.

## Matemática essencial

Uma camada densa transforma o vetor de entrada $h$ (as features, ou a saída da camada
anterior) assim:

$$h' = f(W h + b)$$

onde $W$ é a matriz de pesos (um peso por par entrada-neurónio), $b$ as ordenadas e $f$ a
ativação, aplicada a cada neurónio:

- **ReLU:** $f(z) = \max(0, z)$. É a habitual: barata e com gradiente 1 sempre que está ligada;
- **tanh:** $f(z) = \tanh(z)$, entre −1 e 1;
- **sigmoide:** $f(z) = 1 / (1 + e^{-z})$, entre 0 e 1.

Sem $f$, duas camadas seguidas seriam $W_2(W_1 h) = (W_2 W_1) h$: outra conta linear. É a
ativação que dá à rede o poder de aproximar qualquer forma.

A **saída** depende da tarefa. Na classificação há um neurónio por classe e o **softmax**
transforma-os em probabilidades; a loss é a **entropia cruzada**, a mesma log-loss da
regressão logística. Na regressão há um neurónio sem ativação, e a loss é o **erro quadrático
médio** (o alvo é padronizado antes de treinar e as previsões voltam à escala original).

Para treinar é preciso o gradiente da loss em cada peso. A **retropropagação** calcula-os
todos de uma vez, da saída para a entrada, com a regra da cadeia: o gradiente de uma camada
obtém-se do da camada seguinte multiplicado pelas derivadas das ativações. O PyTorch faz esta
conta sozinho (`backward`). O treino percorre os dados em **lotes**; em cada lote:

$$\text{previsões} \rightarrow \text{loss} \rightarrow \nabla \text{(retropropagação)} \rightarrow w \leftarrow w - \eta\, \nabla_w$$

O **Adam** é um gradiente descendente que ajusta o passo de cada peso pela história dos seus
gradientes: costuma funcionar à primeira, sem afinar a taxa. O **SGD com momento** segue o
gradiente com alguma inércia.

Os pesos iniciais não podem ser todos iguais (os neurónios aprenderiam todos o mesmo) nem ter
uma escala qualquer: grandes demais saturam as ativações, pequenos demais apagam o sinal de
camada para camada. Por isso tiram-se ao acaso com a escala da ativação: **He** para a ReLU e
**Glorot** para a tanh e a sigmoide.

O **early stopping** põe de lado uma parte do treino, a **validação**. Enquanto a loss na
validação desce, a rede aprende o que generaliza; quando deixa de descer (e a do treino
continua a cair), começa a decorar. O treino pára depois de `patience` épocas sem melhorar, e a
rede fica com os pesos da melhor época.

## Hiperparâmetros

A forma da rede:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `hidden_layers` | Quantas camadas escondidas: mais camadas, formas mais ricas (e mais difíceis de treinar) | 2 |
| `units` | Neurónios em cada camada escondida: poucos, fronteira simples; muitos, mais risco de decorar | 16 |
| `activation` | A não linearidade: `relu`, `tanh` ou `sigmoid` | `relu` |

O treino:

| Hiperparâmetro | O que controla | Omissão |
|---|---|---|
| `optimizer` | `adam` (ajusta o passo de cada peso) ou `sgd` (com momento) | `adam` |
| `learning_rate` | Tamanho de cada passo: grande demais, a loss salta ou explode | 0,01 |
| `batch_size` | Exemplos por passo: lotes pequenos dão passos ruidosos e muitos por época | 32 |
| `max_epochs` | Máximo de passagens pelos dados de treino | 300 |
| `alpha` | Regularização L2: puxa os pesos para zero, a fronteira fica mais suave | 0,0001 |
| `early_stopping` | Se pára quando a validação deixa de melhorar | sim |
| `patience` | Épocas sem melhorar na validação até parar | 20 |
| `validation_fraction` | Parte do treino posta de lado para a validação | 0,1 |

## Quando usar

- Quando a fronteira (ou a relação com o alvo) é **não linear** e há dados suficientes: as
  luas, os círculos, interações entre features que um modelo linear não vê.
- Como ponto de partida para as redes maiores (de imagem, de texto): o ciclo de treino é o
  mesmo.
- Em dados tabulares pequenos, as florestas e o boosting costumam ganhar com menos trabalho;
  a rede vale pelo que ensina sobre treino, gradientes e ativações.

## Armadilhas

- **Escala das features.** Uma rede com features em escalas muito diferentes treina mal:
  padroniza na preparação.
- **Taxa de aprendizagem.** Grande demais, a loss salta ou explode (o treino pára com o aviso
  "a loss explodiu"); pequena demais, arrasta-se. Com o Adam, 0,01 ou 0,001 costumam servir.
- **Neurónios mortos.** Uma ReLU cuja soma ficou sempre negativa dá sempre zero, e o gradiente
  que passa por ela também é zero: não volta. Muitos neurónios parados (em **As camadas da rede**) pedem
  uma taxa menor.
- **Gradiente que desaparece.** Com a sigmoide (ou a tanh) em redes fundas, o gradiente encolhe
  de camada para camada e as primeiras quase não aprendem. Vê-se no gráfico dos gradientes.
- **Overfitting.** Com muitos neurónios e poucos dados, a rede decora o treino. O early
  stopping, o `alpha` e redes mais pequenas ajudam; a loss na validação a subir enquanto a do
  treino desce é o sinal.
- **Aleatoriedade.** Os pesos iniciais e a ordem dos lotes vêm da seed: com outra seed, o
  resultado muda um pouco. Compara runs com a mesma seed (ou várias seeds) no ecrã
  Experiências.
- **Poucos dados.** Com poucas linhas, a validação fica com tão poucas que o early stopping
  pára ao acaso (ou fica desligado, se não sobrar nenhuma).

## O que observar no treino

No ecrã **Treinar e avaliar**, a rede é fotografada época a época:

- a **fronteira de decisão** começa quase reta e vai-se curvando à volta das luas;
- a **curva da loss** mostra o treino e a validação: enquanto descem juntas, a rede aprende;
  quando a da validação estabiliza ou sobe, o early stopping está perto;
- o **diagrama da rede** pinta cada neurónio com a sua ativação média; os parados aparecem
  esbatidos. Muda o passo e vê as camadas a organizarem-se;
- o **gráfico dos gradientes** mostra o tamanho do gradiente de cada camada. Experimenta 4
  camadas com a sigmoide e vê o gradiente das primeiras a encolher;
- as **ativações** de cada camada: na ReLU, a barra em zero são os neurónios desligados; na
  tanh, barras altas nos extremos são neurónios saturados.

Experimenta nas luas: com 2 neurónios por camada, a fronteira fica quase reta; com 16, curva-se
à volta de cada lua; com uma taxa de 0,5, os neurónios morrem e a loss pára de descer.
