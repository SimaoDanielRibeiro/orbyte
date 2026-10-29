# 0039 — O pacote do modelo: a preparação e o que o modelo aprendeu num ficheiro

- **Estado:** Proposta
- **Data:** 2026-10-29

## Contexto

O pedido: uma secção nova para **guardar o que se preparou e o que os modelos aprenderam**.
Deve mostrar, antes de guardar, uma pré-visualização do ficheiro que vai ser gerado e o que
vai lá dentro. É também o primeiro passo do laboratório de previsões (fase 12), que vai prever
com um modelo guardado.

O que já existe não chega:

- **Guardar uma experiência** (ADR 0020) grava o modelo (joblib) e o treino passo a passo numa
  pasta do laboratório. Serve para reabrir o run, não para levar o modelo para outro sítio.
- **A preparação aprendida não se guarda.** O pipeline aprende com o treino (a mediana para os
  nulos, as categorias, a média e o desvio de cada coluna), transforma as partes e é deitado
  fora (`_assemble`, em `core/data/preparation.py`). Reabrir um run volta a preparar os dados a
  partir da fonte. Sem a fonte (um ficheiro apagado, outro computador), uma linha nova não tem
  como chegar ao modelo na forma em que ele aprendeu.
- **O joblib é um pickle.** Abri-lo corre o código que lá estiver, e por isso o laboratório só
  abre os que ele próprio gravou. Ninguém consegue ler o que lá está.
- **Nada fica legível fora do laboratório:** nem o que o modelo prevê, nem com que dados, nem
  como se portou.

As forças em jogo:

- **legível:** é um laboratório para aprender, e abrir o ficheiro deve ensinar alguma coisa;
- **completo:** prever uma linha nova sem a fonte dos dados;
- **seguro:** um pickle de outra pessoa nunca corre;
- **honesto:** dizer que dados vão dentro do ficheiro (linhas de exemplo, modelos que guardam o
  treino);
- **reprodutível:** saber com que versões se fez;
- **stack enxuta** (ADR 0003): nenhuma dependência nova.

## Decisão

**Um ficheiro `.mllab`: um ZIP com nomes fixos.** Abre-se com qualquer programa de ZIP. Os nomes
dos ficheiros lá dentro estão em inglês, como os de `artifacts/runs/`; os textos estão em
português.

| Ficheiro | O que tem | Sempre? |
|---|---|---|
| `README.md` | A ficha do modelo, para pessoas: o que prevê, com que dados, como se portou, os limites e como se usa | Sim |
| `manifest.json` | A identidade: a versão do formato, o run, a seed, o dataset (com uma impressão digital), o modelo, a métrica principal, as versões do ML Lab e das bibliotecas, e o SHA-256 e o tamanho de cada ficheiro | Sim |
| `manifest.sig` | A assinatura do manifesto (HMAC-SHA256, com a chave desta instalação) | Sim |
| `input.json` | As colunas que o modelo espera, como chegam (antes da preparação): o tipo, o intervalo ou as categorias vistas no treino, um valor típico e se pode faltar; o alvo e as classes | Sim |
| `preparation.json` | A receita e o que ela aprendeu: as limpezas que valem para linhas novas e cada transformação com os seus parâmetros e os valores aprendidos no treino | Sim |
| `model.json` | O modelo, legível: o nome, a família, os hiperparâmetros e o que aprendeu (o mesmo do "O que aprendeu") | Sim |
| `model.joblib` | O modelo treinado, para voltar a prever | A pedido (por omissão, sim) |
| `evaluation.json` | As métricas em cada parte, as matrizes e os avisos | Sim |
| `lineage.json` | A linhagem, da entrada à saída | Sim |
| `training.json` | A curva da loss, passo a passo | A pedido |
| `examples.csv` | Algumas linhas do teste em valores originais, com o valor real e o previsto | A pedido |

**A preparação vai em JSON, sem pickle.** Cada transformação ganha dois métodos:

- `learned()`: o que aprendeu, em valores simples (as medianas, as categorias, as médias e os
  desvios, os limites dos intervalos);
- `restore(params, learned)`: volta a criá-la, já treinada, a partir desses valores.

Um teste de contrato confirma, para cada transformação, que a restaurada transforma igual à
original. As limpezas que valem para linhas novas vão como receita: remover colunas, converter
tipos e limitar extremos (com os limites que se calcularam, porque uma linha sozinha não tem
quartis). As que só fazem sentido numa tabela inteira (remover duplicados, remover linhas)
ficam na linhagem.

**O modelo é o único pickle, e vai assinado.**

- O manifesto tem o SHA-256 de cada ficheiro, e o `manifest.sig` assina o manifesto com uma
  chave desta instalação. A chave são 32 bytes ao acaso, criados no primeiro uso em
  `artifacts/package.key`.
- Ler um pacote confirma os tamanhos, as impressões digitais e a assinatura. Só se leem os nomes
  conhecidos, sem extrair nada para o disco.
- A parte legível abre sempre. O `model.joblib` só se abre com a assinatura certa: um pacote de
  outra instalação, ou alterado, lê-se mas não prevê, e a mensagem diz porquê.
- Sem o modelo ("só a ficha"), o pacote não tem pickle nenhum e pode partilhar-se à vontade.

**A pré-visualização é o ficheiro.** O mesmo código monta os documentos para a pré-visualização
e para o ZIP, e um teste confirma que cada documento pré-visualizado é igual, byte a byte, ao
que fica no ficheiro. O modelo, que é binário, mostra só o tamanho e o que é.

**O backend decide as opções e os avisos** (regra 4 do `CONTRIBUTING.md`):

- **as linhas de exemplo** vêm ligadas nos datasets de exemplo e nos sintéticos, e desligadas
  num ficheiro carregado (os dados podem ser de alguém);
- **os avisos**, cada um com um código estável:
  - as linhas de exemplo são dados reais;
  - o modelo guarda linhas do treino dentro dele. Uma marca nova na ficha técnica,
    `ModelInfo.keeps_training_rows`, diz quais: os dois KNN, o LOF, as três SVM (os vetores de
    suporte), o DBSCAN, o aglomerativo e o t-SNE;
  - o modelo não prevê linhas novas (o t-SNE);
  - o pacote é grande;
  - o pacote sem o modelo não prevê.

**A biblioteca e o descarregar.**

- Os pacotes guardados ficam em `artifacts/packages/<id>.mllab`. Os pacotes são a memória da
  fase 12, que vai prever com eles.
- O descarregar é a primeira rota da API que não devolve JSON (`application/zip`). Lê os bytes
  de um método `@internal` do serviço, como o carregar de um ficheiro faz no sentido contrário.

**Só experiências.** Os runs de reinforcement learning ficam de fora: não têm preparação, e a
tabela Q é outro formato.

**O que se pôs de lado:**

- **tudo em joblib:** é simples, mas fechado e perigoso de partilhar;
- **ONNX** (`skl2onnx`): é portátil, mas não serve os modelos didáticos nem todos os do
  scikit-learn, e traz dependências novas;
- **skops:** é um formato seguro, mas só para o scikit-learn (fica de fora o NumPy e o PyTorch);
- **o formato de modelos do MLflow:** é uma dependência grande, com um registo próprio.

## Consequências

- **Os contratos mudam:**
  - o `Dataset` passa a guardar a preparação aprendida (a `FittedPreparation`), e um run
    reaberto do histórico volta a tê-la, porque a preparação repete-se;
  - o `Transform` ganha `learned` e `restore`, testados em todas as transformações;
  - o `ModelInfo` ganha `keeps_training_rows`.
- **Só há pacote de um run que tenha o modelo:** os desta sessão e os guardados no histórico. Um
  run só registado tem de se reproduzir primeiro (ADR 0020).
- **A chave da instalação** fica em `artifacts/package.key`. Apagá-la não estraga nada: os
  pacotes antigos continuam a ler-se, mas os seus modelos deixam de abrir.
- **O formato tem versão** (`format_version`). Um pacote de um formato mais novo do que o
  laboratório recusa-se com uma mensagem clara. Versões diferentes das bibliotecas dão um
  aviso, porque um pickle do scikit-learn pode não abrir noutra versão.
- **O que se aprende com o ecrã**, nos "?":
  - porque é que a preparação viaja com o modelo;
  - o risco de um pickle;
  - o que é uma ficha de modelo;
  - que há modelos que memorizam os dados;
  - porque se registam as versões.
- **Fica para depois:**
  - prever com um pacote (fase 12);
  - pacotes de agentes de RL;
  - um formato do modelo sem pickle, para partilhar entre instalações.
