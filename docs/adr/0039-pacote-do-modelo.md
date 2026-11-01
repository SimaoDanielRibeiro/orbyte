# 0039 — O pacote do modelo: guardar, levar e voltar a carregar um modelo treinado

- **Estado:** Aceite
- **Data:** 2026-10-29

## Contexto

O pedido: uma secção nova para **guardar o que se preparou e o que os modelos aprenderam**.
Deve mostrar, antes de guardar, uma pré-visualização do ficheiro que vai ser gerado e o que
vai lá dentro. Depois de ver o plano, o pedido ficou mais claro:

- o ficheiro tem de abrir **nesta app, noutro computador, ou numa app parecida**;
- tem de haver a opção de **carregar um modelo já treinado**;
- um ficheiro feito noutra instalação **abre, depois de um aviso**;
- o botão de guardar do treino **leva ao ecrã novo**.

É também a base do laboratório de previsões (fase 12), que vai prever com um modelo guardado.

O que já existe não chega:

- **Guardar uma experiência** (ADR 0020) grava o modelo (joblib) e o treino passo a passo numa
  pasta do laboratório. Serve para reabrir o run nesta instalação, não para o levar.
- **A preparação aprendida não se guarda.** O pipeline aprende com o treino (a mediana para os
  nulos, as categorias, a média e o desvio de cada coluna), transforma as partes e é deitado
  fora (`_assemble`, em `core/data/preparation.py`). Reabrir um run volta a preparar os dados a
  partir da fonte. Sem a fonte (um ficheiro apagado, outro computador), uma linha nova não tem
  como chegar ao modelo na forma em que ele aprendeu.
- **O joblib é um pickle.** Abri-lo corre o código que lá estiver, e só o Python com o ML Lab o
  sabe abrir. Nenhuma outra app o usa.
- **Nada fica legível fora do laboratório:** nem o que o modelo prevê, nem com que dados, nem
  como se portou.

As forças em jogo:

- **legível:** é um laboratório para aprender, e abrir o ficheiro deve ensinar alguma coisa;
- **completo:** prever uma linha nova sem a fonte dos dados;
- **aberto:** outra app consegue lê-lo e usar o modelo;
- **seguro:** um pickle de outra pessoa nunca abre sem a pessoa saber o risco;
- **honesto:** dizer que dados vão dentro do ficheiro (linhas de exemplo, modelos que guardam o
  treino);
- **reprodutível:** saber com que versões se fez.

## Decisão

**Um ficheiro `.mllab`: um ZIP com nomes fixos e um formato documentado.**

- Abre-se com qualquer programa de ZIP.
- O formato fica descrito em `docs/formato-mllab.md`, com a versão (`format_version`) e o que
  cada ficheiro tem, para outra app o saber ler.
- Os nomes dos ficheiros lá dentro estão em inglês, como os de `artifacts/runs/`; os textos estão
  em português.

| Ficheiro | O que tem | Sempre? |
|---|---|---|
| `README.md` | A ficha do modelo, para pessoas: o que prevê, com que dados, como se portou, os limites, o que vai no pacote e como se usa | Sim |
| `manifest.json` | A identidade: a versão do formato, o run, a seed, o dataset (com uma impressão digital), o modelo, a métrica principal, as versões do ML Lab e das bibliotecas, e o SHA-256 e o tamanho de cada ficheiro | Sim |
| `manifest.sig` | A assinatura do manifesto (HMAC-SHA256, com a chave desta instalação) | Sim |
| `input.json` | As colunas que o modelo espera, como chegam (antes da preparação): o tipo, o intervalo ou as categorias vistas no treino, um valor típico e se pode faltar; o alvo e as classes | Sim |
| `preparation.json` | A receita e o que ela aprendeu: as limpezas que valem para linhas novas, cada transformação com os seus parâmetros e os valores aprendidos no treino, e a linhagem | Sim |
| `run.json` | O run: a configuração, as métricas em cada parte, o que o modelo aprendeu, a curva da loss e os avisos (o mesmo documento que o histórico guarda) | Sim |
| `model.joblib` | O modelo treinado, para este laboratório | A pedido (por omissão, sim) |
| `model.onnx` | A preparação e o modelo num só grafo ONNX, para outras apps | Sempre que dá |
| `animation.json`, `network.json` | O treino passo a passo, para o rever depois de carregar | A pedido |
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

**Para outras apps, o ONNX.** É o formato aberto para trocar modelos treinados: o ONNX Runtime
abre-o em Python, C#, Java, JavaScript e noutras linguagens, e há apps que o leem diretamente.

- O `model.onnx` junta a preparação e o modelo num só grafo. Recebe as colunas como chegam (os
  números como números, o texto como texto, um texto vazio onde falta o valor) e devolve a
  previsão e as probabilidades.
- Um teste confirma, para cada modelo que o tem, que o ONNX Runtime dá as mesmas previsões do
  laboratório.
- Quando não dá (o t-SNE, uma preparação com datas), o pacote vai sem ele. A pré-visualização
  diz porquê.
- Traz três dependências novas: `onnx`, `skl2onnx` e `onnxruntime`. É a exceção deliberada à
  stack enxuta (ADR 0003), porque abrir o ficheiro noutra app é parte do pedido. Como o PyTorch,
  o ONNX fica num só sítio (`core/export/`), com uma regra de arquitetura que o confirma.

**Carregar um modelo treinado.**

- Carrega-se um `.mllab` no ecrã Guardar. O pacote entra na biblioteca, e o run entra no
  histórico como um run guardado (marcado como carregado), com o modelo e o treino passo a
  passo.
- A partir daí, abre-se como os outros:
  - no Treinar e avaliar;
  - no Explicar, quando os dados se podem preparar outra vez (um dataset de exemplo, um
    sintético ou um ficheiro que exista);
  - no Prever, na fase 12.

**Um pacote de outra instalação abre depois de um aviso.**

- O manifesto tem o SHA-256 de cada ficheiro, e o `manifest.sig` assina-o com uma chave desta
  instalação: 32 bytes ao acaso, criados no primeiro uso em `artifacts/package.key`.
- Ler um pacote confirma os tamanhos, as impressões digitais e a assinatura. Só se leem os nomes
  conhecidos, sem extrair nada para o disco.
- **Um pacote feito aqui** abre logo.
- **Um pacote de outra instalação** lê-se logo: a ficha, a avaliação, a preparação. O modelo só
  abre depois de um aviso explícito: o `model.joblib` é um pickle, e abri-lo corre o código de
  quem o fez. A pessoa confirma que confia em quem lhe deu o ficheiro.
- **Um pacote estragado ou alterado** (as impressões digitais não conferem) nunca abre o modelo.
- **Sem o modelo** ("só a ficha"), o pacote não tem pickle nenhum e pode partilhar-se à vontade.
- Versões diferentes das bibliotecas dão um aviso, porque um pickle do scikit-learn pode não
  abrir noutra versão.

**A pré-visualização é o ficheiro.** O mesmo código monta os documentos para a pré-visualização
e para o ZIP, e um teste confirma que cada documento pré-visualizado é igual, byte a byte, ao
que fica no ficheiro. Os ficheiros binários (o modelo, o ONNX) mostram só o tamanho e o que são.

**O backend decide as opções e os avisos** (regra 4 do `CONTRIBUTING.md`):

- **as linhas de exemplo** vêm ligadas nos datasets de exemplo e nos sintéticos, e desligadas
  num ficheiro carregado (os dados podem ser de alguém);
- **os avisos**, cada um com um código estável:
  - as linhas de exemplo são dados reais;
  - o modelo guarda linhas do treino dentro dele. Uma marca nova na ficha técnica,
    `ModelInfo.keeps_training_rows`, diz quais: os dois KNN, o LOF, as três SVM (os vetores de
    suporte), o DBSCAN, o aglomerativo e o t-SNE;
  - o modelo não prevê linhas novas (o t-SNE);
  - o pacote vai sem o ONNX, e porquê;
  - o pacote é grande;
  - o pacote sem o modelo não prevê.

**O ecrã Guardar faz as duas coisas.** O botão do treino, que era "Guardar experiência", passa a
chamar-se "Guardar" e leva ao ecrã novo, com o run escolhido. Guardar lá grava o pacote na
biblioteca, descarrega-o e guarda também o run no histórico, como o botão fazia.

**A biblioteca e o descarregar.**

- Os pacotes ficam em `artifacts/packages/<id>/`.
- O descarregar é a primeira rota da API que não devolve JSON (`application/zip`). Lê os bytes
  de um método `@internal` do serviço; o carregar recebe-os como o carregar de um dataset.

**Só experiências.** Os runs de reinforcement learning ficam de fora: não têm preparação, e a
tabela Q é outro formato. O treino deles mantém o "Guardar experiência" de sempre.

**O que se pôs de lado:**

- **tudo em joblib:** é simples, mas fechado, perigoso de partilhar e só serve o Python;
- **skops:** é um formato seguro, mas só para o scikit-learn (fica de fora o NumPy e o PyTorch),
  e nenhuma outra app o lê;
- **o formato de modelos do MLflow:** é uma dependência grande, com um registo próprio.

## Consequências

- **Os contratos mudam:**
  - o `Dataset` passa a guardar a preparação aprendida (a `FittedPreparation`), e um run
    reaberto do histórico volta a tê-la, porque a preparação repete-se;
  - o `Transform` ganha `learned` e `restore`, testados em todas as transformações;
  - as limpezas guardam o que aprenderam (os limites dos extremos);
  - o `ModelInfo` ganha `keeps_training_rows`;
  - cada modelo diz se tem conversão para ONNX, e um teste de contrato confirma as previsões.
- **Só há pacote de um run que tenha o modelo:** os desta sessão e os guardados no histórico. Um
  run só registado tem de se reproduzir primeiro (ADR 0020).
- **A chave da instalação** fica em `artifacts/package.key`. Apagá-la não estraga nada: os
  pacotes antigos passam a ser tratados como de outra instalação (abrem depois do aviso).
- **O formato tem versão.** Um pacote de um formato mais novo do que o laboratório recusa-se com
  uma mensagem clara.
- **Três dependências novas** (`onnx`, `skl2onnx`, `onnxruntime`), confinadas a `core/export/`.
- **O que se aprende com o ecrã**, nos "?":
  - porque é que a preparação viaja com o modelo;
  - o risco de um pickle;
  - o que é uma ficha de modelo;
  - o que é o ONNX;
  - que há modelos que memorizam os dados;
  - porque se registam as versões.
- **Fica para depois:**
  - prever com um pacote, e com um modelo ONNX de outra app (fase 12);
  - os pacotes dos agentes de reinforcement learning.
