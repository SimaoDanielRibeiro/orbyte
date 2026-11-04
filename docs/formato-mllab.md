# O formato `.mllab` (versão 1)

Um ficheiro `.mllab` é o pacote de um modelo treinado no ML Lab (ADR 0039): tudo o que é preciso
para perceber o modelo e para voltar a prever com ele, sem a fonte dos dados. Esta página
descreve-o para quem o quiser ler noutra app.

Em resumo:

- é um **ZIP** normal (abre em qualquer programa de ZIP), com os ficheiros comprimidos e sem
  pastas;
- os nomes dos ficheiros são **fixos** e a ordem também;
- o **manifesto** diz o tamanho e o SHA-256 de cada ficheiro, e está **assinado** pela
  instalação que o fez;
- o modelo vem em dois sabores: **`model.joblib`**, para o ML Lab (um pickle), e
  **`model.onnx`**, para qualquer app (sem código de ninguém lá dentro).

## Os ficheiros

Por esta ordem dentro do ZIP:

| Ficheiro | Sempre? | O que tem |
|---|---|---|
| `README.md` | sim | A ficha do modelo, para pessoas: o que prevê, como foi feito, como se portou, os limites e como se usa. |
| `manifest.json` | sim | A identidade do pacote, as versões das bibliotecas e a impressão digital de cada ficheiro. |
| `manifest.sig` | sim | A assinatura do manifesto (diz se o pacote foi feito numa instalação e se não mudou desde então). |
| `input.json` | sim | As colunas que o modelo espera, como chegam, com o que se viu no treino. |
| `preparation.json` | sim | A receita da preparação, o que ela aprendeu no treino e a linhagem. |
| `run.json` | sim | O run: a configuração, as métricas em cada parte, o que o modelo aprendeu e a curva da loss. |
| `model.joblib` | não | O modelo treinado, para voltar a prever no ML Lab. |
| `model.onnx` | não | A preparação e o modelo num só grafo ONNX, para o ONNX Runtime. |
| `animation.json` | não | O treino passo a passo: o que o modelo respondia em cada snapshot. |
| `network.json` | não | O que cada camada da rede fez em cada snapshot. |
| `examples.csv` | não | Algumas linhas do teste, como chegaram, com o valor real e o previsto. |

Um leitor do formato deve recusar nomes que não estejam nesta lista (e nunca os extrair para o
disco), e ler o ZIP com um limite: o ML Lab não abre pacotes com mais de 512 MB, comprimidos ou
não.

## `manifest.json`

Com valores de exemplo:

```json
{
  "format": "ml-lab-package",
  "format_version": 1,
  "name": "Titanic — Regressão logística",
  "created_at": "2026-10-29T14:32:05+00:00",
  "run_id": "run-c36c0f26",
  "seed": 42,
  "task_type": "binary_classification",
  "target": "Survived",
  "dataset": {
    "name": "Titanic",
    "source": { "kind": "sample", "params": { "name": "titanic" } },
    "rows": { "train": 712, "test": 179 }
  },
  "model": {
    "name": "logistic_regression",
    "display_name": "Regressão logística",
    "family": "linear",
    "implementation": "sklearn",
    "hyperparameters": { "C": 1.0, "max_iter": 1000 }
  },
  "primary_metric": {
    "name": "accuracy", "label": "Exatidão", "partition": "test", "value": 0.81, "as_percent": true
  },
  "contents": { "model": true, "onnx": true, "examples": 5, "training": false },
  "versions": { "python": "3.12.8", "ml-lab": "0.1.0", "numpy": "2.5.4", "scikit-learn": "1.9.1" },
  "files": [
    { "path": "README.md", "bytes": 4210, "sha256": "41c0…" }
  ]
}
```

- `format` e `format_version` dizem o que é: uma versão maior do que a que o leitor conhece
  recusa-se, com a versão que falta.
- `files` lista **todos os outros ficheiros** (menos o próprio manifesto e a assinatura), com o
  tamanho em bytes e o SHA-256 do conteúdo descomprimido. Um ficheiro que não confere, que falta
  ou que sobra torna o pacote **estragado**: lê-se o que der, e o modelo não abre.
- `versions` regista as bibliotecas de que o `model.joblib` depende: noutra versão, o pickle
  pode não abrir, ou prever de outra maneira.

## `manifest.sig`

O HMAC-SHA256 dos bytes de `manifest.json`, em hexadecimal, com a chave da instalação que fez o
pacote (`artifacts/package.key`, criada na primeira vez, só legível por quem a criou). Como o
manifesto tem o SHA-256 de cada ficheiro, a assinatura cobre o pacote inteiro.

A assinatura não diz quem fez o pacote; diz se foi **esta** instalação. Um pacote feito aqui
abre logo; um de fora (ou sem assinatura) abre a ficha, a avaliação e a preparação, e o modelo
só depois de a pessoa confirmar que confia em quem lho deu: abrir um pickle corre o código de
quem o fez.

## `input.json`

As colunas que o modelo espera, como chegam (antes das transformações), tiradas das linhas de
treino:

```json
{
  "target": { "name": "Survived", "classes": ["0", "1"] },
  "columns": [
    { "name": "Pclass", "kind": "numeric", "dtype": "int64", "min": 1, "max": 3, "typical": 3.0,
      "missing_in_train": 0, "can_be_missing": true },
    { "name": "Sex", "kind": "categorical", "dtype": "str", "categories": ["female", "male"],
      "typical": "male", "missing_in_train": 0, "can_be_missing": true }
  ],
  "left_out": ["PassengerId", "Name", "Ticket", "Cabin"]
}
```

`kind` é `numeric`, `categorical`, `boolean` ou `datetime`. `can_be_missing` diz se a preparação
preenche o valor quando ele falta.

## `preparation.json`

A receita, pela ordem em que corre, com o que cada passo aprendeu no treino:

- `cleaning`: os passos da limpeza (`operation`, `params`, `learned`), e se cada um se repete
  numa linha nova (`on_new_rows`: limitar um extremo e converter um tipo, sim; tirar
  duplicados, não);
- `transforms`: as transformações (`impute`, `binning`, `boolean_to_number`, `one_hot`,
  `polynomial_features`, `standard_scaler`, `minmax_scaler`, `date_features`,
  `drop_columns`), cada uma com `params`, `learned` e uma descrição;
- `features`: as features que saem, pela ordem que o modelo conhece;
- `lineage`: a linhagem dos dados, passo a passo.

Com `input.json` e `preparation.json`, o ML Lab volta a montar a preparação sem pickle.

## `model.onnx`

A preparação e o modelo num só grafo ONNX (opset 19, `ai.onnx.ml` 3, IR 9), que abre no ONNX
Runtime em Python, C#, Java, JavaScript ou C++.

**Entradas:** uma por coluna de `input.json` (menos as datas), com o nome da coluna e a forma
`(N, 1)`, uma linha por exemplo:

| `kind` | Tipo | Quando falta |
|---|---|---|
| `numeric` | `double` | `NaN` |
| `boolean` | `double` (0 ou 1) | `NaN` |
| `categorical` | `string`, como o texto do valor (`3` é `"3"`) | `""` |

Uma categoria que o treino nunca viu dá zeros em todas as colunas do one-hot, como no ML Lab.

**Saídas**, pelo que o modelo faz:

| Tarefa | Saídas |
|---|---|
| Classificação | `label` (`string`, `(N,)`: a classe, pelo nome) e, se o modelo as der, `probabilities` (`double`, `(N, classes)`) |
| Regressão | `prediction` (`double`, `(N,)`) |
| Clustering | `cluster` (`int64`, `(N,)`; -1 é ruído) |
| Anomalias | `anomaly` (`int64`, `(N,)`: 1 é anómala) |
| Redução de dimensionalidade | `projection` (`double`, `(N, k)`) |

**Metadados** (`metadata_props`): `ml_lab.format` (`ml-lab-onnx`), `ml_lab.model`,
`ml_lab.kind`, `ml_lab.target`, `ml_lab.classes` (JSON: as classes, pela ordem das colunas de
`probabilities`) e `ml_lab.features` (JSON).

Antes de entrar no pacote, o ficheiro corre com as linhas de exemplo e tem de dar as mesmas
previsões do ML Lab (as mesmas classes; os números a menos de 1e-4 relativo, a precisão dos
modelos que correm em `float32`). Se não der, fica de fora, e a ficha diz porquê. Também fica de
fora quando a preparação tem datas, ou quando o modelo não sabe colocar pontos novos (o t-SNE).

Em Python:

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
linha = {
    "Pclass": np.array([[3.0]]),
    "Sex": np.array([["male"]], dtype=object),
    "Age": np.array([[28.0]]),
}
label, probabilities = session.run(None, linha)
```

(A ficha de cada pacote traz este exemplo já com as colunas do modelo.)

## `examples.csv`

Algumas linhas do teste, como chegaram, com o alvo e uma coluna `predicted` com o que o modelo
previu (o nome em inglês, como todo o site; ADR 0043). Servem para confirmar que o modelo prevê o mesmo noutro sítio. Vêm por omissão nos
datasets de exemplo e nos sintéticos, e não nos ficheiros carregados (podem ser dados de alguém).

## `model.joblib`

O objeto do modelo do ML Lab, em joblib (um pickle). Só o ML Lab o sabe usar, e só se deve abrir
de quem se confia: para outras apps, existe o `model.onnx`.
