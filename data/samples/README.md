# Datasets de exemplo

Versionados no Git (ADR 0010). Para os voltar a descarregar e confirmar com SHA-256:
`uv run python scripts/download_samples.py --force`.

| Ficheiro | Linhas | Alvo | Origem |
|---|---|---|---|
| `iris.csv` | 150 | `species` (3 classes) | R. A. Fisher (1936), cópia do repositório [seaborn-data](https://github.com/mwaskom/seaborn-data) |
| `wine.csv` | 178 | `cultivar` (3 classes) | M. Forina et al., [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/109/wine); cabeçalho com os nomes do scikit-learn |
| `titanic.csv` | 891 | `Survived` (0/1) | Kaggle, «Titanic: Machine Learning from Disaster» (`train.csv`), cópia da documentação do [pandas](https://github.com/pandas-dev/pandas/tree/main/doc/data) |
| `california_housing.csv.gz` | 20 640 | `median_house_value` | Censo dos EUA de 1990, R. K. Pace e R. Barry (1997), StatLib; versão de A. Géron, [«Hands-On Machine Learning»](https://github.com/ageron/handson-ml2) |

As descrições, as regras que cada ficheiro cumpre e o tipo de tarefa estão em
`src/ml_lab/core/data/io/builtin_source.py`.
