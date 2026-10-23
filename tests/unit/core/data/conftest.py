"""Tabelas pequenas feitas à mão para testar cada etapa da camada de dados."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def passengers() -> pd.DataFrame:
    """Um "mini Titanic" com tudo o que a camada de dados tem de saber tratar.

    - ``PassengerId``: identificador (inteiros seguidos);
    - ``Name``: texto livre (um valor diferente por linha, com espaços);
    - ``Pclass``: inteiro com poucos valores;
    - ``Sex`` e ``Embarked``: categóricas (``Embarked`` com um nulo);
    - ``Age``: numérica com nulos;
    - ``Fare``: numérica;
    - ``Survived``: alvo booleano (0/1), com as duas classes equilibradas.
    """
    return pd.DataFrame(
        {
            "PassengerId": list(range(1, 21)),
            "Name": [f"Passageiro número {i}" for i in range(1, 21)],
            "Pclass": [1, 2, 3, 3, 1, 2, 3, 3, 2, 1] * 2,
            "Sex": ["male", "female"] * 10,
            "Age": [
                *(22.0, 38.0, np.nan, 35.0, 54.0, np.nan, 2.0, 27.0, 14.0, 4.0),
                *(58.0, 20.0, 39.0, np.nan, 55.0, 31.0, 34.0, 15.0, 28.0, 8.0),
            ],
            "Embarked": ["S", "C", "S", "Q", "S", "C", "S", "S", None, "Q"] * 2,
            "Fare": [7.25, 71.28, 7.92, 53.1, 8.05, 8.46, 51.86, 21.07, 11.13, 30.07] * 2,
            "Survived": [0, 1, 1, 1, 0, 0, 0, 1, 1, 0] * 2,
        }
    )


@pytest.fixture
def numbers() -> pd.DataFrame:
    """Duas features numéricas e um alvo contínuo, sem nulos."""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=40)
    x2 = rng.normal(size=40)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": 3 * x1 - 2 * x2 + 1})
