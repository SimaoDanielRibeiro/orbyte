"""Gestão central de seeds: toda a aleatoriedade do ML Lab parte de uma seed explícita.

Cada experiência tem uma seed raiz. Cada componente (divisão dos dados, modelo,
validação cruzada...) recebe uma seed própria, derivada da raiz pelo nome do
componente. Assim:

- a mesma seed raiz reproduz exatamente a mesma experiência;
- as seeds dos componentes são independentes: trocar o modelo não muda a divisão
  dos dados, porque a seed da divisão só depende da raiz e do nome ``"split"``.

Este módulo usa só a biblioteca padrão. Quem precisar de um gerador do NumPy
faz ``numpy.random.default_rng(seeds.seed_for("split"))``.

Exemplo:
    >>> seeds = SeedManager(42)
    >>> seeds.seed_for("split") == SeedManager(42).seed_for("split")
    True
    >>> seeds.seed_for("split") == seeds.seed_for("model")
    False
"""

import hashlib
import json
import operator
import random
import secrets
from dataclasses import dataclass
from typing import Final, Self, SupportsIndex

from ml_lab.core.foundation.errors import InvalidSeedError

MAX_SEED: Final = 2**32 - 1
"""Maior seed aceite: o NumPy e o scikit-learn usam inteiros de 32 bits sem sinal."""


def validate_seed(seed: object) -> int:
    """Confirma que ``seed`` é um inteiro entre 0 e ``MAX_SEED``.

    Aceita inteiros do Python e do NumPy; recusa ``bool``, ``float`` e texto.

    Args:
        seed: Valor a validar.

    Returns:
        A seed como ``int`` do Python.

    Raises:
        InvalidSeedError: Se não for um inteiro ou estiver fora do intervalo.
    """
    if isinstance(seed, bool) or not isinstance(seed, SupportsIndex):
        raise InvalidSeedError(f"The seed must be an integer, not {type(seed).__name__}.")
    value = operator.index(seed)
    if not 0 <= value <= MAX_SEED:
        raise InvalidSeedError(f"The seed must be between 0 and {MAX_SEED}; got {value}.")
    return value


def generate_seed() -> int:
    """Gera uma seed nova a partir da entropia do sistema operativo.

    Usa-se quando o utilizador não indica nenhuma seed. A seed gerada fica
    registada com a experiência, por isso o run continua reproduzível.
    """
    return secrets.randbelow(MAX_SEED + 1)


def derive_seed(seed: int, name: str) -> int:
    """Deriva a seed de um componente a partir de outra seed, de forma determinística.

    A mesma combinação ``(seed, name)`` dá sempre o mesmo resultado, em qualquer
    computador e versão do Python. Por isso usa-se um hash criptográfico (BLAKE2)
    e não ``hash()``, cujo resultado para texto muda de execução para execução.

    Args:
        seed: Seed de origem, normalmente a seed raiz da experiência.
        name: Nome do componente, por exemplo ``"split"`` ou ``"model"``.

    Returns:
        Uma seed entre 0 e ``MAX_SEED``.

    Raises:
        InvalidSeedError: Se ``seed`` não for válida.
        ValueError: Se ``name`` estiver vazio.
    """
    value = validate_seed(seed)
    if not name:
        raise ValueError("The component's name can't be empty.")
    payload = json.dumps([value, name]).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=4).digest()
    return int.from_bytes(digest, "big")


def make_rng(seed: int) -> random.Random:
    """Cria um gerador do módulo ``random``, separado do gerador global do Python."""
    return random.Random(validate_seed(seed))


@dataclass(frozen=True, slots=True)
class SeedManager:
    """Seed raiz de uma experiência e as seeds derivadas para cada componente.

    Attributes:
        seed: Seed raiz, entre 0 e ``MAX_SEED``.
    """

    seed: int

    def __post_init__(self) -> None:
        # Normaliza (por exemplo, um inteiro do NumPy passa a int) sem perder o frozen.
        object.__setattr__(self, "seed", validate_seed(self.seed))

    @classmethod
    def create(cls, seed: int | None = None) -> Self:
        """Cria o gestor com a seed dada ou, se for ``None``, com uma seed nova."""
        return cls(generate_seed() if seed is None else seed)

    def seed_for(self, name: str) -> int:
        """Seed do componente ``name``: sempre a mesma para a mesma seed raiz."""
        return derive_seed(self.seed, name)

    def child(self, name: str) -> Self:
        """Gestor para um subcomponente, com a raiz derivada de ``name``.

        Serve para componentes com partes aleatórias próprias, por exemplo cada
        fold da validação cruzada: ``seeds.child("cv").seed_for("fold_0")``.
        """
        return type(self)(self.seed_for(name))

    def rng(self, name: str) -> random.Random:
        """Gerador do módulo ``random`` para o componente ``name``."""
        return make_rng(self.seed_for(name))
