"""ML Lab: laboratório de machine learning para aprender a treinar, observar e avaliar modelos.

O pacote está organizado em camadas, e as dependências só apontam para dentro
(ver docs/architecture.md):

- ``ml_lab.core.foundation``: tipos, erros, registo de componentes, eventos e seeds;
- ``ml_lab.core``: dados, modelos, treino, avaliação e registo de experiências;
- ``ml_lab.core.services``: a única porta de entrada do núcleo;
- ``ml_lab.api``: a API HTTP (fase 7).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ml-lab")
except PackageNotFoundError:  # código a correr sem estar instalado
    __version__ = "0.0.0"

__all__ = ["__version__"]
