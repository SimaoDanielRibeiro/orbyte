"""Exceções do domínio do ML Lab.

Todas herdam de ``MLLabError``, por isso ``except MLLabError`` apanha qualquer erro
do projeto. As mensagens explicam o que aconteceu e o que fazer a seguir.
"""

import difflib
from collections.abc import Iterable


class MLLabError(Exception):
    """Base de todas as exceções do ML Lab."""


class ConfigurationError(MLLabError):
    """Configuração inválida: variável de ambiente, ficheiro ``.env`` ou parâmetro."""


class InvalidSeedError(MLLabError, ValueError):
    """Seed que não é um inteiro ou que está fora do intervalo aceite."""


class RegistryError(MLLabError):
    """Erro no registo de componentes (plugins)."""


class DuplicateComponentError(RegistryError):
    """Já existe um componente registado com o mesmo nome.

    Args:
        registry: Nome do registo, por exemplo ``"models"``.
        name: Nome repetido.
    """

    def __init__(self, registry: str, name: str) -> None:
        self.registry = registry
        self.name = name
        super().__init__(
            f"Já existe um componente chamado '{name}' no registo '{registry}'. "
            "Escolhe outro nome ou remove o registo repetido."
        )


class ComponentNotFoundError(RegistryError, LookupError):
    """Não existe nenhum componente com o nome pedido.

    A mensagem sugere o nome mais parecido e lista os disponíveis.

    Args:
        registry: Nome do registo, por exemplo ``"models"``.
        name: Nome pedido.
        available: Nomes registados.
    """

    def __init__(self, registry: str, name: str, available: Iterable[str]) -> None:
        self.registry = registry
        self.name = name
        self.available = tuple(sorted(available))
        super().__init__(self._build_message())

    def _build_message(self) -> str:
        message = (
            f"Não existe nenhum componente chamado '{self.name}' no registo '{self.registry}'."
        )
        if not self.available:
            return (
                f"{message} O registo está vazio: os componentes registam-se quando o seu "
                "módulo é importado, por isso confirma que o ficheiro existe e usa o decorador "
                "register."
            )
        close = difflib.get_close_matches(self.name, self.available, n=1)
        if close:
            message += f" Querias dizer '{close[0]}'?"
        return f"{message} Disponíveis: {', '.join(self.available)}."
