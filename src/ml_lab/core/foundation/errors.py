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


class InvalidParameterError(MLLabError, ValueError):
    """Parâmetro com um valor que não é aceite, por exemplo uma proporção de teste de 1,5."""


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


class NotFittedError(MLLabError):
    """Um componente que aprende com os dados foi usado antes de aprender (antes do ``fit``)."""


class DataError(MLLabError):
    """Erro na camada de dados: dados em falta, ilegíveis ou que não servem para o pedido."""


class DataSourceError(DataError):
    """Não foi possível obter os dados: ficheiro em falta, formato não suportado, ilegível..."""


class DataValidationError(DataError):
    """Os dados têm problemas que impedem usá-los.

    Args:
        problems: Explicação de cada problema encontrado.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(" ".join(self.problems) or "Os dados não passaram a validação.")


class ColumnNotFoundError(DataError, LookupError):
    """Não existe nenhuma coluna com o nome pedido; a mensagem sugere a mais parecida.

    Args:
        column: Nome pedido.
        available: Colunas que existem.
    """

    def __init__(self, column: str, available: Iterable[str]) -> None:
        self.column = column
        self.available = tuple(available)
        message = f"Não existe nenhuma coluna chamada '{column}'."
        close = difflib.get_close_matches(column, self.available, n=1)
        if close:
            message += f" Querias dizer '{close[0]}'?"
        if self.available:
            message += f" Colunas: {', '.join(self.available)}."
        super().__init__(message)


class ModelError(MLLabError):
    """Erro de um modelo: configuração inválida, tarefa que não resolve, capacidade em falta."""


class HyperparameterError(ModelError, ValueError):
    """Hiperparâmetros com valores que o modelo não aceita.

    Args:
        model: Nome do modelo.
        problems: Explicação de cada valor recusado.
    """

    def __init__(self, model: str, problems: Iterable[str]) -> None:
        self.model = model
        self.problems = tuple(problems)
        super().__init__(f"Hiperparâmetros inválidos para '{model}': {' '.join(self.problems)}")


class CapabilityError(ModelError):
    """O modelo não tem a capacidade pedida (por exemplo, não dá probabilidades)."""
