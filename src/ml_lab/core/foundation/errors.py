"""Exceções do domínio do Orbyte.

Todas herdam de ``MLLabError``, por isso ``except MLLabError`` apanha qualquer erro
do projeto. As mensagens, em inglês como todo o site (ADR 0043), explicam o que aconteceu e
o que fazer a seguir.
"""

import difflib
from collections.abc import Iterable


class MLLabError(Exception):
    """Base de todas as exceções do Orbyte."""


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
            f"There's already a component called '{name}' in the '{registry}' registry. "
            "Choose another name or remove the repeated registration."
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
        message = f"There's no component called '{self.name}' in the '{self.registry}' registry."
        if not self.available:
            return (
                f"{message} The registry is empty: components register themselves when their "
                "module is imported, so check that the file exists and uses the register "
                "decorator."
            )
        close = difflib.get_close_matches(self.name, self.available, n=1)
        if close:
            message += f" Did you mean '{close[0]}'?"
        return f"{message} Available: {', '.join(self.available)}."


class NotFittedError(MLLabError):
    """Um componente que aprende com os dados foi usado antes de aprender (antes do ``fit``)."""


class DataError(MLLabError):
    """Erro na camada de dados: dados em falta, ilegíveis ou que não servem para o pedido."""


class DataSourceError(DataError):
    """Não foi possível obter os dados: ficheiro em falta, formato não suportado, ilegível..."""


class DatasetNotFoundError(MLLabError, LookupError):
    """Não há nenhum dataset carregado com o identificador pedido.

    Args:
        dataset_id: O identificador pedido.
    """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(
            f"There's no loaded dataset with the identifier '{dataset_id}': load it first in Data."
        )


class DataValidationError(DataError):
    """Os dados têm problemas que impedem usá-los.

    Args:
        problems: Explicação de cada problema encontrado.
    """

    def __init__(self, problems: Iterable[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(" ".join(self.problems) or "The data didn't pass validation.")


class ColumnNotFoundError(DataError, LookupError):
    """Não existe nenhuma coluna com o nome pedido; a mensagem sugere a mais parecida.

    Args:
        column: Nome pedido.
        available: Colunas que existem.
    """

    def __init__(self, column: str, available: Iterable[str]) -> None:
        self.column = column
        self.available = tuple(available)
        message = f"There's no column called '{column}'."
        close = difflib.get_close_matches(column, self.available, n=1)
        if close:
            message += f" Did you mean '{close[0]}'?"
        if self.available:
            message += f" Columns: {', '.join(self.available)}."
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
        super().__init__(f"Invalid hyperparameters for '{model}': {' '.join(self.problems)}")


class CapabilityError(ModelError):
    """O modelo não tem a capacidade pedida (por exemplo, não dá probabilidades)."""


class RunNotInMemoryError(ModelError, LookupError):
    """O run já não está em memória: só os últimos ficam, com o modelo e os dados.

    Args:
        run_id: O identificador do run.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Run {run_id} is no longer in memory (only the latest ones stay). If you saved it, "
            "open it in History, and the model comes back with it; if not, train it again."
        )


class JobNotFoundError(MLLabError, LookupError):
    """Não há nenhum treino em segundo plano com o identificador pedido.

    Args:
        run_id: O run que o treino produz.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"There's no training '{run_id}' in this session: the oldest ones leave the memory, "
            "and a server that restarts starts with none."
        )


class SearchNotFoundError(MLLabError, LookupError):
    """Não há nenhuma procura dos agentes com o identificador pedido.

    Args:
        search_id: O identificador pedido.
    """

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        super().__init__(
            f"There's no search '{search_id}' in this session: only the latest ones stay in "
            "memory, and a server that restarts starts with none. Each search's winner is in "
            "the history (History)."
        )


class CandidateNotFoundError(MLLabError, LookupError):
    """A procura não tem nenhum candidato com o número pedido.

    Args:
        search_id: A procura.
        number: O número pedido.
        count: Quantos candidatos a procura tem.
    """

    def __init__(self, search_id: str, number: int, count: int) -> None:
        self.search_id = search_id
        self.number = number
        where = f"it has candidates 1 to {count}" if count else "it measured no candidate"
        super().__init__(f"The search '{search_id}' has no candidate {number}: {where}.")


class TrainingStateError(MLLabError):
    """O pedido não combina com o estado de um treino em segundo plano (decorre, ou falhou)."""


class TrainingInProgressError(TrainingStateError):
    """O treino ainda decorre: o resultado só existe quando terminar.

    Args:
        run_id: O run que o treino produz.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Run '{run_id}' is still training: the result only exists when the training "
            "finishes (the training's state says when)."
        )


class TrainingFailedError(TrainingStateError):
    """O treino falhou, por isso não há resultado.

    Args:
        run_id: O run que o treino devia produzir.
        reason: A explicação do erro que o fez falhar.
    """

    def __init__(self, run_id: str, reason: str) -> None:
        self.run_id = run_id
        self.reason = reason
        super().__init__(f"The training of run '{run_id}' failed: {reason}")


class TrackingError(MLLabError):
    """Erro no registo de experiências: a base de dados ou os ficheiros de um run."""


class RunNotFoundError(TrackingError, LookupError):
    """Não há nenhum run com o identificador pedido no registo.

    Args:
        run_id: O identificador pedido.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"There's no run '{run_id}' in the experiment log: it may have been deleted."
        )


class PackageError(DataError):
    """Um ficheiro que não é um pacote do Orbyte, ou que está estragado (ADR 0039)."""


class PackageNotFoundError(TrackingError, LookupError):
    """Não há nenhum pacote com o identificador pedido na biblioteca.

    Args:
        package_id: O identificador pedido.
    """

    def __init__(self, package_id: str) -> None:
        self.package_id = package_id
        super().__init__(
            f"There's no package '{package_id}' in the library: it may have been deleted."
        )


class LanguageModelError(MLLabError):
    """O modelo de linguagem não respondeu, ou respondeu algo que não se percebe (ADR 0042).

    Acontece quando o Ollama não está a correr, quando o modelo não está descarregado, ou
    quando a resposta não é o JSON pedido. A mensagem diz como se resolve.
    """
