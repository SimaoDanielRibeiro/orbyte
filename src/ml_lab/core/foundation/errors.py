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


class DatasetNotFoundError(MLLabError, LookupError):
    """Não há nenhum dataset carregado com o identificador pedido.

    Args:
        dataset_id: O identificador pedido.
    """

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(
            f"Não há nenhum dataset carregado com o identificador '{dataset_id}': carrega-o "
            "primeiro no ecrã Dados."
        )


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


class RunNotInMemoryError(ModelError, LookupError):
    """O run já não está em memória: só os últimos ficam, com o modelo e os dados.

    Args:
        run_id: O identificador do run.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"O run {run_id} já não está em memória (só ficam os últimos). Se o guardaste, "
            "abre-o no ecrã Experiências, que o modelo volta com ele; se não, treina-o outra vez."
        )


class JobNotFoundError(MLLabError, LookupError):
    """Não há nenhum treino em segundo plano com o identificador pedido.

    Args:
        run_id: O run que o treino produz.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        super().__init__(
            f"Não há nenhum treino '{run_id}' nesta sessão: os mais antigos saem da memória, e "
            "um servidor que reinicia começa sem nenhum."
        )


class SearchNotFoundError(MLLabError, LookupError):
    """Não há nenhuma procura dos agentes com o identificador pedido.

    Args:
        search_id: O identificador pedido.
    """

    def __init__(self, search_id: str) -> None:
        self.search_id = search_id
        super().__init__(
            f"Não há nenhuma procura '{search_id}' nesta sessão: só as últimas ficam em memória, "
            "e um servidor que reinicia começa sem nenhuma. O vencedor de cada procura está no "
            "histórico (ecrã Experiências)."
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
        where = f"tem os candidatos de 1 a {count}" if count else "não mediu nenhum candidato"
        super().__init__(f"A procura '{search_id}' não tem o candidato {number}: {where}.")


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
            f"O run '{run_id}' ainda está a treinar: o resultado só existe quando o treino "
            "terminar (o estado do treino diz quando)."
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
        super().__init__(f"O treino do run '{run_id}' falhou: {reason}")


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
            f"Não há nenhum run '{run_id}' no registo de experiências: pode ter sido apagado."
        )


class PackageError(DataError):
    """Um ficheiro que não é um pacote do ML Lab, ou que está estragado (ADR 0039)."""


class PackageNotFoundError(TrackingError, LookupError):
    """Não há nenhum pacote com o identificador pedido na biblioteca.

    Args:
        package_id: O identificador pedido.
    """

    def __init__(self, package_id: str) -> None:
        self.package_id = package_id
        super().__init__(
            f"Não há nenhum pacote '{package_id}' na biblioteca: pode ter sido apagado."
        )
