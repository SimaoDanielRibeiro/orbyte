"""Registo genérico de componentes (padrão plugin).

Cada componente (modelo, métrica, fonte de dados...) regista-se sozinho com um
decorador, e o resto do sistema encontra-o pelo nome. Assim não há
``if algoritmo == "knn"`` espalhados pelo código, e acrescentar um algoritmo é
criar um ficheiro novo, sem alterar os existentes.

Exemplo:
    >>> TOOLS: Registry[type] = Registry("tools")
    >>> @TOOLS.register("hammer")
    ... class Hammer:
    ...     pass
    >>> TOOLS.get("hammer") is Hammer
    True
    >>> TOOLS.names()
    ['hammer']
"""

import importlib
import logging
import pkgutil
import re
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Final, cast

from ml_lab.core.foundation.errors import (
    ComponentNotFoundError,
    DuplicateComponentError,
    RegistryError,
)
from ml_lab.core.foundation.types import TaskType

logger = logging.getLogger(__name__)

_NAME_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]*")

type TaskTypesReader[T] = Callable[[T], Iterable[TaskType]]
"""Função que diz que tipos de tarefa um componente resolve."""


@dataclass(frozen=True, slots=True)
class _Entry[T]:
    component: T
    task_types: frozenset[TaskType]


class Registry[T]:
    """Guarda componentes do tipo ``T`` pelo nome.

    Args:
        name: Nome do registo, usado nas mensagens de erro (por exemplo ``"models"``).
        task_types: Função que lê os tipos de tarefa de um componente, por exemplo
            ``lambda model: model.info.task_types``. Só os registos com esta função
            podem ser filtrados com ``for_task``.
        discover: Pacote cujos módulos são todos importados na primeira consulta,
            para que os decoradores dos componentes corram. Com isto, um ficheiro
            novo nessa pasta fica registado sem mexer em mais nenhum.
    """

    def __init__(
        self,
        name: str,
        *,
        task_types: TaskTypesReader[T] | None = None,
        discover: str | None = None,
    ) -> None:
        self._name = name
        self._read_task_types = task_types
        self._discover = discover
        self._discovered = False
        self._discovering = False
        self._lock = threading.RLock()
        self._entries: dict[str, _Entry[T]] = {}

    @property
    def name(self) -> str:
        """Nome do registo."""
        return self._name

    def register[C](self, name: str) -> Callable[[C], C]:
        """Decorador que regista o componente decorado com o nome ``name``.

        O componente é devolvido tal como está, por isso a classe ou função
        decorada continua igual.

        Args:
            name: Nome único, em minúsculas, dígitos e ``_`` (por exemplo ``"extra_trees"``).

        Returns:
            O decorador.
        """

        def decorator(component: C) -> C:
            self.add(name, cast("T", component))
            return component

        return decorator

    def add(self, name: str, component: T) -> T:
        """Regista ``component`` com o nome ``name`` e devolve-o.

        Raises:
            RegistryError: Se o nome for inválido ou se não for possível ler os tipos
                de tarefa do componente.
            DuplicateComponentError: Se o nome já estiver ocupado.
        """
        if not _NAME_PATTERN.fullmatch(name):
            raise RegistryError(
                f"Nome inválido '{name}' no registo '{self._name}': usa só minúsculas, "
                "dígitos e '_', a começar por uma letra (por exemplo 'extra_trees')."
            )
        task_types = self._extract_task_types(name, component)
        with self._lock:
            if name in self._entries:
                raise DuplicateComponentError(self._name, name)
            self._entries[name] = _Entry(component, task_types)
        return component

    def get(self, name: str) -> T:
        """Devolve o componente registado com o nome ``name``.

        Raises:
            ComponentNotFoundError: Se não existir; a mensagem sugere o nome mais parecido.
        """
        self._ensure_discovered()
        entry = self._entries.get(name)
        if entry is None:
            raise ComponentNotFoundError(self._name, name, self._entries)
        return entry.component

    def task_types_of(self, name: str) -> frozenset[TaskType]:
        """Tipos de tarefa que o componente ``name`` resolve (vazio se o registo não os usa)."""
        self.get(name)
        return self._entries[name].task_types

    def for_task(self, task_type: TaskType) -> dict[str, T]:
        """Componentes que resolvem ``task_type``, ordenados pelo nome.

        Raises:
            RegistryError: Se o registo não tiver sido criado com ``task_types``.
        """
        if self._read_task_types is None:
            raise RegistryError(
                f"O registo '{self._name}' não está organizado por tipo de tarefa: "
                "cria-o com o argumento task_types para poder filtrar."
            )
        self._ensure_discovered()
        return {
            name: entry.component
            for name, entry in sorted(self._entries.items())
            if task_type in entry.task_types
        }

    def names(self) -> list[str]:
        """Nomes registados, por ordem alfabética."""
        self._ensure_discovered()
        return sorted(self._entries)

    def items(self) -> list[tuple[str, T]]:
        """Pares ``(nome, componente)``, por ordem alfabética do nome."""
        self._ensure_discovered()
        return [(name, entry.component) for name, entry in sorted(self._entries.items())]

    def __contains__(self, name: object) -> bool:
        self._ensure_discovered()
        return name in self._entries

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        self._ensure_discovered()
        return len(self._entries)

    def __repr__(self) -> str:
        return f"Registry({self._name!r}, componentes={sorted(self._entries)})"

    def _extract_task_types(self, name: str, component: T) -> frozenset[TaskType]:
        if self._read_task_types is None:
            return frozenset()
        try:
            task_types = frozenset(self._read_task_types(component))
        except Exception as error:
            raise RegistryError(
                f"Não foi possível ler os tipos de tarefa de '{name}' no registo "
                f"'{self._name}': {error}"
            ) from error
        if not task_types:
            raise RegistryError(
                f"'{name}' não declara nenhum tipo de tarefa, por isso nunca apareceria "
                f"no catálogo do registo '{self._name}'."
            )
        return task_types

    def _ensure_discovered(self) -> None:
        if self._discover is None or self._discovered:
            return
        # O lock impede que dois fios descubram ao mesmo tempo. É reentrante porque um
        # módulo importado durante a descoberta pode consultar o registo; nesse caso
        # recebe o que já estiver registado, em vez de recomeçar a descoberta.
        with self._lock:
            self._discover_components(self._discover)

    def _discover_components(self, package: str) -> None:
        # Chamado com o lock: outro fio pode ter acabado a descoberta enquanto este
        # esperava, ou este mesmo fio pode estar a meio dela.
        if self._discovered or self._discovering:
            return
        self._discovering = True
        try:
            modules = import_submodules(package)
        finally:
            self._discovering = False
        self._discovered = True
        logger.debug("registo %r: %d módulos importados de %s", self._name, len(modules), package)


def import_submodules(package: str) -> list[str]:
    """Importa todos os módulos de um pacote, incluindo os dos subpacotes.

    Os componentes registam-se quando o seu módulo é importado (é aí que o
    decorador corre). Esta função importa todos os ficheiros de uma pasta, sem
    ser preciso manter uma lista à mão. Os erros de importação não são
    escondidos: propagam-se a quem chamou.

    Args:
        package: Nome do pacote, por exemplo ``"ml_lab.core.models"``.

    Returns:
        Os nomes dos módulos importados, pela ordem de importação.
    """
    module = importlib.import_module(package)
    search_path = getattr(module, "__path__", None)
    if search_path is None:  # é um módulo simples, não tem submódulos
        return []
    imported: list[str] = []
    for info in pkgutil.iter_modules(search_path, prefix=f"{package}."):
        importlib.import_module(info.name)
        imported.append(info.name)
        if info.ispkg:
            imported.extend(import_submodules(info.name))
    return imported
