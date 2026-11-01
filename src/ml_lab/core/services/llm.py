"""O serviço dos agentes de IA: se o modelo de linguagem está pronto, e o que falta fazer.

O Ollama é opcional (ADR 0042): sem ele, o laboratório funciona como sempre e os agentes de IA
ficam desligados. O ``status`` diz à interface se os pode ligar e, se não, o que fazer, em
avisos com um código estável. Não lança erro quando o Ollama não está a correr: isso é um
estado, não uma falha do pedido.
"""

from dataclasses import dataclass

from ml_lab.config import Settings
from ml_lab.core.foundation.errors import LanguageModelError
from ml_lab.core.foundation.notices import Notice
from ml_lab.core.llm import LanguageModel
from ml_lab.core.llm.ollama import OllamaServer
from ml_lab.core.schemas.serialization import internal
from ml_lab.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class LanguageModelStatus:
    """O estado do modelo de linguagem dos agentes de IA.

    Attributes:
        ready: Se os agentes de IA podem trabalhar (o Ollama responde e tem o modelo).
        running: Se o Ollama responde.
        url: Onde o Ollama devia estar.
        model: O modelo configurado (``ML_LAB_OLLAMA_MODEL``).
        installed: Se o Ollama tem o modelo configurado.
        version: A versão do Ollama, quando responde.
        models: Os modelos que o Ollama tem.
        notices: O que falta fazer, com um código estável (``ollama_not_running``,
            ``model_not_installed``).
    """

    ready: bool
    running: bool
    url: str
    model: str
    installed: bool
    version: str | None
    models: tuple[str, ...]
    notices: tuple[Notice, ...] = ()


class LanguageModelService:
    """O modelo de linguagem dos agentes de IA (o Ollama).

    Args:
        settings: As definições (o endereço do Ollama e o modelo).
        server: O servidor a usar; por omissão, o Ollama do endereço das definições.
    """

    def __init__(self, settings: Settings, *, server: OllamaServer | None = None) -> None:
        self._model = settings.ollama_model
        self._server = server if server is not None else OllamaServer(settings.ollama_url)

    @property
    def model_name(self) -> str:
        """O modelo configurado ("qwen3:8b"), sem perguntar nada ao Ollama."""
        return self._model

    @internal
    def model(self) -> LanguageModel:
        """O modelo de linguagem configurado, pronto a conversar (serve o serviço dos agentes).

        Raises:
            LanguageModelError: Se o Ollama não estiver pronto; a mensagem diz o que fazer.
        """
        status = self.status()
        if not status.ready:
            raise LanguageModelError(" ".join(notice.message for notice in status.notices))
        return self._server.model(self._model)

    def status(self) -> LanguageModelStatus:
        """Se os agentes de IA podem trabalhar: o Ollama, o modelo e o que falta fazer."""
        url = self._server.url
        try:
            info = self._server.info()
        except LanguageModelError as error:
            logger.debug("o Ollama não responde", extra={"url": url, "error": str(error)})
            return LanguageModelStatus(
                ready=False,
                running=False,
                url=url,
                model=self._model,
                installed=False,
                version=None,
                models=(),
                notices=(
                    Notice.warning(
                        "ollama_not_running",
                        f"O Ollama não está a correr em {url}. Para ligar os agentes de IA, abre "
                        f"a aplicação do Ollama (ou corre «ollama serve») e descarrega o modelo "
                        f"com «ollama pull {self._model}». O resto do laboratório funciona "
                        "sem ele.",
                    ),
                ),
            )
        installed = has_model(info.models, self._model)
        notices: tuple[Notice, ...] = ()
        if not installed:
            found = ", ".join(info.models) or "nenhum"
            notices = (
                Notice.warning(
                    "model_not_installed",
                    f"O Ollama está a correr, mas não tem o modelo {self._model}: descarrega-o "
                    f"com «ollama pull {self._model}» (só da primeira vez). Os modelos que tem: "
                    f"{found}.",
                ),
            )
        return LanguageModelStatus(
            ready=installed,
            running=True,
            url=url,
            model=self._model,
            installed=installed,
            version=info.version,
            models=info.models,
            notices=notices,
        )


def has_model(models: tuple[str, ...], wanted: str) -> bool:
    """Se ``wanted`` está entre os modelos (sem etiqueta, o Ollama lê ``:latest``)."""
    name = wanted if ":" in wanted else f"{wanted}:latest"
    return name in models or wanted in models
