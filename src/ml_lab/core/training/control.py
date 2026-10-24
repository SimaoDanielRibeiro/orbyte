"""Controlo do treino: pausar, retomar e parar, a partir de outro fio de execução.

O treino corre num fio (thread) e a interface noutro. O controlo é a ponte entre os
dois: a interface carrega em "Pausar" e o modelo, no fim do passo seguinte, espera;
carrega em "Cancelar" e o modelo pára no fim do passo seguinte, com o que já
aprendeu. Ninguém interrompe o modelo a meio de um passo.
"""

import threading
from typing import Final

_POLL_SECONDS: Final = 0.05


class TrainingControl:
    """Pedidos de pausa e de paragem para um treino em curso (seguro entre fios)."""

    def __init__(self) -> None:
        self._running = threading.Event()
        self._running.set()
        self._stop = threading.Event()

    @property
    def is_paused(self) -> bool:
        """Se foi pedida uma pausa (e ainda não se retomou)."""
        return not self._running.is_set()

    @property
    def stop_requested(self) -> bool:
        """Se foi pedido que o treino pare."""
        return self._stop.is_set()

    def pause(self) -> None:
        """Pede ao treino que espere no fim do passo atual."""
        self._running.clear()

    def resume(self) -> None:
        """Retoma um treino em pausa."""
        self._running.set()

    def stop(self) -> None:
        """Pede ao treino que pare no fim do passo atual (também acaba com uma pausa)."""
        self._stop.set()
        self._running.set()

    def wait_while_paused(self) -> None:
        """Bloqueia enquanto o treino estiver em pausa (e não tiver sido mandado parar)."""
        while not self._running.wait(_POLL_SECONDS):
            if self._stop.is_set():
                return
